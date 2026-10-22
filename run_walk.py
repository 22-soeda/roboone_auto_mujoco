# -*- coding: utf-8 -*-
"""歩行計画を model/auto.xml に流して、転ばずに歩けるかを見る。

walk_core（ws から写した歩行計画）が 5 ms ごとに出す足裏の目標を、IK で関節角に直して、
理想化した位置サーボに渡す。IMU のフィードバックも、脚のたわみの補正も入れない、
理論式だけの純フィードフォワード。実機の motion ノードが歩行中にやっている
「walk_core の足裏（骨盤から見た位置）-> IK」と同じ順に通している。

使い方:
    .venv/Scripts/python.exe run_walk.py                  # ビューアで見る（実時間）
    .venv/Scripts/python.exe run_walk.py --no-viewer      # 数字だけ出す
    .venv/Scripts/python.exe run_walk.py --record out/walk.mp4          # mp4 に書き出す（ffmpeg が要る）
    .venv/Scripts/python.exe run_walk.py --touchdown ws --no-viewer     # 着地まわりを実機と同じにする
    .venv/Scripts/python.exe run_walk.py --vx 0.1 --vy 0.0 --walk-until 4.5 --end 8
"""

import argparse
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ws_copy.leg_ik import JOINT_NAMES, ik
from ws_copy.walk_core import GaitParams, WalkEngine

MODEL_PATH = Path(__file__).resolve().parent / "model" / "auto.xml"
SIDES = ("right", "left")
WALK_START_S = 0.5          # 指令を入れ始める時刻。ws の walk_dump と同じ
FALLEN_HEIGHT_RATIO = 0.5   # 骨盤が z_c のこの割合より下がったら転倒とみなす
FALLEN_TILT_DEG = 45.0
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
VIDEO_FPS = 30              # 実時間で書き出す

#: 着地まわりの 2 つの設定の組み合わせ。値は (td_overdrive [m], td_speed_max [m/s])。
#:   td_overdrive  遊脚を名目の床より 4 mm 下まで下ろす
#:   td_speed_max  遊脚の降下速度の上限 0.1 m/s。実機の衝撃対策
#: 上限があると 5 次曲線が途中で出したい 0.17 m/s に届かず、歩の終わりでも遊脚が床から 1.6〜1.9 mm
#: 浮いたまま支持脚が入れ替わる。前後・横・斜めの指令 6 通りで、上限のない pure（と突き抜けだけ）は
#: 計画どおり歩いた（骨盤のずれ最大 20〜100 mm、向きの変化 4° 以下）。上限のある ws は転ばないが
#: 機体が 16〜87° 回って計画から 33〜44 cm ずれ、limit-only は 5 通りで転んだ。
TOUCHDOWN_SETTINGS = {
    "pure": (0.0, math.inf),         # 理論式だけ。遊脚は純粋な 5 次曲線で降りる
    "ws": (0.004, 0.10),             # 実機と同じ（gait.yaml の値）
    "limit-only": (0.0, 0.10),       # 上限だけ残す比較用。実機には無い組み合わせ
}
TOUCHDOWN_CHOICES = tuple(TOUCHDOWN_SETTINGS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vx", type=float, default=0.10, help="前後の速度指令 [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="左右の速度指令 [m/s]（左が正）")
    parser.add_argument("--walk-until", type=float, default=4.5, help="指令を 0 に戻す時刻 [s]")
    parser.add_argument("--end", type=float, default=8.0, help="評価を締める時刻 [s]")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアを開かずに数字だけ出す")
    parser.add_argument("--record", type=Path, help="ビューアを開かずに、この mp4 に書き出す")
    parser.add_argument("--touchdown", choices=TOUCHDOWN_CHOICES, default="pure",
                        help="着地まわりの設定。pure = 純 5 次曲線 / ws = 実機と同じ / limit-only = 上限だけ（比較用）")
    return parser.parse_args()


def gait_with_touchdown(touchdown: str) -> GaitParams:
    """gait.yaml と同じ値のうち、着地まわりの 2 つだけを touchdown の組み合わせにしたもの。"""
    overdrive, descent_limit = TOUCHDOWN_SETTINGS[touchdown]
    return replace(GaitParams(), td_overdrive=overdrive, td_speed_max=descent_limit)


def command_at(t: float, args) -> tuple:
    walking = WALK_START_S <= t < args.walk_until
    return (args.vx, args.vy) if walking else (0.0, 0.0)


def actuator_ids_by_side(model) -> dict:
    """side -> JOINT_NAMES の順に並べたアクチュエータ番号。"""
    return {side: [model.actuator(f"{side}_{name}").id for name in JOINT_NAMES] for side in SIDES}


def joint_targets(walk_out, actuator_ids: dict, num_actuators: int) -> np.ndarray:
    """walk_core の 1 周期分の出力 -> 12 軸の指令角。足裏は水平のまま（実機の歩行と同じ）。"""
    targets = np.zeros(num_actuators)
    feet = walk_out.foot_targets_pelvis()
    for side in SIDES:
        targets[actuator_ids[side]] = ik(side, feet[side], np.eye(3), clamp=True)
    return targets


def place_at_start(model, data, engine, gait, actuator_ids):
    """歩行計画の最初の姿勢（両足支持で静止）に機体を置く。walk_core の世界座標 = シミュレータの世界座標。"""
    first = engine.update(0.0, 0.0, 1.0 / gait.loop_hz)
    targets = joint_targets(first, actuator_ids, model.nu)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = first.pelvis              # 骨盤 = 重心の水平位置と z_c。足裏がちょうど床に着く
    for side in SIDES:
        for actuator_id in actuator_ids[side]:
            joint_id = model.actuator_trnid[actuator_id, 0]
            data.qpos[model.jnt_qposadr[joint_id]] = targets[actuator_id]
    data.ctrl[:] = targets
    mujoco.mj_forward(model, data)


def pelvis_tilt_deg(data) -> float:
    z_axis_z = data.body("pelvis").xmat[8]
    return math.degrees(math.acos(max(-1.0, min(1.0, z_axis_z))))


def pelvis_yaw_deg(data) -> float:
    """骨盤の向き（z まわり）。walk_core は向きを変えないので、計画では常に 0。"""
    x_axis = data.body("pelvis").xmat.reshape(3, 3)[:, 0]
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


class WalkReport:
    """歩いている間の数字を集める。転倒・骨盤の追従・向きの変化・圧力中心のずれ・足の滑りを見る。

    転倒だけでは足りない。立ったまま回って計画からずれても、転倒の判定には掛からない。
    """

    def __init__(self, model, gait):
        self.model = model
        self.gait = gait
        self.weight_n = float(model.body_subtreemass[model.body("pelvis").id]) * gait.gravity
        self.floor_id = model.geom("floor").id
        self.foot_body_ids = {side: model.body(f"{side}_foot").id for side in SIDES}
        self.fallen_at = None
        self.worst_pelvis_error = 0.0
        self.worst_yaw_deg = 0.0
        self.cop_errors = []
        self.slip_anchor = {side: None for side in SIDES}
        self.worst_slip = {side: 0.0 for side in SIDES}

    def record(self, data, walk_out):
        if self.fallen_at is not None:
            return
        if self._has_fallen(data):
            self.fallen_at = data.time
            return
        planned_pelvis = np.array(walk_out.pelvis[:2])
        self.worst_pelvis_error = max(self.worst_pelvis_error,
                                      float(np.linalg.norm(data.body("pelvis").xpos[:2] - planned_pelvis)))
        self.worst_yaw_deg = max(self.worst_yaw_deg, abs(pelvis_yaw_deg(data)))
        normal_force, cop = self._floor_contact(data)
        if sum(normal_force.values()) > 0.5 * self.weight_n:
            self.cop_errors.append(float(np.linalg.norm(cop - np.array(walk_out.zmp))))
        self._track_slip(data, normal_force)

    def _has_fallen(self, data) -> bool:
        too_low = data.body("pelvis").xpos[2] < FALLEN_HEIGHT_RATIO * self.gait.z_c
        return too_low or pelvis_tilt_deg(data) > FALLEN_TILT_DEG

    def _floor_contact(self, data):
        """足ごとの床からの垂直抗力 [N] と、床の上の圧力中心 (x, y)。"""
        normal_force = {side: 0.0 for side in SIDES}
        moment = np.zeros(2)
        force_6d = np.zeros(6)
        for i, contact in enumerate(data.contact[:data.ncon]):
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            body = self.model.geom_bodyid[other]
            mujoco.mj_contactForce(self.model, data, i, force_6d)
            pressing = force_6d[0]   # 接触座標の第 1 軸が法線
            moment += pressing * contact.pos[:2]
            for side, foot_id in self.foot_body_ids.items():
                if body == foot_id:
                    normal_force[side] += pressing
        total = sum(normal_force.values())
        cop = moment / total if total > 0 else np.full(2, np.nan)
        return normal_force, cop

    def _track_slip(self, data, normal_force):
        """しっかり踏んでいる（体重の 2 割以上）間に、足裏中心が床の上でどれだけずれたか。"""
        for side in SIDES:
            sole_xy = data.site(f"{side}_sole").xpos[:2].copy()
            if normal_force[side] < 0.2 * self.weight_n:
                self.slip_anchor[side] = None
                continue
            if self.slip_anchor[side] is None:
                self.slip_anchor[side] = sole_xy
            drift = float(np.linalg.norm(sole_xy - self.slip_anchor[side]))
            self.worst_slip[side] = max(self.worst_slip[side], drift)

    def print_summary(self, engine):
        print(f"歩数 {engine.step_idx} / 最後の状態 {engine.state}")
        if self.fallen_at is not None:
            print(f"転倒した: {self.fallen_at:.2f} s")
        else:
            print("転倒しなかった")
        print(f"骨盤の水平位置と計画の重心のずれ: 最大 {self.worst_pelvis_error * 1000:.1f} mm")
        print(f"骨盤の向きの変化（計画では 0）: 最大 {self.worst_yaw_deg:.1f}°")
        if self.cop_errors:
            errors_mm = np.array(self.cop_errors) * 1000
            print(f"圧力中心と計画の ZMP のずれ: 平均 {errors_mm.mean():.1f} mm / 最大 {errors_mm.max():.1f} mm")
        for side in SIDES:
            print(f"{side} 足の滑り（踏んでいる間のずれ）: 最大 {self.worst_slip[side] * 1000:.1f} mm")


def control_tick(model, data, engine, gait, actuator_ids, args):
    """制御 1 周期 = walk_core を 1 回進め、指令を出して、物理を周期ぶん進める。

    指令は周期の中で前の値から直線でつなぐ。理想化したサーボは硬いので、5 ms ごとに
    階段状に変えると、変わった瞬間に大きなトルクが出て機体が跳ねる（0.02° の段差で
    約 3.5 N·m。体重 1.6 kg の機体が床から浮いた）。
    """
    control_dt = 1.0 / gait.loop_hz
    walk_out = engine.update(*command_at(data.time, args), control_dt)
    previous_targets = data.ctrl.copy()
    next_targets = joint_targets(walk_out, actuator_ids, model.nu)
    substeps = round(control_dt / model.opt.timestep)
    for k in range(1, substeps + 1):
        data.ctrl[:] = previous_targets + (next_targets - previous_targets) * (k / substeps)
        mujoco.mj_step(model, data)
    return walk_out


class WalkSimulation:
    """機体・歩行計画・記録をまとめたもの。tick() で制御 1 周期ぶん進み、--end までは記録する。"""

    def __init__(self, args):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.gait = gait_with_touchdown(args.touchdown)
        self.engine = WalkEngine(self.gait)
        self.actuator_ids = actuator_ids_by_side(self.model)
        place_at_start(self.model, self.data, self.engine, self.gait, self.actuator_ids)
        self.report = WalkReport(self.model, self.gait)

    def tick(self):
        walk_out = control_tick(self.model, self.data, self.engine, self.gait, self.actuator_ids, self.args)
        if self.data.time < self.args.end:
            self.report.record(self.data, walk_out)

    def print_summary(self):
        self.report.print_summary(self.engine)


def run_headless(sim):
    while sim.data.time < sim.args.end:
        sim.tick()
    sim.print_summary()


def camera_over_walk(args):
    """歩く道のりの真ん中を斜め前から見る固定カメラ。床に模様がないので、追いかけると進んで見えない。"""
    walking_s = args.walk_until - WALK_START_S
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (args.vx * walking_s / 2, args.vy * walking_s / 2, 0.15)
    camera.distance, camera.azimuth, camera.elevation = 1.2, 135, -15
    return camera


def record_video(sim, path: Path):
    """--end まで回し、実時間・VIDEO_FPS で描いたコマを ffmpeg に流して mp4 にする。"""
    sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = VIDEO_WIDTH, VIDEO_HEIGHT
    renderer = mujoco.Renderer(sim.model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    camera = camera_over_walk(sim.args)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(VIDEO_FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)
    next_frame_s = 0.0
    while sim.data.time < sim.args.end:
        sim.tick()
        if sim.data.time >= next_frame_s:
            renderer.update_scene(sim.data, camera=camera)
            ffmpeg.stdin.write(renderer.render().tobytes())
            next_frame_s += 1.0 / VIDEO_FPS
    ffmpeg.stdin.close()
    ffmpeg.wait()
    sim.print_summary()
    print(f"動画: {path}")


def run_with_viewer(sim):
    summary_printed = False
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        wall_start = time.perf_counter() - sim.data.time
        while viewer.is_running():
            sim.tick()
            if sim.data.time >= sim.args.end and not summary_printed:
                sim.print_summary()
                summary_printed = True
            viewer.sync()
            # 実時間に合わせる。計算のほうが速いので、追いつくまで待つ
            lag = sim.data.time - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)


def main():
    args = parse_args()
    sim = WalkSimulation(args)
    if args.record:
        record_video(sim, args.record)
    elif args.no_viewer:
        run_headless(sim)
    else:
        run_with_viewer(sim)


if __name__ == "__main__":
    main()
