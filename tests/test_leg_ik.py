# -*- coding: utf-8 -*-
"""ws_copy/leg_ik.py を model/auto.xml と突き合わせる。

IK と MJCF は同じ寸法（leg_config.hpp）を別々に写しているので、片方だけ写し直すと食い違う。
ここが通れば、IK の出した関節角をモデルに入れると足裏が狙いどおりに来る。
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from ws_copy.leg_ik import JOINT_NAMES, Unreachable, ik

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "auto.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def leg_qpos_addresses(model, side):
    return [model.jnt_qposadr[model.joint(f"{side}_{name}").id] for name in JOINT_NAMES]


def sole_pose_from_pelvis(model, data, side):
    """骨盤から見た足裏中心の位置と姿勢。骨盤は qpos0 で回転していないので、世界座標の差でよい。"""
    mujoco.mj_kinematics(model, data)
    pos = data.site(f"{side}_sole").xpos - data.body("pelvis").xpos
    rot = data.site(f"{side}_sole").xmat.reshape(3, 3).copy()
    return pos, rot


@pytest.mark.parametrize("side", ["right", "left"])
def test_ik_puts_sole_on_target(model, side):
    """届く狙いを FK で作り、IK の関節角をモデルに入れて、足裏が狙いに戻るかを見る。"""
    data = mujoco.MjData(model)
    addresses = leg_qpos_addresses(model, side)
    rng = np.random.default_rng(0)
    # 膝は伸び切り（特異点）と深い曲げを避ける。その外は IK の枝選びで別の姿勢が選ばれうる
    lowest = np.radians([-45, -30, -30, 10, -30, -30])
    highest = np.radians([45, 30, 30, 120, 30, 30])

    for _ in range(200):
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = rng.uniform(lowest, highest)
        target_pos, target_rot = sole_pose_from_pelvis(model, data, side)

        data.qpos[:] = model.qpos0
        data.qpos[addresses] = ik(side, target_pos, target_rot, clamp=False)
        reached_pos, reached_rot = sole_pose_from_pelvis(model, data, side)

        np.testing.assert_allclose(reached_pos, target_pos, atol=1e-12)
        np.testing.assert_allclose(reached_rot, target_rot, atol=1e-12)


def test_home_keyframe_matches_ik(model):
    """model/auto.xml のホーム姿勢は ws の home_pose.yaml（高さ 261・前後 -20 mm・足裏水平）を IK に通した値。"""
    home_ctrl = model.key("home").ctrl
    for side, hip_y in (("right", -0.0893), ("left", 0.0893)):
        angles = ik(side, [-0.020, hip_y, -0.261], np.eye(3), clamp=False)
        first = model.actuator(f"{side}_hip_pitch").id
        np.testing.assert_allclose(home_ctrl[first:first + 6], angles, atol=1e-5)


def test_far_target_is_unreachable():
    """脚の全長 0.3 m より遠い狙いは、clamp=False なら Unreachable になる。"""
    with pytest.raises(Unreachable):
        ik("right", [0.0, -0.0893, -0.35], np.eye(3), clamp=False)
