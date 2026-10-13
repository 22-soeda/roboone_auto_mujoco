# -*- coding: utf-8 -*-
"""片脚の逆運動学 (IK)。足裏中心の位置・姿勢から 6 関節の角度を出す。反復なしの閉形式解。

ros2_ws から写したコードと値（ws とシミュレータは互いを参照しないので、コピーして持つ）:
    解法  scripts/leg_ik.py の ik()                                     ws commit 17df77a
    寸法  src/roboone_kinematics/include/roboone_kinematics/leg_config.hpp   同上
_solve() の中身は ws と行ごとに見比べられるよう、変数名・式番号・コメントをそのまま残した。
ws 側で式や寸法が変わったら、ここも写し直す。

座標系は 2 つある。
    外側（ik() の入出力と model/auto.xml）は機体座標 Σ_B。x = 前、y = 左、z = 上。原点は骨盤。
    解法の内側は導出文書の Σ_S。x = 右、y = 前、z = 上。ws の Python 版がこの座標系で書かれている。
読み替えは ik() の入口と出口の 1 か所だけで行う（ws の C++ 版と同じ分け方）。

ws にあって、ここに持ってこなかったもの:
    AXIS_FLIP     サーボの回転方向。シミュレータの関節は Σ_B の向きのままなので要らない
    fk() ほか     順運動学は MuJoCo が計算する
    自己検算      tests/test_leg_ik.py で model/auto.xml と突き合わせる形に置き換えた
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 寸法（leg_config.hpp の右脚の値。Σ_B、mm を m に直した）
# ---------------------------------------------------------------------------
L3 = 0.1425        # 股中心 o3 -> 膝軸 o4（大腿）
L4 = 0.141021      # 膝軸 o4 -> 足首ロール軸 o5（下腿）
L5 = 0.007924      # 足首ロール軸 o5 -> 足首ピッチ軸 o6
L6 = 0.0085        # 足首ピッチ軸 o6 -> 足裏中心
# P3_Y・P4_Y は膝軸の上のどこに膝の点を取るかで、IK に効くのは和だけ（ここでは 0）。
# model/auto.xml は膝を脚の面の上（どちらも 0）に置いているが、IK は hpp の値のまま写した
P3_Y = 0.014642    # 大腿で膝の点が内側へずれる量
P4_Y = -0.014642   # 下腿で同じだけ戻る量。足首は股の真下に来る
P3_X = 0.0         # 大腿の前後オフセット
HIP_Y = -0.0893    # 骨盤 -> 右の股中心（左右）
KNEE_FORWARD = +1  # 人型。膝が前に出る

#: ik() が返す関節角の並び。ws の J1〜J6 と同じ順で、名前は model/auto.xml の関節に合わせた。
#: J5・J6 は ws の enum ではそれぞれ ANKLE_PITCH・ANKLE_ROLL（サーボの呼び名）だが、
#: Σ_B での回転軸は J5 が x（ロール）、J6 が y（ピッチ）。
JOINT_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")

# ---------------------------------------------------------------------------
# Σ_B <-> Σ_S
# ---------------------------------------------------------------------------
#: Σ_B の成分 = B_FROM_S @ Σ_S の成分
B_FROM_S = np.array([[0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])
#: Σ_S で x まわりの関節（J1・J4・J6）は、Σ_B では -y まわりになるので角の符号が反転する
ANGLE_SIGN_S_TO_B = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])


class Unreachable(ValueError):
    """目標が脚の到達範囲の外（文書 §5.6「到達不能の判定」）。"""


@dataclass(frozen=True)
class _SolverLeg:
    """_solve() が読む片脚の定数。Σ_S の成分で、名前は ws の LegParams と同じ。"""

    l3e: float         # 有効大腿長 ℓ3' = hypot(ℓ3, b)
    l4: float
    l5: float
    a: float           # p3, p4 の x 成分の和。膝軸方向のずれはこの和だけが効く
    phi: float         # 膝角オフセット φ = atan2(b, ℓ3)
    sigma: int         # 膝の分岐。Σ_B の KNEE_FORWARD とは符号が逆
    p0: np.ndarray     # 骨盤 -> 股中心 o3
    p6: np.ndarray     # 足首ピッチ軸 o6 -> 足裏中心


def _solver_leg(side: str) -> _SolverLeg:
    """leg_config.hpp の値（Σ_B・右脚）から、side の脚の Σ_S の定数を作る。"""
    # 左脚は Σ_B の y を反転する。Σ_S では x（= 膝軸方向）の反転になる
    lateral = {"right": 1.0, "left": -1.0}[side]
    hip_center_B = np.array([0.0, lateral * HIP_Y, 0.0])
    return _SolverLeg(
        l3e=math.hypot(L3, P3_X),
        l4=L4,
        l5=L5,
        a=lateral * (-P3_Y - P4_Y),
        phi=math.atan2(P3_X, L3),
        sigma=-KNEE_FORWARD,
        p0=B_FROM_S.T @ hip_center_B,
        p6=np.array([0.0, 0.0, -L6]),
    )


_LEGS = {side: _solver_leg(side) for side in ("right", "left")}


def ik(side: str, sole_pos, sole_rot, *, clamp: bool = True) -> np.ndarray:
    """足裏中心の位置・姿勢 -> 片脚 6 関節の角度 [rad]。

    side      : "right" か "left"
    sole_pos  : 骨盤から見た足裏中心の位置 [m]（Σ_B）
    sole_rot  : 足裏の姿勢（Σ_B の 3x3 回転行列）。水平なら単位行列
    clamp     : True なら届かない目標を最寄りの姿勢に丸める。False なら Unreachable を送出
    戻り値    : JOINT_NAMES の順の関節角。model/auto.xml の関節にそのまま入れられる
    """
    leg = _LEGS[side]
    p_S = B_FROM_S.T @ np.asarray(sole_pos, dtype=float).reshape(3)
    R_S = B_FROM_S.T @ np.asarray(sole_rot, dtype=float).reshape(3, 3) @ B_FROM_S
    theta_S = _solve(leg, p_S, R_S, clamp)
    return theta_S * ANGLE_SIGN_S_TO_B


def Rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def Ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def _solve(prm: _SolverLeg, p: np.ndarray, R: np.ndarray, clamp: bool) -> np.ndarray:
    """ws の leg_ik.ik() の本体。Σ_S の (p, R) -> 文書の符号での関節角 θ1..θ6。

    ws から変えたのは 2 点だけ: 引数の受け取り方と、最後に AXIS_FLIP を掛けないこと。
    式番号は ws の roboone_kinematics/leg_kinematics.hpp の「x 成分を入れた導出」に対応。
    """
    l3, l4, l5, a = prm.l3e, prm.l4, prm.l5, prm.a

    # 1. 股中心 -> 足首ロール軸 を Σ_6 で見たベクトル (IK-1)
    r = R.T @ (p - prm.p0) - prm.p6
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])

    # 2. (X-6)
    K = float(r @ r) - a * a - l3 * l3 + l4 * l4 - l5 * l5

    # 3. θ5 についての線形式 (X-7):  P·s5 + Q·c5 = C
    P = K
    Q = 2.0 * (a * l4 - l5 * rx)
    C = 2.0 * (l4 * rx - a * l5)
    Rm = math.hypot(P, Q)
    if Rm < 1e-12:
        raise Unreachable("(X-7) が退化している (P = Q = 0)")
    ratio = C / Rm
    exact = abs(ratio) <= 1.0
    if not exact and not clamp:
        raise Unreachable(f"|C|/|(P,Q)| = {abs(ratio):.6g} > 1: x 方向に遠すぎる")
    beta = math.asin(max(-1.0, min(1.0, ratio)))
    psi = math.atan2(Q, P)

    # 4. 2 つの根から枝を選ぶ。A > 0 のうち膝が範囲内、次に cosθ5 が大きい方
    best = None
    for t5 in (beta - psi, math.pi - beta - psi):
        c5, s5 = math.cos(t5), math.sin(t5)
        # (X-8)。分母 2(ℓ4 + ℓ5·c5) は ℓ4 > ℓ5 なので 0 にならない
        A = (K + 2.0 * l5 * a * s5) / (2.0 * (l4 + l5 * c5))
        if A <= 0.0:
            continue                      # 文書 §5.6「A > 0 の仮定」
        c4 = (A - l4) / l3
        over = abs(c4) > 1.0
        if best is None or (best[0] and not over) or (over == best[0] and c5 > best[1]):
            best = (over, c5, s5, t5, A, max(-1.0, min(1.0, c4)))
    if best is None:
        raise Unreachable("A > 0 の枝が無い: 足先が股中心に近すぎる")
    over, c5, s5, t5, A, c4 = best
    if over and not clamp:
        raise Unreachable(f"cosθ4 が範囲外: 遠すぎるか近すぎる")

    # 5. 膝 (IK-9)
    t4e = prm.sigma * math.acos(c4)
    B = l3 * math.sin(t4e)
    t4 = t4e + prm.phi                    # 本来の θ4（文書 §6）

    # 6. 足首ロール (X-9)
    V = a * s5 - A * c5 - l5
    t6 = math.atan2(V * ry + B * rz, V * rz - B * ry)

    # 7. 残った回転から股 3 軸。M = R123 = Rx(θ1)·Ry(θ2)·Rz(θ3) の x-y-z 順で取り出す
    #    （文書 (IK-12)〜(IK-14) は Ry·Rx·Rz 前提なので式が違う）
    M = R @ Rx(-t6) @ Ry(-t5) @ Rx(-t4)
    t2 = math.atan2(M[0, 2], math.hypot(M[0, 0], M[0, 1]))
    t3 = math.atan2(-M[0, 1], M[0, 0])
    t1 = math.atan2(-M[1, 2], M[2, 2])

    return np.array([t1, t2, t3, t4, t5, t6])
