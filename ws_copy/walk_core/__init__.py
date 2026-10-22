# -*- coding: utf-8 -*-
# ros2_ws の src/roboone_motion/roboone_motion/walk_core/__init__.py を写したもの（ws commit 17df77a）。
# 中身は書き換えていない。ws 側で変わったら、このファイルごと写し直す。
"""walk_core — ROS 非依存の歩行計画ライブラリ。

docs/ros2_walk_implementation.pdf の設計に従う。時計も乱数も持たず、
update(vx, vy, dt) の入力列だけで決定的に動く。
"""

from .engine import (ESTOP, IDLE, LEFT, RIGHT, START, STEP, StepRecord, STOP,
                     WalkEngine, WalkOutputs)
from .params import GaitParams

__all__ = [
    'GaitParams', 'WalkEngine', 'WalkOutputs', 'StepRecord',
    'IDLE', 'START', 'STEP', 'STOP', 'ESTOP', 'LEFT', 'RIGHT',
]
