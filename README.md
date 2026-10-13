# roboone_auto_mujoco

ROBO-ONE Auto 2026 の二足を MuJoCo の物理シミュレーションに載せて、実機を動かす前に
歩行計画が正しいかを確かめるためのリポジトリ。歩行計画が出す関節角をそのまま位置サーボに
流し、転ばないか、ZMP が足裏に収まるか、足が滑らないかを見る。機械学習はやらない。

## ros2_ws との関係

歩行計画・IK・実機を動かすコードは [ros2_ws](https://github.com/22-soeda/ros2_ws) にある。
ws は ROS 2 の上にあるので、このリポジトリはそこから切り離して置く。ws とこのリポジトリは互いを参照しない。

- ws の値やコードが要るときは、このリポジトリにコピーして持つ。コードは `ws_copy/` に置く
- 写したものには、写し元の ws のファイルと commit を書いておく。ws 側で変わったら写し直す

| 写したもの | 写し元（ws commit 17df77a） |
|---|---|
| `ws_copy/leg_ik.py` 片脚の IK | `scripts/leg_ik.py` の `ik()` と、`leg_config.hpp` の寸法 |
| `model/auto.xml` の寸法 | `src/roboone_kinematics/include/roboone_kinematics/leg_config.hpp` |
| `model/auto.xml` のホーム姿勢 | `src/roboone_motion/config/home_pose.yaml` を ws の IK に通した値 |

## 準備（1 回）

Windows の開発 PC、Python 3.14、CPU だけで回す。

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## モデルをビューアで開く

```bash
.venv/Scripts/python.exe -m mujoco.viewer --mjcf=model/auto.xml          # 機体
.venv/Scripts/python.exe -m mujoco.viewer --mjcf=examples/pendulum.xml   # 練習用の振り子
```

`model/auto.xml` は ROBO-ONE Auto 2026 の機体モデル。寸法は ws の `leg_config.hpp` から写した。
関節は理想化してあり、トルク上限のない硬い位置サーボで、指令角にほぼそのまま追従する（誤差 0.05° 以内）。
モータやリンクの事情と切り分けて、歩行計画そのものが正しいかを確かめるため。
右パネルの Control で関節ごとに指令を出せる。左パネル Simulation の Key を 0 にして Load key を押すと、
ws のホーム姿勢（高さ 261 mm）から始まる。質量は CAD の値がまだ入っていないので、立ち方の数字には意味がない。

## テスト

```bash
.venv/Scripts/python.exe -m pytest tests
```

`tests/test_leg_ik.py` は、IK の出した関節角を `model/auto.xml` に入れて足裏が狙いどおりに来るかを見る。
IK とモデルは同じ寸法を別々に写しているので、片方だけ写し直したときの食い違いもここで見つかる。

## 練習用のモデル

`examples/` は MuJoCo の操作を覚えるための練習用モデル。`pendulum.xml` はサーボ 1 個で動く振り子で、
右パネルの Control で指令を出すと、重力で目標から垂れる様子やトルク上限の効き方が見られる。
