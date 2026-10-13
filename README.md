# roboone_auto_mujoco

ROBO-ONE Auto 2026 の二足を MuJoCo の物理シミュレーションに載せて、実機を動かす前に
歩行計画が正しいかを確かめるためのリポジトリ。歩行計画が出す関節角をそのまま位置サーボに
流し、転ばないか、ZMP が足裏に収まるか、足が滑らないかを見る。機械学習はやらない。

## ros2_ws との関係

歩行計画・IK・実機を動かすコードは [ros2_ws](https://github.com/22-soeda/ros2_ws) にある。
ws は ROS 2 の上にあるので、このリポジトリはそこから切り離して置く。

- ws のコードは import しない。ws から受け取るのは値だけで、ファイルとして持ち込む
- 持ち込んだ値には、出どころの ws のファイルと commit を書いておく

## 準備（1 回）

Windows の開発 PC、Python 3.14、CPU だけで回す。

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## モデルをビューアで開く

```bash
.venv/Scripts/python.exe -m mujoco.viewer --mjcf=examples/pendulum.xml
```

`examples/` は MuJoCo の操作を覚えるための練習用モデル。`pendulum.xml` はサーボ 1 個で動く振り子で、
右パネルの Control で指令を出すと、重力で目標から垂れる様子やトルク上限の効き方が見られる。
