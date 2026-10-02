# 旋風丸 (Senpumaru) — 外乱耐性直立制御

20自由度の小型二足ロボット『旋風丸』が、両足を接地したまま（歩行・踏み替えなし）
外乱を受けても直立を維持する方策を、MuJoCo MJX + Brax PPO で学習するリポジトリです。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [`docs/master_plan.md`](docs/master_plan.md) | 計画・MDP仕様・Gate 判定基準（正本） |
| [`docs/status.md`](docs/status.md) | 現在地・直近の変更・次のステップ |
| [`docs/current.md`](docs/current.md) | 現行コードの構成と契約の要約 |
| [`robot/config.py`](robot/config.py) | ハイパーパラメータ・報酬重み・観測次元の正本 |

## 構成

```
envs/      MJX 環境 (mjx_env.py)、報酬 (mjx_rewards.py)、安定性指標、CBF 安全フィルタ、
           サーボ熱・電圧モデル、学習用ラッパー (training_wrapper.py)
robot/     設定 (config.py)、方策ネットワークと checkpoint 読み込み (policy_network.py)、数学関数
train/     学習 (train_mjx.py)、可視化 (visualize_rl.py)、ONNX 出力、軌跡出力
scratch/   Gate 0 / A / B の評価・判定スクリプト、診断ツール
scripts/   MuJoCo モデル（当たり判定・質量）の調整ツール
real/      実機 (Raspberry Pi 5 + Teensy 4.1) の制御ループと I/O
assets/    MuJoCo モデル (humanoid/humanoid.xml) とメッシュ
tests/     pytest
```

## よく使うコマンド (WSL2)

```bash
# テスト
./venv_wsl/bin/python -m pytest tests/ -v

# 学習 (未コミットの変更があると起動しない。--allow_dirty で明示的に許可)
./venv_wsl/bin/python train/train_mjx.py --exp_name gate_a --seed 0

# Gate A 評価と判定
./venv_wsl/bin/python scratch/phase0_eval_diagnostics.py --exp_name gate_a --version 0 \
    --episodes 200 --out log/gate_a/version_0/gate_a_report.json
./venv_wsl/bin/python scratch/gate_a_qualification.py --reports log/gate_a/version_*/gate_a_report.json --threshold 0.95

# 可視化 / 実機用 ONNX 出力
./venv_wsl/bin/python train/visualize_rl.py --exp_name gate_a --mode plot
./venv_wsl/bin/python train/export_onnx.py --model log/gate_a/version_0/best_params.pkl --output policy.onnx
```
