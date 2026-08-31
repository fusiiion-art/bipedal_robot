# Walkthrough: Phase 1 — 基本直立の安定化

## 変更の概要

v0/v1 の学習ログ解析の結果、**ロボットが0.4秒（40ステップ）で転倒し、全エピソードで `fall_penalty = -500` が発動する完全な学習失敗**であることが判明しました。

Phase 1 では、外乱耐性以前の問題として「まず外乱なしで直立を安定させる」ための修正を実施しました。

## 変更ファイル一覧

### [config.py](file:///c:/bipedal_robot/robot/config.py) — 8箇所の修正

| パラメータ | 旧値 | 新値 | 理由 |
|-----------|------|------|------|
| `ACTION_SCALE` | ±90° | ±30° | 初期探索での暴走防止 |
| `alive` | 100.0 | 200.0 | 生存報酬を倍増し「立つだけで正報酬」保証 |
| `fall_penalty` | -500.0 | -300.0 | 転倒ペナルティ緩和で探索促進 |
| `upright` | 3.0 | 5.0 | 直立維持を最重要報酬に |
| `energy` | 0.001 | 0.0002 | ペナルティを最小化 |
| `MAX_EPISODE_STEPS` | 1000 | 500 | 短エピソードで高速学習 |
| `RANDOM_PUSH_MAX_FORCE` | 10.0 | 0.0 | 外乱完全無効化 |
| DR範囲 | 広い | 狭い | 学習環境を簡素化 |

### [mjx_env.py](file:///c:/bipedal_robot/envs/mjx_env.py) — CRITICAL-1 バグ修正

- `_get_curriculum_scale()` を `global_step` → `training_progress` ベースに変更
- `CURRICULUM_SCHEDULE` → `CURRICULUM_SCHEDULE_FRACTIONS` を使用
- 外乱スケーリングが学習全体を通じて正しく機能するようになった

### [mjx_rewards.py](file:///c:/bipedal_robot/envs/mjx_rewards.py) — CRITICAL-1 バグ修正

- `_get_curriculum_disturbance_scale()` を `training_progress` ベースに変更
- `compute()` 内の呼び出しも `training_progress` に修正

### [train_mjx.py](file:///c:/bipedal_robot/train/train_mjx.py) — reward_scaling 調整

- `reward_scaling`: 0.1 → 0.01 (alive=200.0 で報酬スケールが大きくなったため)

## 検証結果

- ✅ WSL環境でインポートテスト成功
- ✅ 報酬バランスチェック: 40ステップ生存時のネット報酬 = **+7,629.5** (旧: 負)
- ⏳ 実際の学習実行は未実施 (ユーザー実行待ち)

## 次のステップ

学習を実行して Phase 1 の効果を確認してください:

```bash
# WSL環境で実行 (GPU使用時)
wsl bash -lc "cd /mnt/c/bipedal_robot && /mnt/c/bipedal_robot/venv_wsl/bin/python train/train_mjx.py"
```

**成功基準**: `episode_alive > 200` (2秒以上生存) かつ報酬カーブが単調増加
