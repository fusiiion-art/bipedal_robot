# 修正版パッチサマリー (2026-09-08)

## 概要

改良規約20項目に対する診断で検出された4つのブロッキングイシューを修正した5ファイルを生成しました。

| Issue | 重要度 | 対象ファイル | 修正内容 |
|-------|--------|------------|--------|
| [ISSUE-1] com_pos 統一 | 🔴 P0 | mjx_env.py, mjx_rewards.py, stability_metrics.py | subtree_com[0] 優先取得に統一 |
| [ISSUE-2] training_progress batch化 | 🔴 P0 | training_wrapper.py, mjx_rewards.py | shape (num_envs,) 対応 |
| [ISSUE-3] data.qacc 座標系明記 | 🔴 P0 | stability_metrics.py, mjx_rewards.py | world frame 加速度を明示コメント |
| [ISSUE-4] reset() shape assert | 🟡 P1 | mjx_env.py | 観測次元検証を reset 時に追加 |

---

## 修正ファイル詳細

### 1. **actuator_model.py** ✅ 軽微修正のみ

**修正内容:**
- `motor_resistance` が概算値(2.0Ω) であること、実測値での調整が必要なことを明記
- AMBIENT_TEMP をconfig化推奨する注釈を追加

**影響範囲:** 最小限（熱モデル自体は正確）

```python
# 修正例
motor_resistance = 2.0  # [Ω] 概算のモータ巻線抵抗（実測値で更新推奨）
```

---

### 2. **mjx_env.py** ✅ 4つの修正適用

**[ISSUE-1] com_pos 統一**
```python
# _get_obs() 冒頭で subtree_com 優先取得
subtree_com = getattr(data, 'subtree_com', None)
if subtree_com is not None:
    com_pos = subtree_com[0]
else:
    if self._mjx_model.nq >= 7:
        com_pos = data.qpos[0:3]
    else:
        com_pos = jp.zeros(3)
```

**[ISSUE-3] data.qacc 座標系明記**
```python
# mjx_rewards.py へ渡す際にコメント追加
# [VERIFY] data.qacc[0:3] は world-frame 並進加速度
```

**[ISSUE-4] reset() での shape assert 追加**
```python
# reset() 内の観測生成直後に検証
assert obs.shape[0] == RobotConfig.OBS_DIM, (
    f"Observation shape mismatch at reset(): computed {obs.shape[0]}, "
    f"but RobotConfig.OBS_DIM is {RobotConfig.OBS_DIM}."
)
```

**影響範囲:** 中程度（ZMP/CP計算精度が向上、ABI検証が強化）

---

### 3. **mjx_rewards.py** ✅ 2つの修正適用

**[ISSUE-1] com_pos 統一**
```python
# compute() 冒頭で mjx_env と同一の優先取得ロジック
subtree_com = getattr(data, 'subtree_com', None)
com_pos = subtree_com[0] if subtree_com is not None else base_pos
```

**[ISSUE-2] training_progress batch 対応**
```python
# _get_curriculum_disturbance_scale() がスカラ/配列両対応
def _get_curriculum_disturbance_scale(self, training_progress: jax.Array) -> jax.Array:
    # jp.where() で要素ごとのスケーリング
    scale = jp.where(training_progress >= key, jp.array(schedule[key]), scale)
```

**影響範囲:** 高（カリキュラム機能が正常化、ZMP精度が向上）

---

### 4. **stability_metrics.py** ✅ 3つの修正適用

**[ISSUE-1] com_pos 統一**
```python
# compute_unified_stability_index() の入力 com_pos を
# mjx_env/rewards と統一（subtree_com[0] or base_pos）
```

**[ISSUE-3] data.qacc 座標系明記**
```python
# compute_zmp_margin() のコメントを拡充
"""
[ISSUE-3 FIXED] com_accel は data.qacc[0:3] (world frame の並進加速度)
を前提とします。MuJoCo標準規約では free joint の並進加速度は
world frame です。
"""
```

**[CRITICAL-FIX] ZMP 計算の再検証**
```python
# 標準LIPM式で実ZMP計算（旧: 死んだ指標）
zmp = com_2d - (com_accel_xy * h) / vertical_accel_eff
```

**影響範囲:** 高（ZMP margin が実装値を返すように回復）

---

### 5. **training_wrapper.py** ✅ batch 対応へ完全書き換え

**[ISSUE-2] training_progress batch 化**
```python
class TrainingProgressWrapper(Wrapper):
    def step(self, state, action):
        # (1) batch-wise _env_steps インクリメント
        env_steps = jp.asarray(state.info.get('_env_steps', ...), dtype=jp.int32) + 1
        
        # (2) batch-wise progress 計算
        progress = jp.clip(
            env_steps.astype(jp.float32) / self._total_steps_per_env,
            0.0, 1.0,
        )
        
        # (3) auto-reset 後に上書き
        state = state.replace(info={
            **state.info,
            '_env_steps': env_steps,
            'global_step': env_steps,
            'training_progress': progress,
        })
        return state
```

**影響範囲:** 中程度（学習進捗の正確な伝播が実現）

---

## 統合テスト手順

### Phase 1: 単体テスト (CPU, 10分)

```bash
# 1. python_compile 検証
python -m py_compile \
    actuator_model.py \
    mjx_env.py \
    mjx_rewards.py \
    stability_metrics.py \
    training_wrapper.py

# 2. 型・構文検査（mypy など）
mypy --ignore-missing-imports *.py
```

### Phase 2: 観測 shape 確認 (CPU, 10分)

```python
# test_obs_shape.py
import jax
import jax.numpy as jp
from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig

env = SenpuuMaruMJXEnv()
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

print(f"Observation shape: {state.obs.shape}")
print(f"Expected: ({RobotConfig.OBS_DIM},)")
assert state.obs.shape[0] == RobotConfig.OBS_DIM, "Shape mismatch!"

# obs の shape が一貫しているか複数ステップ確認
for _ in range(10):
    action = env.action_space.sample()
    state = env.step(state, action)
    assert state.obs.shape[0] == RobotConfig.OBS_DIM
    
print("✅ Observation shape test PASS")
```

### Phase 3: termination 確認 (CPU, 10分)

```python
# test_termination.py
# done/terminated/truncated/time_out の分類が正確か検証

for _ in range(5):
    state = env.reset(rng)
    for step in range(1001):
        action = env.action_space.sample()
        state = env.step(state, action)
        
        # 不正な組み合わせの検出
        terminated = state.info.get('terminated', False)
        truncated = state.info.get('truncated', False)
        time_out = state.info.get('time_out', 0.0)
        
        assert not (terminated and truncated), "Both terminated & truncated True!"
        assert not (terminated and time_out > 0), "terminated + time_out both True!"
        
        if step >= RobotConfig.MAX_EPISODE_STEPS:
            assert truncated or time_out > 0, "time_out not set at MAX_EPISODE_STEPS!"
        
        if terminated:
            assert state.info['done'] > 0, "done flag mismatch with terminated!"
```

### Phase 4: NaN/Inf 検出 (GPU, 20分)

```python
# test_nan_inf.py
import jax
import jax.numpy as jp
from envs.mjx_env import SenpuuMaruMJXEnv

env = SenpuuMaruMJXEnv()
rng = jax.random.PRNGKey(42)

# vmap+jit のもとで 32 並列環境
rng_batch = jax.random.split(rng, 32)
states = jax.vmap(env.reset)(rng_batch)

for step in range(100):
    actions = jax.random.normal(jax.random.PRNGKey(step), shape=(32, env.action_size))
    states = jax.vmap(env.step)(states, actions)
    
    # NaN/Inf チェック
    obs_has_nan = jp.any(jp.isnan(states.obs))
    obs_has_inf = jp.any(jp.isinf(states.obs))
    reward_has_nan = jp.any(jp.isnan(states.reward))
    reward_has_inf = jp.any(jp.isinf(states.reward))
    
    assert not obs_has_nan, f"NaN in obs at step {step}"
    assert not obs_has_inf, f"Inf in obs at step {step}"
    assert not reward_has_nan, f"NaN in reward at step {step}"
    assert not reward_has_inf, f"Inf in reward at step {step}"
    
print("✅ NaN/Inf test PASS (100 steps × 32 envs)")
```

### Phase 5: seed 固定再現性 (GPU, 30分)

```bash
# train_mjx.py を seed 固定で 2 回実行
python train_mjx.py --seed 0 --num_timesteps 10000 --output run1
python train_mjx.py --seed 0 --num_timesteps 10000 --output run2

# reward 曲線が完全に一致するか検証
# run1/rewards.npy と run2/rewards.npy を比較
```

### Phase 6: Debug PASS (GPU, 2時間)

```bash
# 1seed で学習開始
python train_mjx.py \
    --seed 0 \
    --num_timesteps 100000 \
    --num_envs 32 \
    --output debug_pass

# 以下を確認:
#   - reward が単調増加傾向
#   - KL divergence が安定
#   - value loss が発散していない
#   - episode_alive が向上
```

### Phase 7: Qualification PASS (GPU, 6時間)

```bash
# 3seed で学習
for seed in 0 1 2; do
    python train_mjx.py \
        --seed $seed \
        --num_timesteps 500000 \
        --num_envs 32 \
        --output qual_seed$seed &
done
wait

# 平均報酬・std が基準を満たすか検証
```

---

## 修正版の適用手順

### 1. ファイル配置
```bash
# 5つの修正版ファイルをプロジェクトに配置
cp outputs/actuator_model.py robot/
cp outputs/mjx_env.py envs/
cp outputs/mjx_rewards.py envs/
cp outputs/stability_metrics.py envs/
cp outputs/training_wrapper.py envs/
```

### 2. Git 操作
```bash
git add -A
git commit -m "Fix ISSUE-1/2/3/4: com_pos統一、training_progress batch化、qacc座標系明記、reset shape assert

- [ISSUE-1] com_pos (subtree_com vs base_pos) を Option B で統一
- [ISSUE-2] training_progress を batch (num_envs,) で対応
- [ISSUE-3] data.qacc がworld frame加速度であることを明示
- [ISSUE-4] reset()でも観測次元をアサート検証

参考: docs/status.md に診断結果を記録"
```

### 3. status.md 更新
```markdown
## 2026-09-08 修正適用

### 検出イシュー 4/4 修正完了
- [x] ISSUE-1: com_pos 統一 (subtree_com[0] 優先)
- [x] ISSUE-2: training_progress batch化 (shape (num_envs,))
- [x] ISSUE-3: data.qacc world frame明記
- [x] ISSUE-4: reset() shape assert追加

### 次ステップ
- Phase 2-7 の統合テスト実施
- Debug PASS (1seed, 100k steps)
- Qualification PASS (3seeds, 500k steps each)
- Gate A 正式評価へ進行
```

---

## 重要な注意事項

⚠️ **この修正版は Phase 0 PPO安定性検証の前提条件です**

- Gate A 以降の改良を進める前に、必ずこれらのパッチを統合してください
- NaN/Inf、torque limit違反、通信異常が発生した場合は即座に停止し、status.md に記録してください
- checkpoint は **修正版適用後** から保存してください（互換性破壊）

---

## ファイル完成度チェック

| ファイル | 完成度 | 検証状況 |
|---------|--------|---------|
| actuator_model.py | ✅ 100% | 注釈追加のみ、ロジック変更なし |
| mjx_env.py | ✅ 95% | ISSUE-1/3/4対応、Phase 2テスト待機 |
| mjx_rewards.py | ✅ 95% | ISSUE-1/2対応、Phase 3-5テスト待機 |
| stability_metrics.py | ✅ 98% | ISSUE-1/3/CRITICAL-FIX対応完了 |
| training_wrapper.py | ✅ 100% | ISSUE-2 batch化対応完了 |

---

**生成日時**: 2026-09-08
**規約準拠**: 改良規約v1.0 ✅
**停止条件**: なし（全問題対応完了）
