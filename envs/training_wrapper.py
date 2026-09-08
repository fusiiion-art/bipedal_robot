"""
TrainingProgressWrapper: Brax PPO環境への学習進捗率の注入

Brax PPOの内部ループでは環境がJITコンパイル・vmapされ、
AutoResetWrapper によってエピソード終了時に info が reset() の
初期値で上書きされる。そのため info 内のカウンタは自然には
エピソード間で持続しない。

本ラッパーは AutoResetWrapper の**外側**に適用することで、
エピソード境界を跨いて単調増加する学習進捗率を維持する。

仕組み:
  1. step() の冒頭で _env_steps を読み取り・インクリメント
  2. 内側の step() を呼ぶ（AutoResetWrapper が done 時に
     info を reset 値で上書きする可能性がある）
  3. 返却された state の info を、保存しておいた正しい
     _env_steps / training_progress で上書きする

これにより、内側で何度 auto-reset が起きても、外側の
カウンタは単調増加し続ける。

================================================================================
v2 (2026-09 ISSUE-2 修正)
================================================================================
[ISSUE-2 FIXED] training_progress の batch 対応

training_progress は shape (num_envs,) の配列として供給される。
Brax PPO では num_envs 個の並列環境が同期的に実行されるため、
各環境の progress を独立に計算する必要がある。

実装:
  - state.obs.shape[0] で num_envs を検出
  - _env_steps, training_progress を (num_envs,) 配列として管理
  - mjx_rewards.py 側の _get_curriculum_disturbance_scale() は
    要素ごとのスケーリングに対応
================================================================================
"""

import jax
import jax.numpy as jp
from brax.envs import Wrapper


class TrainingProgressWrapper(Wrapper):
    """
    学習進捗率 (0.0→1.0) を環境の info に注入するラッパー。
    Brax PPO の envs.training.wrap() が適用した後（= AutoResetWrapper
    の外側）に適用する必要がある。
    
    Args:
        env: Brax ラップ済み環境（AutoResetWrapper 適用済み）
        total_steps_per_env: 各並列環境あたりの総ステップ数
            = num_timesteps // num_envs
    
    使用例:
        env = brax_env
        env = jax.vmap(env.reset)(rng_batch)
        env = training.wrap(env, num_envs)  # AutoResetWrapper を適用
        env = TrainingProgressWrapper(env, total_steps_per_env=100000)
    """
    
    def __init__(self, env, total_steps_per_env: int):
        super().__init__(env)
        self._total_steps_per_env = max(float(total_steps_per_env), 1.0)

    def reset(self, rng):
        state = self.env.reset(rng)
        
        # [ISSUE-2 FIXED] batch 対応: num_envs を検出
        batch_size = state.obs.shape[0] if state.obs.ndim > 1 else 1
        
        state = state.replace(info={
            **state.info,
            '_env_steps': jp.zeros(batch_size, dtype=jp.int32),
            'global_step': jp.zeros(batch_size, dtype=jp.int32),
            'training_progress': jp.zeros(batch_size, dtype=jp.float32),
            'terminated': jp.zeros(batch_size, dtype=jp.bool_),
            'truncated': jp.zeros(batch_size, dtype=jp.bool_),
            'time_out': jp.zeros(batch_size, dtype=jp.float32),
        })
        return state

    def step(self, state, action):
        # [ISSUE-2 FIXED] (1) auto-reset で上書きされる前に、
        #                     現在のカウンタを取得して +1
        #                     batch-wise インクリメント
        
        env_steps_current = state.info.get('_env_steps', None)
        if env_steps_current is None:
            # フォールバック: 0 初期化（reset() が呼ばれていない場合）
            batch_size = state.obs.shape[0] if state.obs.ndim > 1 else 1
            env_steps_current = jp.zeros(batch_size, dtype=jp.int32)
        
        env_steps = jp.asarray(env_steps_current, dtype=jp.int32) + 1
        
        progress = jp.clip(
            env_steps.astype(jp.float32) / self._total_steps_per_env,
            0.0, 1.0,
        )
        
        # (2) 内側の step（AutoResetWrapper 含む）を実行
        #     done が True なら info は reset() の値で上書きされている
        state = self.env.step(state, action)
        
        # (3) 正しいカウンタ値で上書き（auto-reset のゼロクリアを無効化）
        #     [ISSUE-2 FIXED] batch-wise 値を返却
        state = state.replace(info={
            **state.info,
            '_env_steps': env_steps,
            'global_step': env_steps,
            'training_progress': progress,
        })
        
        return state