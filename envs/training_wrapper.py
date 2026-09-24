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
        fixed_progress: [2026-09-22 追加] Noneでなければ、reset()時点の
            training_progressをこの固定値に強制し、step()内でも
            env_steps由来の値で上書きせずこの値を維持し続ける。
            これは「eval専用」の使い方を想定している。
            Brax標準のEvaluator.generate_eval_unroll()はeval_env.reset()
            を評価のたびに呼び直す実装になっており、通常の
            TrainingProgressWrapper(fixed_progress=None)のままだと
            training_progressが常に0近傍（最大でも
            episode_length/total_steps_per_env 程度）にしかならない。
            これにより_get_curriculum_disturbance_scale()が学習の
            進捗に関わらず常に「外乱なし」を返し、外乱カリキュラムが
            eval側では一切検証できない（train側は別経路で正しく
            進捗するため、trainとevalで見えるものが食い違う）。
            fixed_progressを指定したTrainingProgressWrapperをeval_env
            専用に用意することで、「特定のカリキュラム段階を固定して
            評価する」ための評価パスを構築できる
            (train/train_mjx.py の _build_disturbed_evaluator 参照)。
    
    使用例:
        env = brax_env
        env = jax.vmap(env.reset)(rng_batch)
        env = training.wrap(env, num_envs)  # AutoResetWrapper を適用
        env = TrainingProgressWrapper(env, total_steps_per_env=100000)
    """
    
    def __init__(self, env, total_steps_per_env: int, fixed_progress: float = None):
        super().__init__(env)
        self._total_steps_per_env = max(float(total_steps_per_env), 1.0)
        self._fixed_progress = fixed_progress

    def reset(self, rng):
        state = self.env.reset(rng)
        
        # [ISSUE-2 FIXED] batch 対応: num_envs を検出
        batch_size = state.obs.shape[0] if state.obs.ndim > 1 else 1
        
        if self._fixed_progress is None:
            initial_progress = jp.zeros(batch_size, dtype=jp.float32)
        else:
            initial_progress = jp.full(
                (batch_size,), float(self._fixed_progress), dtype=jp.float32
            )

        state = state.replace(info={
            **state.info,
            '_env_steps': jp.zeros(batch_size, dtype=jp.int32),
            'global_step': jp.zeros(batch_size, dtype=jp.int32),
            'training_progress': initial_progress,
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
        
        if self._fixed_progress is None:
            progress = jp.clip(
                env_steps.astype(jp.float32) / self._total_steps_per_env,
                0.0, 1.0,
            )
        else:
            # 固定進捗モード: env_stepsは（他の用途のため）通常通り
            # 積算するが、training_progressはfixed_progressに維持する。
            batch_size = env_steps.shape[0]
            progress = jp.full(
                (batch_size,), float(self._fixed_progress), dtype=jp.float32
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


class EpisodeInfoResetWrapper(Wrapper):
    """AutoResetWrapperはpipeline_state/obs/Brax自身のinfo['steps']しかリセットしない。
    raw envが独自にinfoへ積んでいるepisode-scopedなフィールドは、そのままでは
    エピソード境界をまたいで持ち越されてしまう。これを、直前episodeがdoneだった
    envスロットについてのみ、reset()相当の初期値に戻す。"""

    RESETTABLE_KEYS = (
        "step", "last_potential", "action_history", "obs_history",
        "filtered_action", "last_action", "double_last_action", "triple_last_action",
        "servo_temp", "supply_volt", "dr_damping", "dr_friction", "dr_kp_scale",
        "disturbance_scale", "privileged_obs", "disturbance_recovery_steps",
        # DR値 (項目2でinfoに追加)
        "mass_scale", "fric_scale", "com_offset",
    )

    def step(self, state, action):
        was_done = state.done  # 直前ステップでこのスロットのepisodeが終わっていたか
        is_batched = state.obs.ndim > 1
        if is_batched:
            keys = jax.vmap(jax.random.split)(state.info["rng_key"])
            rng_for_fresh = keys[:, 0]
            fresh_state = jax.vmap(self.env.unwrapped.reset)(rng_for_fresh)
        else:
            rng_for_fresh, _ = jax.random.split(state.info["rng_key"])
            fresh_state = self.env.unwrapped.reset(rng_for_fresh)

        state = self.env.step(state, action)

        d = was_done > 0.5
        def pick(fresh, cur):
            cond = d
            if cond.ndim:
                cond = jp.reshape(cond, [cond.shape[0]] + [1] * (cur.ndim - 1))
            return jp.where(cond, fresh, cur)

        new_info = dict(state.info)
        for key in self.RESETTABLE_KEYS:
            if key in fresh_state.info and key in state.info:
                new_info[key] = pick(fresh_state.info[key], state.info[key])
        return state.replace(info=new_info)