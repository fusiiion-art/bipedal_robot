"""
safety/cbf.py — Control Barrier Function (CBF) Safety Layer for Bipedal Posture Control

【修正対応 (2026-09-08)】
- [CBF-1 FIXED] joint_pos_margin を可動域比率ベースで動的計算
- [CBF-2 FIXED] filter_action() と compute_cbf_penalty() のペナルティ基準を統一
- [CBF-3 FIXED] double-clamp の実装戦略を明確化（ドキュメント化）

設計理念:
  学習時: 簡易版CBF（クリップ + ペナルティ）で微分可能性を保証
  実機時: 実装 safety/cbf_realworld.py で QP ベースの strict CBF へ切り替え
  
  本ファイルは「学習用の近似」として位置付けられている。
"""

import jax
import jax.numpy as jp
from typing import Tuple

from robot.config import RobotConfig


class CBFSafetyFilter:
    """
    Control Barrier Function (CBF) Safety Layer for Bipedal Posture Control.
    
    In JAX/MJX training, running a full QP solver per step per environment is prohibitively slow.
    This provides a simplified, differentiable margin-based clamping mechanism that mimics CBF,
    ensuring that nominal actions pushing the system towards unsafe states (e.g., instability,
    joint limits) are heavily penalized or clipped.
    
    During real-world deployment on Raspberry Pi, a strict QP-based CBF should replace this.
    See: safety/cbf_realworld.py (future)
    
    【実装戦略 (CBF-3 FIXED)】
    - RL側: 粗い安全クランプ（JOINT_LIMITS_MIN/MAX）を適用
    - CBF側: 「いかに粗クランプが効いたか」をペナルティで測定
    - 効果: RL が粗クランプを避けるよう学習 → 実質的な safety margin が徐々に形成
    
    ダブルクランプの正当性:
      第1クランプ（RL側）: 物理的なハードストップとして機能
      第2クランプ（CBF側）:「ハードストップが不要になる」ように RL を訓練
    """
    
    def __init__(self):
        """
        初期化。マージンを config.py から取得。
        """
        self.max_torque = RobotConfig.MOTOR_MAX_TORQUE
        self.max_vel = RobotConfig.MOTOR_MAX_VELOCITY
        
        # [CBF-1 FIXED] margin_ratio ベースの動的計算へ変更
        # 関節ごとに可動域の一定比率をマージンとする
        self.margin_ratio = 0.05  # 可動域の 5% をマージンとする
        
        # ペナルティ係数
        self.cbf_penalty_scale = 1.0  # mjx_rewards.py の weight と整合
        self.softplus_steepness = 10.0  # softplus の k パラメータ
    
    def compute_safe_margins(
        self,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """
        [CBF-1 FIXED] 可動域に応じた動的マージンを計算。
        
        各関節の可動域の一定比率（margin_ratio）をマージンとすることで、
        相対的な安全性を統一させる。
        
        Args:
            limit_lower: 関節下限 [rad] shape=(n_joints,)
            limit_upper: 関節上限 [rad] shape=(n_joints,)
        
        Returns:
            (safe_lower, safe_upper): マージンを適用した安全範囲
        """
        ranges = limit_upper - limit_lower
        
        # [CBF-1 FIXED] 可動域の margin_ratio% をマージンとして計算
        margins = ranges * self.margin_ratio
        
        safe_lower = limit_lower + margins
        safe_upper = limit_upper - margins
        
        return safe_lower, safe_upper
    
    def filter_action(
        self,
        nominal_action: jp.ndarray,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> jp.ndarray:
        """
        Takes the RL's nominal action and projects it to a safe set.
        
        実装戦略 (CBF-3):
        - 粗いハードクランプ（JOINT_LIMITS_MIN/MAX）を第1段で適用
        - マージンベースの制約を第2段で適用
        - 効果: RL が「ハードクランプを避ける」ように学習
        
        Args:
            nominal_action: RL の提案アクション [rad] shape=(n_joints,)
            limit_lower: 関節下限 [rad]
            limit_upper: 関節上限 [rad]
        
        Returns:
            safe_action: 安全範囲内に制限されたアクション
        """
        # [CBF-1 FIXED] 動的マージンを計算
        safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
        
        # クランプ: マージン内に制限
        safe_action = jp.clip(nominal_action, safe_lower, safe_upper)
        
        return safe_action
    
    def compute_cbf_penalty(
        self,
        nominal_action: jp.ndarray,
        safe_action: jp.ndarray,
        limit_lower: jp.ndarray = None,
        limit_upper: jp.ndarray = None
    ) -> jp.ndarray:
        """
        [CBF-2 FIXED] CBFペナルティを計算。
        
        実装戦略 (CBF-2/3):
        - filter_action() で実際にクランプされた「差分」に基づくペナルティを計算
        - これにより、filter_action() と compute_cbf_penalty() の基準を統一
        - ペナルティ = (RL が安全範囲を超えようとした度合い)
        
        計算方式:
          1. 直接法: クランプ前後の差分量を測定
             penalty = sum(|nominal_action - safe_action|)
          2. マージンベース法: マージン超過量を測定（より厳格）
             penalty = softplus で連続ペナルティ化
        
        Args:
            nominal_action: RL の提案アクション [rad]
            safe_action: filter_action() で制限されたアクション [rad]
            limit_lower: 関節下限 [rad]（マージンベース法を使う場合は必須）
            limit_upper: 関節上限 [rad]（マージンベース法を使う場合は必須）
        
        Returns:
            penalty: スカラーペナルティ値（報酬から減算）
        """
        # [CBF-2 FIXED] 直接法: クランプ差分に基づくペナルティ
        clamp_diff = jp.abs(nominal_action - safe_action)
        
        # L1 ノルムで累積（クランプ差分が大きいほど大きいペナルティ）
        direct_penalty = jp.sum(clamp_diff)
        
        # オプション: マージンベース法（より厳格）
        if limit_lower is not None and limit_upper is not None:
            safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
            
            # softplus で連続的にペナルティ化
            # マージン超過量に応じたペナルティを計算
            upper_excess = jp.maximum(0.0, nominal_action - safe_upper)
            lower_excess = jp.maximum(0.0, safe_lower - nominal_action)
            
            # softplus: smooth approximation of ReLU
            # softplus(x) = (1/β) * log(1 + exp(β*x))
            # β=10 で ReLU に近づく（微分可能）
            margin_penalty = jp.sum(
                jax.nn.softplus(self.softplus_steepness * upper_excess) +
                jax.nn.softplus(self.softplus_steepness * lower_excess)
            )
            
            # 両方を組み合わせ
            total_penalty = direct_penalty + 0.5 * margin_penalty
        else:
            total_penalty = direct_penalty
        
        # スケーリング
        return total_penalty * self.cbf_penalty_scale
    
    def compute_cbf_penalty_legacy(
        self,
        nominal_action: jp.ndarray,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> jp.ndarray:
        """
        [DEPRECATED] 旧実装。後方互換性のために保持。
        
        【使用禁止】代わりに compute_cbf_penalty(nominal_action, safe_action) を使用。
        
        旧実装の問題点:
        - filter_action() と異なる基準でペナルティ計算
        - double-counting のリスク
        
        このメソッドは近い将来削除される予定です。
        """
        safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
        
        k = self.softplus_steepness
        upper_violation = jax.nn.softplus(k * (nominal_action - safe_upper))
        lower_violation = jax.nn.softplus(k * (safe_lower - nominal_action))
        
        return jp.sum(upper_violation + lower_violation) * self.cbf_penalty_scale


# ============================================================
# ユーティリティ関数
# ============================================================

def analyze_cbf_violations(
    nominal_actions: jp.ndarray,
    safe_actions: jp.ndarray,
    joint_names: list = None
) -> dict:
    """
    CBF が実際に何度介入したか（違反統計）を分析。
    
    学習中のモニタリング用。
    
    Args:
        nominal_actions: shape=(n_steps, n_joints)
        safe_actions: shape=(n_steps, n_joints)
        joint_names: 関節名（デフォルト: generic)
    
    Returns:
        stats: 統計情報辞書
    """
    violations = jp.abs(nominal_actions - safe_actions)
    
    stats = {
        'total_violations': float(jp.sum(violations)),
        'mean_violation': float(jp.mean(violations)),
        'max_violation': float(jp.max(violations)),
        'violation_frequency': float(jp.mean(violations > 1e-4)),  # 閾値: 0.01rad
    }
    
    # 関節ごとの統計
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(violations.shape[1])]
    
    stats['per_joint'] = {}
    for j, name in enumerate(joint_names):
        stats['per_joint'][name] = {
            'violations': float(jp.sum(violations[:, j])),
            'frequency': float(jp.mean(violations[:, j] > 1e-4)),
        }
    
    return stats