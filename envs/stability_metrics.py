"""
Advanced Stability Metrics for Bipedal Robot Control
- Foot Placement Estimator (FPE) / LIPM Capture Point
- Zero Moment Point (ZMP) Margin (support-polygon based)
- Multi-point Contact Analysis
- Unified Stability Index

================================================================================
v2 (2026-07 レビュー) での主な修正点
================================================================================
1. [CRITICAL FIX] compute_zmp_margin():
   旧実装は `zmp_error = ||zmp - pressure_center||` かつ
   `zmp = pressure_center - zmp_correction` という定義だったため、
   代数的に `zmp_error = ||zmp_correction||` へ完全に相殺されていた。
   pressure_center・com_pos は式の中で一切効いておらず、かつ
   mjx_rewards.py 側が com_accel を [0,0,-9.81] 固定で渡していたことも
   重なって、zmp_margin は常に定数 1.0 を返す「死んだ」指標になっていた
   （検証スクリプトで再現・確認済み）。
   → 標準LIPM式 zmp = com_xy - com_accel_xy * h / (accel_z + g) で
     実際のZMPを算出し、「支持脚(単脚)または両脚を結ぶ線分を
     足平半径で膨らませたカプセル領域」までの符号付き距離として
     再定義した。

2. compute_lipm_metrics():
   Capture Point (p_cp) を戻り値に追加し、mjx_rewards.py 側で
   重複計算していたCPをこちらに一本化（DRY化・数値的不整合の排除）。

3. 統合指標の重み付け構造は既存設計を踏襲しつつ、各サブ指標が
   物理的に意味のある値を返すようになったことで、
   stability_index 全体の信頼性が回復している
   （旧: zmp_margin が常時+1.0のフリークレジットを与えていたため、
   本指標を閾値判定に使う disturbance_recovery_bonus 等が
   実際より「安定している」と誤認しやすい状態だった）。
================================================================================
"""

import jax
import jax.numpy as jp
from typing import Tuple, Dict


class StabilityMetrics:
    """Compute advanced stability metrics for disturbance-resistant control."""

    def __init__(
        self,
        left_foot_id: int,
        right_foot_id: int,
        com_height: float = 0.28,
        foot_support_radius: float = 0.06,
    ):
        self._left_foot_id = left_foot_id
        self._right_foot_id = right_foot_id
        self._com_height = com_height
        # 足平の実効支持半径。mjx_env.py の FSR レイアウト
        # (前後~8cm, 左右半幅~5cm) に整合する概算値。
        # 実URDFの足裏形状に合わせて要調整。
        self._foot_support_radius = foot_support_radius

    # ------------------------------------------------------------------
    # 幾何ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _point_to_segment_distance(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
        """点 p から線分 ab までの最短距離（2Dベクトル入力）。"""
        ab = b - a
        ab_len_sq = jp.dot(ab, ab) + 1e-9
        t = jp.clip(jp.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        return jp.linalg.norm(p - closest)

    # ------------------------------------------------------------------
    # 1. LIPM Capture Point
    # ------------------------------------------------------------------
    def compute_lipm_metrics(
        self,
        com_pos: jax.Array,
        com_vel: jax.Array,
        rpy: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        計算: Capture Point、最寄り足までのマージン、CP座標そのもの。

        Returns:
            (best_cp_dist, stability_margin, p_cp)
        """
        h = self._com_height
        g = 9.81
        omega_0 = jp.sqrt(g / h)

        p_com = com_pos[0:2]
        v_com = com_vel[0:2]
        p_cp = p_com + v_com / omega_0

        left_foot_2d = left_foot_pos[0:2]
        right_foot_2d = right_foot_pos[0:2]

        cp_dist_l = jp.linalg.norm(left_foot_2d - p_cp)
        cp_dist_r = jp.linalg.norm(right_foot_2d - p_cp)
        best_cp_dist = jp.minimum(cp_dist_l, cp_dist_r)

        foot_span = jp.linalg.norm(right_foot_2d - left_foot_2d)
        max_margin = foot_span / 2.0 + self._foot_support_radius
        stability_margin = jp.maximum(0.0, max_margin - best_cp_dist)

        return best_cp_dist, stability_margin, p_cp

    # ------------------------------------------------------------------
    # 2. ZMP Margin (support-polygon based) — [CRITICAL FIX]
    # ------------------------------------------------------------------
    def compute_zmp_margin(
        self,
        com_pos: jax.Array,
        com_accel: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        ZMP (Zero Moment Point) マージンを計算する。

        標準LIPM方程式 zmp = com_xy - com_accel_xy * h / (accel_z + g) で
        実際のZMPを求め、支持基底（片脚支持ではその足、両脚支持では
        両足を結ぶ線分を足平半径で膨らませたカプセル領域）までの
        符号付き距離としてマージンを定義する。

        Args:
            com_pos: 重心位置 [3]
            com_accel: 重心の線形加速度 [3] (data.qacc[0:3] 相当。
                       ワールド座標系であることを前提とする。要検証)
            left_foot_pos / right_foot_pos: 足位置 [3]
            left_foot_force / right_foot_force: 足裏鉛直反力 [スカラ]

        Returns:
            (zmp_margin [0,1], zmp_point [2])
        """
        g = 9.81
        h = jp.clip(com_pos[2], 0.05, 1.0)  # 高さ0付近での特異点回避

        com_accel_xy = jp.clip(com_accel[0:2], -30.0, 30.0)  # 接触衝撃ノイズの飽和
        # 鉛直加速度 -> ほぼ自由落下(accel_z≈-g)の特異点回避のためクリップ。
        # 自由落下に近いほど分母が小さくなり補正項が急増する
        # = ZMPの物理的信頼性が失われる、という意図した挙動。
        vertical_accel_eff = jp.clip(com_accel[2] + g, 3.0, 50.0)

        com_2d = com_pos[0:2]
        zmp = com_2d - (com_accel_xy * h) / vertical_accel_eff

        total_force = left_foot_force + right_foot_force + 1e-6
        force_ratio_l = left_foot_force / total_force
        force_ratio_r = right_foot_force / total_force

        left_foot_2d = left_foot_pos[0:2]
        right_foot_2d = right_foot_pos[0:2]

        # 片脚支持(荷重比が大きく偏っている)では支持領域を
        # 荷重側の足1点に収縮させる。閾値0.6は要チューニング。
        is_single_support = jp.abs(force_ratio_l - force_ratio_r) > 0.6
        stance_foot = jp.where(force_ratio_l > force_ratio_r, left_foot_2d, right_foot_2d)
        seg_a = jp.where(is_single_support, stance_foot, left_foot_2d)
        seg_b = jp.where(is_single_support, stance_foot, right_foot_2d)

        dist_to_support = self._point_to_segment_distance(zmp, seg_a, seg_b)
        support_radius = self._foot_support_radius + 0.05  # 安全マージン込み

        zmp_margin = jp.clip(1.0 - (dist_to_support / support_radius), 0.0, 1.0)

        return zmp_margin, zmp

    # ------------------------------------------------------------------
    # 3. Foot Contact Balance
    # ------------------------------------------------------------------
    def compute_foot_contact_balance(
        self,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
    ) -> jax.Array:
        """
        左右の足接触圧力バランスを計算。
        完全にバランス (1:1) なら 1.0、一方に全て集中なら 0.0。
        """
        total_force = left_foot_force + right_foot_force + 1e-6
        ratio_l = left_foot_force / total_force
        balance = 1.0 - jp.abs(ratio_l - 0.5) * 2.0
        return jp.clip(balance, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. Orientation Margin
    # ------------------------------------------------------------------
    def compute_orientation_margin(
        self,
        rpy: jax.Array,
        base_ang_vel: jax.Array,
        safe_angle: float = 0.3,
    ) -> jax.Array:
        """姿勢安全マージン。ロール・ピッチが小さく角速度が低いほど高い。"""
        tilt_err = jp.sqrt(jp.square(rpy[0]) + jp.square(rpy[1]))
        ang_vel_xy = jp.linalg.norm(base_ang_vel[0:2])

        angle_margin = jp.clip(1.0 - (tilt_err / (safe_angle + 1e-6)), 0.0, 1.0)
        ang_vel_margin = jp.exp(-5.0 * ang_vel_xy)

        margin = angle_margin * ang_vel_margin
        return jp.clip(margin, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Unified Stability Index
    # ------------------------------------------------------------------
    def compute_unified_stability_index(
        self,
        com_pos: jax.Array,
        com_vel: jax.Array,
        com_accel: jax.Array,
        rpy: jax.Array,
        base_ang_vel: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
        gait_phase: float = 0.0,
        cp_margin_norm_dist: float = 0.15,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        複数の安定性指標を統合し、統一的な安定性インデックスを計算する。

        Returns:
            (stability_index, metrics_dict)
        """
        # 1. LIPM Capture Point
        cp_dist, cp_margin, p_cp = self.compute_lipm_metrics(
            com_pos, com_vel, rpy, left_foot_pos, right_foot_pos
        )
        cp_margin_norm = jp.clip(cp_margin / cp_margin_norm_dist, 0.0, 1.0)

        # 2. ZMP Margin [FIXED]
        zmp_margin, zmp_point = self.compute_zmp_margin(
            com_pos, com_accel, left_foot_pos, right_foot_pos,
            left_foot_force, right_foot_force
        )

        # 3. Foot Contact Balance
        foot_balance = self.compute_foot_contact_balance(
            left_foot_force, right_foot_force
        )

        # 4. Orientation Margin
        orient_margin = self.compute_orientation_margin(
            rpy, base_ang_vel, safe_angle=0.3
        )

        # Gait Phase に応じた動的重み付け
        # Double support (0.0~0.2, 0.8~1.0) では foot_balance を重視
        # Single support (0.2~0.8) では CP と orientation を重視
        is_single_support = jp.logical_or(
            jp.logical_and(gait_phase > 0.2, gait_phase < 0.8),
            gait_phase < 0.0  # フェーズ情報がない場合はデフォルト
        )

        w_cp = jp.where(is_single_support, 0.45, 0.25)
        w_zmp = jp.where(is_single_support, 0.25, 0.35)
        w_balance = jp.where(is_single_support, 0.15, 0.25)
        w_orient = 0.15

        stability_index = (
            w_cp * cp_margin_norm +
            w_zmp * zmp_margin +
            w_balance * foot_balance +
            w_orient * orient_margin
        )

        metrics = {
            'cp_dist': cp_dist,
            'cp_margin': cp_margin_norm,
            'cp_point': p_cp,
            'zmp_margin': zmp_margin,
            'zmp_point': zmp_point,
            'foot_balance': foot_balance,
            'orient_margin': orient_margin,
            'stability_index': stability_index,
        }

        return stability_index, metrics
