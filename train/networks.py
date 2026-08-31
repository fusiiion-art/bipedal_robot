"""
RMA (Rapid Motor Adaptation) 2段階ネットワークアーキテクチャ

学習時:
  Teacher Policy: (Base Obs + Privileged Info) -> Action
  Adaptation Module: (Obs/Action History) -> Latent Vector (特権情報を再現)
  
デプロイ時 (RPi5):
  Base Policy: (Base Obs + Latent Vector) -> Action
  Adaptation Module: バックグラウンドスレッドで10-20Hzで推論
"""
import jax
import jax.numpy as jp
from flax import linen as nn
from typing import Sequence

from robot.config import RobotConfig


class AdaptationModule(nn.Module):
    """
    RMA Adaptation Module (適応器)
    過去の観測・行動履歴から環境の潜在パラメータ(摩擦、質量、温度、電圧等)を推定する。
    1D-CNNベース。Raspberry Pi 5上では10-20Hzで非同期推論される想定。
    """
    latent_dim: int = 8
    hidden_dims: Sequence[int] = (128, 64)
    
    @nn.compact
    def __call__(self, obs_history: jp.ndarray, act_history: jp.ndarray):
        """
        Args:
            obs_history: (batch, HISTORY_LEN, BASE_OBS_DIM)
            act_history: (batch, HISTORY_LEN, ACT_DIM)
        Returns:
            latent: (batch, latent_dim) - 環境パラメータの潜在表現
        """
        # 観測と行動を時間軸で結合
        x = jp.concatenate([obs_history, act_history], axis=-1)
        
        # 1D Convolution (時間方向のパターン抽出)
        x = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        
        # Global Average Pooling (時間軸を潰す)
        x = jp.mean(x, axis=-2)  # (batch, 32)
        
        # MLP Head -> Latent Vector
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        latent = nn.Dense(self.latent_dim)(x)
        return latent


class BasePolicy(nn.Module):
    """
    RMA Base Policy (ベース方策)
    現在の観測 + 適応器が推定した潜在ベクトル -> 行動(残差)
    デプロイ時にはこのネットワークのみが100Hzで実行される。
    """
    action_dim: int = RobotConfig.NUM_JOINTS
    hidden_dims: Sequence[int] = (256, 256, 128)
    
    @nn.compact
    def __call__(self, obs: jp.ndarray, latent: jp.ndarray):
        """
        Args:
            obs: (batch, BASE_OBS_DIM) - 現在の観測
            latent: (batch, latent_dim) - 適応器の出力
        Returns:
            mean: (batch, action_dim) - 行動の平均
            log_std: (action_dim,) - 行動の対数標準偏差
        """
        x = jp.concatenate([obs, latent], axis=-1)
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.elu(x)
        
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        
        return mean, log_std


class TeacherPolicy(nn.Module):
    """
    RMA Teacher Policy (教師方策) - 学習フェーズ1でのみ使用
    特権情報(質量・摩擦・温度・電圧等)を直接受け取って最適な行動を学習する。
    この教師の行動を、AdaptationModule + BasePolicy が模倣するように蒸留する。
    """
    action_dim: int = RobotConfig.NUM_JOINTS
    hidden_dims: Sequence[int] = (512, 256, 128)
    
    @nn.compact
    def __call__(self, obs: jp.ndarray, privileged_obs: jp.ndarray):
        """
        Args:
            obs: (batch, BASE_OBS_DIM)
            privileged_obs: (batch, PRIVILEGED_OBS_DIM)
        Returns:
            mean, log_std
        """
        x = jp.concatenate([obs, privileged_obs], axis=-1)
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.elu(x)
        
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        
        return mean, log_std
