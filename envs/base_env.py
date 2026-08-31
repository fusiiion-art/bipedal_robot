from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseEnv(ABC):
    """
    Sim/Real 共通インターフェース抽象クラス (ABC)
    MJX (シミュレーション) と Real (Raspberry Pi 5) の両方で
    同一の観測空間・行動空間・リセット/ステップAPIを共有するための基底クラス。
    """

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        環境をリセットし、初期観測と情報辞書を返す。
        Returns:
            obs: 初期観測ベクトル
            info: 追加情報辞書
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        """
        1制御ステップを実行する。
        Args:
            action: 行動ベクトル (関節角度目標値 [rad])
        Returns:
            obs: 観測ベクトル
            reward: 報酬スカラー
            done: エピソード終了フラグ
            info: 追加情報辞書
        """
        pass

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """観測空間の次元数"""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """行動空間の次元数"""
        pass
