"""
real/rma_worker.py — RMA適応モジュール非同期ワーカー

10-20Hz で別プロセスとして実行。
メインループ (100Hz) とは共有メモリ経由で latent vector (8次元) を受け渡す。
"""

import time
import signal
import atexit
import numpy as np
from multiprocessing import shared_memory

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class RMAWorker:
    """
    RMA (Rapid Motor Adaptation) 適応器の非同期推論ワーカー。
    
    過去5ステップの観測+行動履歴を入力として、
    環境の潜在パラメータ (8次元) を推定し、共有メモリに書き込む。
    
    メインループより低頻度 (10-20Hz) で動作することで、
    100Hzのメインループに1msも追加しない。
    """
    
    BASE_OBS_DIM = 84
    ACT_DIM = 20
    HISTORY_LEN = 5
    LATENT_DIM = 8
    UPDATE_HZ = 20  # 20Hz (50ms間隔)
    
    def __init__(
        self,
        model_path: str = "/var/lib/bipedal_runtime/models/adaptation.onnx"
    ):
        self.dt = 1.0 / self.UPDATE_HZ
        
        # ONNX Runtime (シングルスレッド)
        self.session = None
        if ort is not None:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
                print(f"[RMA] Adaptation module loaded: {model_path}")
            except Exception as e:
                print(f"[RMA] Failed to load model: {e}")
        
        # 共有メモリにアタッチ (メインプロセスが作成済み)
        self.shm_latent = None
        self._setup_shared_memory()
        
        # クリーンアップ登録
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, lambda *_: (self._cleanup(), exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (self._cleanup(), exit(0)))
    
    def _setup_shared_memory(self):
        """メインプロセスが作成した共有メモリにアタッチ"""
        max_retries = 10
        for i in range(max_retries):
            try:
                self.shm_latent = shared_memory.SharedMemory(
                    name="robot_rma_latent", create=False
                )
                print("[RMA] Attached to shared memory 'robot_rma_latent'")
                return
            except FileNotFoundError:
                print(f"[RMA] Waiting for main process to create shared memory... ({i+1}/{max_retries})")
                time.sleep(1.0)
        
        print("[RMA] WARNING: Could not attach to shared memory. Running without.")
    
    def _cleanup(self):
        if self.shm_latent:
            try:
                self.shm_latent.close()
            except Exception:
                pass
    
    def run(self):
        """20Hzのメインループ"""
        print(f"[RMA] Starting adaptation loop at {self.UPDATE_HZ}Hz")
        
        while True:
            t_start = time.monotonic()
            
            if self.session is not None:
                # 入力構築 (簡易: ゼロ入力。実装時はメインプロセスから
                # 共有メモリ経由で観測/行動履歴を受け取る)
                history_dim = (self.BASE_OBS_DIM + self.ACT_DIM) * self.HISTORY_LEN
                history_input = np.zeros((1, history_dim), dtype=np.float32)
                
                try:
                    result = self.session.run(None, {self.input_name: history_input})
                    latent = result[0].flatten()[:self.LATENT_DIM].astype(np.float32)
                except Exception as e:
                    latent = np.zeros(self.LATENT_DIM, dtype=np.float32)
                    if time.monotonic() - t_start < 0.001:
                        print(f"[RMA] Inference error: {e}")
            else:
                latent = np.zeros(self.LATENT_DIM, dtype=np.float32)
            
            # 共有メモリに書き込み
            if self.shm_latent:
                self.shm_latent.buf[:32] = latent.tobytes()
            
            # タイミング制御
            elapsed = time.monotonic() - t_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    """
    実行方法:
      sudo chrt -f 90 taskset -c 2 python3 -m real.rma_worker
    """
    worker = RMAWorker()
    worker.run()


if __name__ == "__main__":
    main()
