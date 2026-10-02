"""学習済み checkpoint を実機用 ONNX (real/real_env.py の PolicyRunner が読む) に変換する。

[2026-10-02] 旧実装は JAX → jax2tf → TensorFlow SavedModel → tf2onnx の経路だったが、
JAX 0.4.3x 以降の jax2tf は常に native serialization (グラフ全体が1個の XlaCallModule op)
になり tf2onnx が ONNX の演算子へ変換できない。requirements-lock.txt の構成
(JAX 0.11 / tf2onnx 1.17) では export が必ず失敗していた。

現在は actor の deterministic 推論 (robot/policy_network.py と同じ構成) を ONNX の
基本演算子で直接組み立てる:
  x = (obs - mean) / std                       # 学習時の観測正規化 (running statistics)
  h = swish(x W0 + b0) → swish(h W1 + b1) → swish(h W2 + b2)
  loc_raw = (h W3 + b3)[:, :NUM_JOINTS]         # 出力の前半が平均、後半は std (推論では不要)
  action = tanh(POLICY_MEAN_CLIP_SCALE * loc_raw / (1 + |loc_raw|))
書き出し後、onnxruntime の出力が JAX の推論関数と一致することを検証する (不一致なら失敗)。

必要なパッケージ: onnx, onnxruntime
"""

import argparse
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from robot.policy_network import (
    POLICY_MEAN_CLIP_SCALE,
    POLICY_OBS_KEY,
    load_checkpoint,
    make_inference_fn_from_params,
    strip_device_dim,
)

INPUT_NAME = "observation"
OUTPUT_NAME = "action"
OPSET = 17


def _policy_arrays(params):
    """checkpoint から観測正規化 (mean, std) と MLP の各層 (kernel, bias) を取り出す。"""
    normalizer, policy_params = strip_device_dim(params)[:2]
    mean, std = normalizer.mean, normalizer.std
    if isinstance(mean, dict):
        mean, std = mean[POLICY_OBS_KEY], std[POLICY_OBS_KEY]
    layers = policy_params["params"]
    names = sorted(layers, key=lambda name: int(name.split("_")[-1]))
    if [n.rsplit("_", 1)[0] for n in names] != ["hidden"] * len(names):
        raise ValueError(f"unexpected policy network layers: {names}")
    dense = [(np.asarray(layers[n]["kernel"], np.float32), np.asarray(layers[n]["bias"], np.float32))
             for n in names]
    return np.asarray(mean, np.float32), np.asarray(std, np.float32), dense


def build_onnx_model(params):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    mean, std, dense = _policy_arrays(params)
    obs_dim = mean.shape[0]
    act_dim = RobotConfig.NUM_JOINTS
    if dense[-1][0].shape[1] != 2 * act_dim:
        raise ValueError(f"policy output size {dense[-1][0].shape[1]} != 2 * {act_dim}")

    inits = [
        numpy_helper.from_array(mean, "obs_mean"),
        numpy_helper.from_array(std, "obs_std"),
        numpy_helper.from_array(np.array([0], np.int64), "slice_start"),
        numpy_helper.from_array(np.array([act_dim], np.int64), "slice_end"),
        numpy_helper.from_array(np.array([1], np.int64), "slice_axis"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(POLICY_MEAN_CLIP_SCALE, np.float32), "mean_clip_scale"),
    ]
    nodes = [
        helper.make_node("Sub", [INPUT_NAME, "obs_mean"], ["obs_centered"]),
        helper.make_node("Div", ["obs_centered", "obs_std"], ["h_in"]),
    ]
    h = "h_in"
    for i, (kernel, bias) in enumerate(dense):
        inits += [numpy_helper.from_array(kernel, f"W{i}"), numpy_helper.from_array(bias, f"b{i}")]
        nodes += [
            helper.make_node("MatMul", [h, f"W{i}"], [f"mm{i}"]),
            helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"]),
        ]
        h = f"z{i}"
        if i < len(dense) - 1:  # swish(x) = x * sigmoid(x)。最終層は活性化なし
            nodes += [
                helper.make_node("Sigmoid", [h], [f"sig{i}"]),
                helper.make_node("Mul", [h, f"sig{i}"], [f"act{i}"]),
            ]
            h = f"act{i}"
    nodes += [
        helper.make_node("Slice", [h, "slice_start", "slice_end", "slice_axis"], ["loc_raw"]),
        helper.make_node("Abs", ["loc_raw"], ["loc_abs"]),
        helper.make_node("Add", ["loc_abs", "one"], ["loc_den"]),
        helper.make_node("Div", ["loc_raw", "loc_den"], ["loc_softsign"]),
        helper.make_node("Mul", ["loc_softsign", "mean_clip_scale"], ["loc"]),
        helper.make_node("Tanh", ["loc"], [OUTPUT_NAME]),
    ]
    graph = helper.make_graph(
        nodes, "senpuu_maru_policy",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, ["batch", obs_dim])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch", act_dim])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = min(model.ir_version, 9)  # 古い onnxruntime でも読めるように
    onnx.checker.check_model(model)
    return model


def verify_against_jax(params, onnx_path: str, n_samples: int = 256, atol: float = 1e-4) -> float:
    """ONNX (onnxruntime) と JAX の deterministic 推論の出力差の最大値を返す。atol 超過なら例外。"""
    import onnxruntime as ort

    mean, std, _ = _policy_arrays(params)
    rng = np.random.default_rng(0)
    # 学習時の観測分布付近 (正規化後に ±3σ 程度) の入力で比べる
    obs = (mean + std * rng.normal(size=(n_samples, mean.shape[0])).clip(-3, 3)).astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_action = session.run(None, {session.get_inputs()[0].name: obs})[0]

    infer = jax.jit(make_inference_fn_from_params(params, deterministic=True))
    jax_action = np.stack([
        np.asarray(infer({POLICY_OBS_KEY: jp.asarray(o)}, jax.random.PRNGKey(0))[0]) for o in obs
    ])
    max_diff = float(np.max(np.abs(onnx_action - jax_action)))
    if not np.isfinite(max_diff) or max_diff > atol:
        raise RuntimeError(f"ONNX output does not match JAX inference: max|diff|={max_diff:.3g} > {atol}")
    return max_diff


def export(model_path: str, output_path: str) -> float:
    import onnx

    params = load_checkpoint(model_path)
    model = build_onnx_model(params)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return verify_against_jax(params, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="学習済み checkpoint (best_params.pkl 等)")
    parser.add_argument("--output", type=str, default="brax_policy.onnx")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[Error] Parameter file not found: {args.model}")
        sys.exit(1)

    print("--- ONNX Export ---")
    print(f"Observation Dim: {RobotConfig.OBS_DIM} / Action Dim: {RobotConfig.NUM_JOINTS}")
    try:
        max_diff = export(args.model, args.output)
    except ImportError as e:
        print(f"[Error] {e}\n💡 `pip install onnx onnxruntime` を実行してください")
        sys.exit(1)
    print(f"✅ ONNX Model exported to: {args.output} (JAX との最大誤差 {max_diff:.2e})")


if __name__ == "__main__":
    main()
