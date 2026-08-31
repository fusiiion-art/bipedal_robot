#!/usr/bin/env python3
"""Phase 0 diagnostics: KL trend, log_std range, and truncation signal presence."""

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def collect_std_like(tree, path=""):
    stats = []
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            p = f"{path}/{k}" if path else str(k)
            key_l = str(k).lower()
            if "log_std" in key_l or "std" in key_l:
                arr = np.asarray(v)
                stats.append({
                    "path": p,
                    "shape": list(arr.shape),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                })
            stats.extend(collect_std_like(v, p))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            p = f"{path}[{i}]"
            stats.extend(collect_std_like(v, p))
    return stats


def kl_stats(log_items):
    vals = []
    for item in log_items:
        if "training/kl_mean" in item:
            vals.append(float(item["training/kl_mean"]))
    if not vals:
        return None
    a = np.asarray(vals)
    spikes = []
    for item in log_items:
        value = item.get("training/kl_mean")
        if value is not None and float(value) > 0.1:
            spikes.append({
                key: item[key]
                for key in (
                    "step", "training/kl_mean", "training/learning_rate",
                    "training/entropy", "training/policy_loss",
                    "training/total_loss", "eval/episode_reward",
                )
                if key in item
            })
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "gt_0_1_rate": float(np.mean(a > 0.1)),
        "gt_0_05_rate": float(np.mean(a > 0.05)),
        "spikes": spikes,
    }


def policy_std_stats(log_items):
    keys = (
        "training/policy_dist_min_std",
        "training/policy_dist_mean_std",
        "training/policy_dist_max_std",
    )
    values = {key: [float(item[key]) for item in log_items if key in item] for key in keys}
    return {
        key: {
            "count": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "last": float(vals[-1]),
        }
        for key, vals in values.items()
        if vals
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "log" / "version_7" / "log.json")
    parser.add_argument("--params", type=Path, default=ROOT / "log" / "version_7" / "final_params.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "phase0_ppo_diag.json")
    args = parser.parse_args()

    with args.log.open("r", encoding="utf-8") as f:
        log_items = json.load(f)

    with args.params.open("rb") as f:
        params = pickle.load(f)

    out = {
        "log_path": str(args.log),
        "params_path": str(args.params),
        "kl": kl_stats(log_items),
        "policy_std": policy_std_stats(log_items),
        "std_like": collect_std_like(params),
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[Phase0] saved: {args.out}")


if __name__ == "__main__":
    main()
