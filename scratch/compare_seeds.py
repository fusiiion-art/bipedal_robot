import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / 'log/version_0/gate_a_seed0.json', 'r', encoding='utf-8') as f:
    s0 = json.load(f)
with open(ROOT / 'log/gate_a_seed1/version_0/gate_a_seed1.json', 'r', encoding='utf-8') as f:
    s1 = json.load(f)

for cond in s0['conditions']:
    print(f'=== {cond} ===')
    r0 = s0['conditions'][cond]['reward_component_means']
    r1 = s1['conditions'][cond]['reward_component_means']
    all_keys = sorted(set(r0.keys()) | set(r1.keys()))
    print(f"{'Metric':<30} {'seed0':>12} {'seed1':>12} {'diff (s1-s0)':>15} {'ratio (s1/s0)':>12}")
    print('-' * 85)
    for k in all_keys:
        v0 = r0.get(k, 0.0)
        v1 = r1.get(k, 0.0)
        diff = v1 - v0
        ratio = (v1 / v0) if abs(v0) > 1e-9 else float('nan')
        # 大きく差があるものにマークを付ける
        mark = " ***" if abs(diff) > 0.05 or (ratio < 0.8 or ratio > 1.25) else ""
        print(f"{k:<30} {v0:>12.6f} {v1:>12.6f} {diff:>15.6f} {ratio:>12.4f}{mark}")
    print()
