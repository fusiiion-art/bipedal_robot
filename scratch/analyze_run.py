import json
data = json.load(open('log/version_1/log.json', 'r', encoding='utf-8'))
print(f"{'Step':>8} | {'Reward':>10} | {'KL_mean':>12} | {'Rwd/Step':>10} | {'Penalty':>10} | {'EpLen':>6} | {'LR':>10}")
print("-" * 90)
for d in data:
    s = d['step']
    r = d['reward']
    k = d.get('training/kl_mean', None)
    rps = d.get('eval/episode_reward_per_step', None)
    pen = d.get('eval/episode_total_penalty', None)
    ep = d.get('eval/avg_episode_length', None)
    lr = d.get('training/learning_rate', None)
    k_str = f"{k:.2f}" if k is not None else "N/A"
    rps_str = f"{rps:.2f}" if rps is not None else "N/A"
    pen_str = f"{pen:.0f}" if pen is not None else "N/A"
    ep_str = f"{ep:.1f}" if ep is not None else "N/A"
    lr_str = f"{lr:.2e}" if lr is not None else "N/A"
    print(f"{s:>8} | {r:>10.2f} | {k_str:>12} | {rps_str:>10} | {pen_str:>10} | {ep_str:>6} | {lr_str:>10}")

# Summary
rewards = [d['reward'] for d in data]
best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
worst_idx = min(range(len(rewards)), key=lambda i: rewards[i])
print(f"\nBest:  Step {data[best_idx]['step']} -> Reward {rewards[best_idx]:.2f}")
print(f"Worst: Step {data[worst_idx]['step']} -> Reward {rewards[worst_idx]:.2f}")

# Penalty trend
pens = [(d['step'], d.get('eval/episode_total_penalty', 0)) for d in data]
print(f"\nPenalty trend: {pens[0][1]:.0f} (Step 0) -> {pens[best_idx][1]:.0f} (Best) -> {pens[-1][1]:.0f} (Final)")
