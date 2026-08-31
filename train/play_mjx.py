import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy launcher for visualize_rl.py")
    parser.add_argument("--exp_name", type=str, default="mjx_ppo_rma_100hz")
    parser.add_argument("--version", type=int, default=None, help="Version folder to load (e.g. 17 for version_17). If omitted, use latest.")
    parser.add_argument("--model", type=str, default="best_params.pkl", help="Model file to load (best_params.pkl, last_params.pkl, etc.)")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    target = root_dir / "train" / "visualize_rl.py"
    if not target.exists():
        print(f"Error: {target} not found.")
        return

    cmd = [sys.executable, str(target),
           "--mode", "interactive",
           "--exp_name", args.exp_name,
           "--model", args.model]
    if args.version is not None:
        cmd.append("--version")
        cmd.append(str(args.version))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
