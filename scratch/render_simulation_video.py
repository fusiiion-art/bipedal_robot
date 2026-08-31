import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch the unified MJX replay entrypoint in video mode.")
    parser.add_argument("--exp_name", type=str, default="mjx_ppo_rma_100hz")
    parser.add_argument("--version", type=int, default=None, help="Version number to load (e.g. 17 for version_17).")
    parser.add_argument("--model", type=str, default="best_params.pkl", help="Checkpoint filename")
    parser.add_argument("--steps", type=int, default=300, help="Number of frames to render")
    parser.add_argument("--output", type=str, default="walking_simulation.gif", help="Output GIF filename")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    target = root_dir / "train" / "visualize_rl.py"
    if not target.exists():
        print(f"Error: {target} not found.")
        return

    cmd = [sys.executable, str(target),
           "--mode", "video",
           "--exp_name", args.exp_name,
           "--model", args.model,
           "--steps", str(args.steps),
           "--output", args.output]

    if args.version is not None:
        cmd += ["--version", str(args.version)]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
