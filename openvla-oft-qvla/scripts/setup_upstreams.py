"""Clone the pinned official repository next to PIVOT_Q, never reset an existing checkout."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qvla-root", type=Path, default=root.parent / "third_party" / "QVLA")
    args = parser.parse_args()
    lock = json.loads((root / "upstream.lock.json").read_text())
    target = args.qvla_root.resolve()
    if not target.exists():
        subprocess.run(["git", "clone", lock["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["commit"]], check=True)
    head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if head != lock["commit"]:
        raise SystemExit(f"Existing checkout is {head}, expected {lock['commit']}; left untouched")
    changes = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changes.strip():
        raise SystemExit("Official checkout has tracked modifications; left untouched")
    print(f"Verified official QVLA: {target} @ {head}")


if __name__ == "__main__":
    main()
