"""Train recovery adapters for one setting on one GPU."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pivot_q.configuration import config, parser, result_root, values
from pivot_q.processes import services

def main():
    p = parser(__doc__, ("pivot_q", "full_distill"))
    p.add_argument("--max-rounds", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    cfg = config(args)
    if cfg["backbone"] != "pi05":
        from pivot_q.suite_runs import run
        run(cfg, args, "train")
        return
    train = cfg["train"]
    seed = train["seed"] if args.seed is None else args.seed
    out = result_root(cfg, args) / "train" / "seed" / str(seed)
    run = values(cfg, args, out)
    run.update(max_rounds=args.max_rounds, resume=not args.no_resume)
    if args.dry_run:
        print(json.dumps(run, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    with services(cfg, args, out, anchor=run["lambda_anchor"] > 0):
        from pivot_q.pi05 import run as train_pi05
        train_pi05(run)

if __name__ == "__main__":
    main()
