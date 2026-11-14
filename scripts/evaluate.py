"""Evaluate one setting and seed on one GPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pivot_q.configuration import config, parser, result_root, values
from pivot_q.processes import services

def main():
    p = parser(__doc__, ("original", "quantized", "full_distill", "pivot_q"))
    p.add_argument("--benchmark", choices=("libero", "libero_plus"), default="libero_plus")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--adapter", type=Path)
    args = p.parse_args()
    cfg = config(args)
    if cfg["backbone"] != "pi05":
        from pivot_q.suite_runs import run
        run(cfg, args, "eval")
        return
    seed = cfg["evaluation"]["seed"] if args.seed is None else args.seed
    root = result_root(cfg, args)
    out = root / "eval" / args.benchmark / "seed" / str(seed)
    adapter = args.adapter
    if args.method in {"pivot_q", "full_distill"}:
        adapter = (adapter or root / "train" / "seed" / str(args.train_seed) / "final_adapter").resolve()
        if not args.dry_run and not adapter.is_dir():
            p.error(f"Missing adapter: {adapter}")
    elif adapter:
        p.error("Original and Quantized do not load an adapter")
    run = values(cfg, args, out)
    if args.dry_run:
        print(f"seed={seed} suites={run['suite_order']} adapter={adapter} output={out} benchmark={args.benchmark}")
        return
    out.mkdir(parents=True, exist_ok=True)
    from pivot_q.evaluation import evaluate
    from pivot_q.cost import record_compute
    with record_compute(out, cfg["evaluation"]["estimate_flops"], resume=args.resume):
        evaluate(cfg, run, out, seed, adapter, args.resume, args.benchmark)

if __name__ == "__main__":
    main()
