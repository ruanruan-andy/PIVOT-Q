from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from eval_metrics import CATEGORY_ORDER, DISPLAY_SUITE, build_snapshot, format_duration


METHODS = (
    ("groot-fp16", "FP16"),
    ("groot-quantvla-w4a8", "QuantVLA"),
    ("groot-pivot_q-v2-w4a8", "PIVOT-Q-v2"),
)
SHORT_CATEGORY = {
    "Camera Viewpoints": "Cam",
    "Robot Initial States": "Robot",
    "Language Instructions": "Lang",
    "Light Conditions": "Light",
    "Background Textures": "Bg",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _success_style(value: float | None, evaluated: int) -> str:
    if evaluated <= 0 or value is None:
        return "dim"
    if value < 0.5:
        return "bold red"
    if value < 0.8:
        return "yellow"
    return "bold green"


def _success_text(value: float | None, evaluated: int, text: str | None = None) -> Text:
    return Text(
        text if text is not None else _percent(value),
        style=_success_style(value, evaluated),
    )


def _cell(row: dict[str, Any]) -> Text:
    completed = int(row.get("completed", 0))
    evaluated = int(row.get("evaluated", completed))
    total = int(row.get("total", 0))
    if completed == 0:
        text = f"0/{total}"
    elif completed == total:
        text = _percent(row.get("success_rate"))
    else:
        text = f'{_percent(row.get("success_rate"))} {completed}/{total}'
    return _success_text(row.get("success_rate"), evaluated, text)


def _category_cell(row: dict[str, Any]) -> Text:
    completed = int(row.get("completed", 0))
    evaluated = int(row.get("evaluated", completed))
    rate = row.get("success_rate")
    rate_text = "—" if rate is None else f"{100 * rate:.1f}%"
    return _success_text(rate, evaluated, f"{rate_text}({completed})")


def _category_average_cell(rows: list[dict[str, Any]]) -> Text:
    """Return the micro-average across categories with compact progress."""
    completed = sum(int(row.get("completed", 0)) for row in rows)
    evaluated = sum(int(row.get("evaluated", row.get("completed", 0))) for row in rows)
    successes = sum(int(row.get("successes", 0)) for row in rows)
    rate = successes / evaluated if evaluated else None
    return _category_cell(
        {
            "completed": completed,
            "evaluated": evaluated,
            "success_rate": rate,
        }
    )


def _state(total: dict[str, Any]) -> Text:
    completed = int(total.get("completed", 0))
    expected = int(total.get("total", 0))
    if expected and completed == expected:
        return Text("DONE", style="bold green")
    if completed:
        return Text("RUNNING", style="bold cyan")
    return Text("WAIT", style="dim")


def _overview(snapshot: dict[str, Any]) -> Table:
    table = Table(
        title="LIBERO-Plus Shared-560 · same-task adaptation",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
    )
    table.add_column("Method", style="bold")
    table.add_column("State")
    table.add_column("Progress", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("SR", justify="right")
    table.add_column("ETA", justify="right")
    for model, label in METHODS:
        total = snapshot["models"][model]["totals"]["libero-plus"]
        table.add_row(
            label,
            _state(total),
            f'{total["completed"]}/{total["total"]}',
            str(total["successes"]),
            _success_text(
                total["success_rate"],
                int(total.get("evaluated", total["completed"])),
            ),
            format_duration(total.get("eta_seconds")),
        )
    return table


def _suite_table(snapshot: dict[str, Any]) -> Table:
    table = Table(title="Four-suite comparison", box=box.SIMPLE, pad_edge=False)
    table.add_column("Suite")
    for _, label in METHODS:
        table.add_column(label, justify="right")
    indexed = {
        model: {
            row["suite"]: row
            for row in snapshot["models"][model]["benchmarks"]["libero-plus"]
        }
        for model, _ in METHODS
    }
    for suite in DISPLAY_SUITE:
        table.add_row(
            DISPLAY_SUITE[suite],
            *[_cell(indexed[model][suite]) for model, _ in METHODS],
        )
    return table


def _suite_category_table(snapshot: dict[str, Any]) -> Table:
    table = Table(
        title="4 suites × 7 categories (cell = SR%(done), n=20/category)",
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("Suite")
    table.add_column("Method")
    for category in CATEGORY_ORDER:
        table.add_column(SHORT_CATEGORY[category], justify="right")
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for model, _ in METHODS:
        rows = snapshot["models"][model]["libero_plus_groups"]["suite_categories"]
        indexed[model] = {(row["suite"], row["category"]): row for row in rows}
    for suite_index, suite in enumerate(DISPLAY_SUITE):
        if suite_index:
            table.add_section()
        for method_index, (model, label) in enumerate(METHODS):
            table.add_row(
                DISPLAY_SUITE[suite] if method_index == 0 else "",
                label,
                *[
                    _category_cell(indexed[model][suite, category])
                    for category in CATEGORY_ORDER
                ],
            )
    table.add_section()
    for method_index, (model, label) in enumerate(METHODS):
        table.add_row(
            "Avg" if method_index == 0 else "",
            label,
            *[
                _category_average_cell(
                    [indexed[model][suite, category] for suite in DISPLAY_SUITE]
                )
                for category in CATEGORY_ORDER
            ],
        )
    return table


def _load_status(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _selection_cell(status: dict[str, Any]) -> Text:
    valid = status.get("selection_valid")
    count = int(status.get("selected_state_count", 0))
    if valid is None:
        return Text("—", style="dim")
    return Text(f"{count}/16 {'OK' if valid else 'ERR'}", style="green" if valid else "bold red")


def _gap_cell(status: dict[str, Any]) -> Text:
    gaps = status.get("phase_effective_gaps")
    if not gaps:
        return Text("—", style="dim")
    target = int(status.get("target_min_gap", 4))
    relaxed = any(int(gap) < target for gap in gaps)
    return Text("/".join(map(str, gaps)), style="yellow" if relaxed else "green")


def _training_table(output_root: pathlib.Path, seed: int) -> Table:
    table = Table(title=f"PIVOT-Q-v2 train seed {seed}", box=box.SIMPLE, pad_edge=False)
    table.add_column("Suite")
    table.add_column("State")
    table.add_column("Episode", justify="right")
    table.add_column("Step", justify="right")
    table.add_column("Select", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Last success", justify="right")
    table.add_column("ETA", justify="right")
    table.add_column("Heartbeat", justify="right")
    base = (
        output_root
        / "train"
        / "libero-plus"
        / "pivot_q-v2-s16-shared560-first20"
        / f"seed-{seed:03d}"
    )
    now = time.time()
    for suite in DISPLAY_SUITE:
        suite_root = base / suite
        candidates = [path for path in [suite_root / "status.json"] if path.is_file()]
        candidates.extend(suite_root.glob("*/status.json"))
        candidates.sort(key=lambda path: path.stat().st_mtime)
        status = _load_status(candidates[-1]) if candidates else None
        if status is None:
            table.add_row(
                DISPLAY_SUITE[suite], "WAIT", "0/140", "0/700", "—", "—", "—", "—", "—"
            )
            continue
        age = max(0.0, now - float(status.get("updated_at", now)))
        last_success = status.get("last_episode_success")
        table.add_row(
            DISPLAY_SUITE[suite],
            str(status.get("status", "unknown")).upper(),
            f'{status.get("episode", 0)}/{status.get("episodes_total", 140)}',
            f'{status.get("optimizer_step", 0)}/{status.get("optimizer_steps_total", 700)}',
            _selection_cell(status),
            _gap_cell(status),
            "—" if last_success is None else ("yes" if last_success else "no"),
            format_duration(status.get("eta_seconds")),
            format_duration(age),
        )
    return table


def _render(args: argparse.Namespace, repo_root: pathlib.Path) -> Group:
    snapshot = build_snapshot(
        repo_root,
        eta_window=args.eta_window,
        manifest_path=args.manifest,
        selected_models=[model for model, _ in METHODS],
        selected_benchmarks=["libero-plus"],
        output_root=args.output_root,
        run_name=args.run_name,
        pivot_q_train_seed=args.pivot_q_train_seed,
        legacy_fallback=False,
    )
    return Group(
        _overview(snapshot),
        _suite_table(snapshot),
        _suite_category_table(snapshot),
        _training_table(args.output_root.resolve(), args.pivot_q_train_seed),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact PIVOT-Q-v2 train/eval monitor")
    parser.add_argument("--repo-root", type=pathlib.Path, default=None)
    parser.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("output"))
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("configs/libero_plus/shared560-first20.json"),
    )
    parser.add_argument("--pivot_q-train-seed", type=int, default=0)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--eta-window", type=int, default=20)
    parser.add_argument(
        "--watch",
        "--refresh-seconds",
        dest="watch",
        type=float,
        default=0.0,
        help="Refresh interval; zero prints once",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = (args.repo_root or pathlib.Path(__file__).resolve().parents[1]).resolve()
    if not args.output_root.is_absolute():
        args.output_root = repo_root / args.output_root
    if not args.manifest.is_absolute():
        args.manifest = repo_root / args.manifest
    if args.once:
        args.watch = 0.0
    console = Console()
    if args.watch <= 0:
        console.print(_render(args, repo_root))
        return
    with Live(_render(args, repo_root), console=console, refresh_per_second=4) as live:
        while True:
            time.sleep(args.watch)
            live.update(_render(args, repo_root))


if __name__ == "__main__":
    main()
