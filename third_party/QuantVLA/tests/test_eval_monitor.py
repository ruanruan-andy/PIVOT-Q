from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_monitor import (  # noqa: E402
    _category_average_cell,
    _category_cell,
    _cell,
    _gap_cell,
    _selection_cell,
    _success_style,
)


def test_success_rate_color_thresholds() -> None:
    assert _success_style(None, 0) == "dim"
    assert _success_style(0.0, 4) == "bold red"
    assert _success_style(0.499, 4) == "bold red"
    assert _success_style(0.5, 4) == "yellow"
    assert _success_style(0.799, 4) == "yellow"
    assert _success_style(0.8, 4) == "bold green"
    assert _success_style(1.0, 4) == "bold green"


def test_metric_cell_keeps_compact_progress_text() -> None:
    empty = _cell({"completed": 0, "evaluated": 0, "total": 20, "success_rate": None})
    partial = _cell({"completed": 5, "evaluated": 5, "total": 20, "success_rate": 0.4})
    complete = _cell({"completed": 20, "evaluated": 20, "total": 20, "success_rate": 0.9})

    assert empty.plain == "0/20" and str(empty.style) == "dim"
    assert partial.plain == "40.0% 5/20" and str(partial.style) == "bold red"
    assert complete.plain == "90.0%" and str(complete.style) == "bold green"


def test_category_cell_fits_compact_monitor() -> None:
    empty = _category_cell(
        {"completed": 0, "evaluated": 0, "total": 20, "success_rate": None}
    )
    partial = _category_cell(
        {"completed": 14, "evaluated": 14, "total": 20, "success_rate": 9 / 14}
    )

    assert empty.plain == "—(0)" and str(empty.style) == "dim"
    assert partial.plain == "64.3%(14)" and str(partial.style) == "yellow"


def test_category_average_uses_success_weighted_episode_total() -> None:
    average = _category_average_cell(
        [
            {"completed": 20, "evaluated": 20, "successes": 18},
            {"completed": 5, "evaluated": 5, "successes": 1},
            {"completed": 0, "evaluated": 0, "successes": 0},
        ]
    )
    assert average.plain == "76.0%(25)"
    assert str(average.style) == "yellow"


def test_training_selection_cells_show_quota_and_effective_gap() -> None:
    status = {
        "selection_valid": True,
        "selected_state_count": 16,
        "target_min_gap": 4,
        "phase_effective_gaps": [4, 4, 3, 4],
    }
    selection = _selection_cell(status)
    gap = _gap_cell(status)

    assert selection.plain == "16/16 OK" and str(selection.style) == "green"
    assert gap.plain == "4/4/3/4" and str(gap.style) == "yellow"
