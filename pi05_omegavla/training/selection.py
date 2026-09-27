"""Shared scoring; dense and PIVOT-Q selections differ only at this boundary."""
from pi05_quantvla.pivot_q.selection import action_error, future_risk, select, select_all

def choose(error, risk, *, method, **kwargs):
    if method == "full_distill":
        return select_all(error)
    if method == "pivot_q":
        return select(error, risk, **kwargs)
    raise ValueError(f"unknown selection: {method}")
