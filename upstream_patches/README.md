# Upstream compatibility overrides

These two source files preserve the small compatibility changes used by the
reference installation. Official repositories remain pinned Git submodules.

| Repository | File | Difference from the pinned official source |
|---|---|---|
| LIBERO | `libero/libero/benchmark/__init__.py` | One `torch.load(init_states_path)` call becomes `torch.load(init_states_path, weights_only=False)`; explanatory comments are added. |
| LIBERO-Plus | `libero/libero/benchmark/__init__.py` | Two init-state loading calls explicitly set `weights_only=False`. The task-order message prints the order index and task count rather than the entire index list; task ordering is unchanged. |

The loading change supports benchmark init-state files containing NumPy objects
with PyTorch versions whose default loader restricts such objects. Use only
trusted official benchmark assets. Do not apply this option to untrusted files.

## Installation

From the PIVOT-Q repository:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
git -C "$PIVOT_Q_ROOT" submodule update --init --recursive
python "$PIVOT_Q_ROOT/env/configure_paths.py" "$PIVOT_Q_ROOT"
python "$PIVOT_Q_ROOT/env/configure_paths.py" "$PIVOT_Q_ROOT" --apply
source "$PIVOT_Q_ROOT/env/runtime/paths.sh"
```

The configuration tool checks all five submodule commits against
`manifest.json`. For each overridden file it checks the current contents against
the pinned original or this override before replacing the file with a **relative
symbolic link** to the corresponding file under this directory. Unknown local
changes are rejected before any files are written. Repeated runs are safe.
No backup copy is created; the original file remains in the submodule Git history.

The two submodules consequently show local file-type changes. This is expected:
do not commit these links into the upstream repositories. If you reset a submodule,
run the configuration command again. Relocating the complete repository preserves
the relative links; rerun configuration to update generated absolute paths.

Omega-QVLA, QVLA, and LeRobot have no source overrides. QuantVLA is vendored
separately with its original license and author attribution; see
[third-party provenance](../docs/third_party.md).
