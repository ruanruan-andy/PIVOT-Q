"""Locate named Conda environments without assuming an installation prefix."""
import json
import os
from pathlib import Path
import subprocess
import sys


def conda_python(name):
    if os.environ.get("CONDA_DEFAULT_ENV") == name:
        return Path(sys.executable)
    conda = os.environ.get("CONDA_EXE", "conda")
    data = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))
    matches = [Path(p) / "bin/python" for p in data["envs"]
               if Path(p).name == name and (Path(p) / "bin/python").is_file()]
    matches = list(dict.fromkeys(p.resolve() for p in matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Conda environment named {name}, found {len(matches)}")
    return matches[0]
