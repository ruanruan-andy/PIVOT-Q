"""LeRobot pi0.5 + Omega-QVLA recovery, isolated from the official checkout."""
import sys
sys.dont_write_bytecode = True

import os
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
