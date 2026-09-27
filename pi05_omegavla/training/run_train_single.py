#!/usr/bin/env python3
"""Single-GPU HoloQ-VLA recovery launcher."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pi05_quantvla.pivot_q.run_train_single import main

if __name__ == "__main__":
    main(backend="holoq")
