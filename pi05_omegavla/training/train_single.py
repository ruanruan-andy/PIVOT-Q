#!/usr/bin/env python3
"""Shared single-device worker; use run_train_single.py for managed services."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pi05_quantvla.pivot_q.train_single import main

if __name__ == "__main__":
    main()
