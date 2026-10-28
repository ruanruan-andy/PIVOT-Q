"""Launch 4090 HoloQ pi0.5 recovery with managed simulator services."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pi05_quantvla.pivot_q.run_train_single import main
if __name__ == "__main__":
    main(backend="holoq", worker="train_single_gpu.py")
