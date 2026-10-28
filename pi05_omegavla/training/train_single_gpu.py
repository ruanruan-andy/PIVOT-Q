"""HoloQ pi0.5 uses the shared 4090 worker and its HoloQ bundle loader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pi05_quantvla.pivot_q.train_single_gpu import main
if __name__ == "__main__":
    main()
