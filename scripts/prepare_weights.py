"""Copy trained model weights into the algo package.

Usage
-----
Option A — use existing single-subject weights (quick test):
    python scripts/prepare_weights.py --weights-src <path/to/best_model.pt>

Option B — train cross-subject model first, then copy:
    python scripts/train_cross_subject.py --data-dir <chbmit_root> --out-dir /tmp/mvafa_allsubj
    python scripts/prepare_weights.py --weights-src /tmp/mvafa_allsubj/best_model.pt

The weights are placed at:
    algo/mvafa_szcore/weights/best_model.pt
which is baked into the Docker image at build time.
"""
import argparse
import shutil
from pathlib import Path

DST = Path(__file__).parent.parent / "algo" / "mvafa_szcore" / "weights" / "best_model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-src", required=True,
        help="Path to best_model.pt from a training run",
    )
    args = parser.parse_args()

    src = Path(args.weights_src)
    if not src.exists():
        raise FileNotFoundError(f"Weights not found: {src}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, DST)
    print(f"[OK] Weights copied: {src} → {DST}")
    print(f"     File size: {DST.stat().st_size / 1e6:.1f} MB")
    print("\nNext step: build the Docker image:")
    print("  cd <submission_dir>")
    print("  docker build -t YOUR_DOCKERHUB_USERNAME/mv-afa-szcore:v1.0.0 .")
    print("  docker push YOUR_DOCKERHUB_USERNAME/mv-afa-szcore:v1.0.0")


if __name__ == "__main__":
    main()
