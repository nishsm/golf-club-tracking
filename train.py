"""Fine-tune YOLO11 on the golf-club-tracking dataset.

Defaults match the run that produced weights/best.pt: YOLO11m, 640 px,
batch 2 with mixed precision so it fits an 8 GB laptop GPU, up to 600 epochs
with early stopping (patience 100).

    # download the dataset from Roboflow in "YOLOv11" format into data/
    python train.py --data data/data.yaml
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default=None, help="0 for CUDA, mps for Apple silicon, cpu")
    a = p.parse_args()

    YOLO(a.model).train(
        data=a.data,
        epochs=a.epochs,
        patience=a.patience,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        amp=True,
    )


if __name__ == "__main__":
    main()
