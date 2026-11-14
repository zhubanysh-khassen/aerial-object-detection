"""Train a YOLO detector on VisDrone (drone footage: people, cars, bikes...).

The dataset (~2 GB) is downloaded and converted to YOLO format automatically
by Ultralytics on the first run. Data and training runs live outside the repo
($CV_DATA, default ~/cv-data), only the final weights and metrics are copied here.

Usage:
    python train.py --model yolo26n.pt --epochs 40 --batch 8
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO, settings

DATA_ROOT = Path(os.environ.get("CV_DATA", Path.home() / "cv-data"))
OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)      # fits a 4 GB laptop GPU at 640 px
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="visdrone")
    a = ap.parse_args()

    settings.update({"datasets_dir": str(DATA_ROOT / "datasets"), "runs_dir": str(DATA_ROOT / "runs")})
    model = YOLO(a.model)  # COCO-pretrained weights -> fine-tune on aerial images
    model.train(data="VisDrone.yaml", epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                workers=4, name=a.name, exist_ok=True, patience=15, plots=True)

    best = Path(model.trainer.best)
    metrics = YOLO(best).val(data="VisDrone.yaml", imgsz=a.imgsz, batch=a.batch, plots=True,
                             name=f"{a.name}_val", exist_ok=True)
    OUT.mkdir(exist_ok=True)
    shutil.copy(best, OUT / "best.pt")
    names = metrics.names
    per_class = {names[i]: round(float(m), 4) for i, m in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])}
    summary = {
        "base_model": a.model, "epochs": a.epochs, "imgsz": a.imgsz,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50-95_per_class": per_class,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    run_dir = Path(model.trainer.save_dir)
    for f in ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "val_batch0_pred.jpg"]:
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, OUT / f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
