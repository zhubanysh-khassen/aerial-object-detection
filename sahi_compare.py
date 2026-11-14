"""Plain YOLO inference vs SAHI sliced inference on VisDrone val images.

Objects seen from a drone are tiny. Slicing the frame into overlapping tiles
lets the detector look at them "closer", at the cost of speed.
This script measures that trade-off with a simple, transparent metric:
precision / recall at IoU >= 0.5, overall and for small objects (< 32x32 px).

Usage:
    python sahi_compare.py --weights results/best.pt --n 200
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

DATA_ROOT = Path(os.environ.get("CV_DATA", Path.home() / "cv-data"))
VAL_IMAGES = DATA_ROOT / "datasets" / "VisDrone" / "images" / "val"
OUT = Path(__file__).parent / "results"
SMALL = 32 * 32


def load_gt(img_path, w, h):
    label = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
    boxes = []
    if label.exists():
        for line in label.read_text().split("\n"):
            if line.strip():
                c, xc, yc, bw, bh = map(float, line.split())
                boxes.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def match(preds, gts):
    """Greedy matching by confidence. Returns TP count and per-GT matched flags."""
    used = [False] * len(gts)
    tp = 0
    for cls, score, box in sorted(preds, key=lambda p: -p[1]):
        best, best_j = 0.5, -1
        for j, (gc, *gbox) in enumerate(gts):
            if not used[j] and gc == cls:
                v = iou(box, gbox)
                if v >= best:
                    best, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    return tp, used


def to_preds(result):
    return [(p.category.id, p.score.value, p.bbox.to_xyxy()) for p in result.object_prediction_list]


def evaluate(images, run):
    tp = n_pred = n_gt = 0
    small_gt = small_hit = 0
    t = 0.0
    for img in images:
        h, w = cv2.imread(str(img)).shape[:2]
        gts = load_gt(img, w, h)
        start = time.perf_counter()
        preds = to_preds(run(str(img)))
        t += time.perf_counter() - start
        k, used = match(preds, gts)
        tp += k
        n_pred += len(preds)
        n_gt += len(gts)
        for (c, x1, y1, x2, y2), hit in zip(gts, used):
            if (x2 - x1) * (y2 - y1) < SMALL:
                small_gt += 1
                small_hit += hit
    return {
        "precision": round(tp / n_pred, 4) if n_pred else 0,
        "recall": round(tp / n_gt, 4) if n_gt else 0,
        "recall_small_objects": round(small_hit / small_gt, 4) if small_gt else 0,
        "ms_per_image": round(1000 * t / len(images), 1),
        "images": len(images), "gt_objects": n_gt, "small_gt_objects": small_gt,
    }


def save_example(img, plain, sliced, out_path):
    frame = cv2.imread(str(img))
    left, right = frame.copy(), frame.copy()
    for canvas, preds, color in ((left, plain, (0, 140, 255)), (right, sliced, (0, 200, 0))):
        for _, _, (x1, y1, x2, y2) in preds:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for canvas, title in ((left, "plain YOLO"), (right, "YOLO + SAHI slicing")):
        cv2.putText(canvas, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.imwrite(str(out_path), np.hstack([left, right]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(OUT / "best.pt"))
    ap.add_argument("--n", type=int, default=200, help="number of val images")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    images = sorted(VAL_IMAGES.glob("*.jpg"))
    random.Random(0).shuffle(images)
    images = images[: a.n]
    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=a.weights,
                                               confidence_threshold=a.conf, device="cuda:0")
    plain = lambda p: get_prediction(p, model)  # noqa: E731
    sliced = lambda p: get_sliced_prediction(p, model, slice_height=a.slice, slice_width=a.slice,  # noqa: E731
                                             overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    report = {"plain": evaluate(images, plain), "sahi": evaluate(images, sliced),
              "settings": {"slice": a.slice, "overlap": 0.2, "conf": a.conf, "iou_match": 0.5}}
    OUT.mkdir(exist_ok=True)
    (OUT / "sahi_compare.json").write_text(json.dumps(report, indent=2))
    ex = images[0]
    save_example(ex, to_preds(plain(str(ex))), to_preds(sliced(str(ex))), OUT / "sahi_example.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
