"""Detect the club and draw the club-head swing path on a video.

Examples
--------
    # local weights (no API key needed)
    python track.py --video swing.mp4 --out out/swing_path.mp4

    # raw vs interpolated path, side by side colours
    python track.py --video swing.mp4 --mode compare

    # through the hosted Roboflow workflow
    export ROBOFLOW_API_KEY=...
    python track.py --video swing.mp4 --backend roboflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from golfclub.detect import CLASS_STYLE, HEAD, SHAFT, local_frames, roboflow_frames  # noqa: E402
from golfclub.trajectory import bbox_center, fill_head_from_shaft, reconstruct_path  # noqa: E402

PATH_COLOR = (0, 255, 0)   # smoothed path (green)
RAW_COLOR = (0, 255, 255)  # raw detections + shaft estimates (yellow)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="output .mp4 (default: out/<name>_<mode>.mp4)")
    p.add_argument("--mode", choices=["boxes", "path", "compare"], default="path",
                   help="boxes: detections only; path: smoothed head path; compare: raw (yellow) vs smoothed (green)")
    p.add_argument("--backend", choices=["local", "roboflow"], default="local")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--device", default=None, help="cpu, 0 (CUDA), or mps")
    p.add_argument("--conf", type=float, default=0.7,
                   help="confidence threshold for boxes and path points")
    p.add_argument("--window", type=int, default=5, help="smoothing window (frames)")
    p.add_argument("--workspace", default="trial-z8ely")
    p.add_argument("--workflow", default="custom-workflow-2")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = Path(a.out or f"out/{Path(a.video).stem}_{a.mode}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.backend == "local":
        frames = local_frames(a.video, a.weights, conf=min(a.conf, 0.25), device=a.device)
    else:
        frames = roboflow_frames(a.video, a.workspace, a.workflow)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Pass 1: collect detections (frames kept for drawing).
    images, dets_per_frame = [], []
    heads, shafts = {}, {}
    for i, img, dets in frames:
        images.append(img)
        dets_per_frame.append(dets)
        for bbox, cid, conf in dets:
            if conf < a.conf:
                continue
            if cid == HEAD:
                heads[i] = bbox_center(bbox)
            elif cid == SHAFT:
                shafts[i] = bbox
    if not images:
        raise SystemExit(f"No frames read from {a.video}")

    start, path = reconstruct_path(heads, shafts, a.window)
    raw = fill_head_from_shaft(heads, shafts)
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Pass 2: draw.
    for i, img in enumerate(images):
        frame = img.copy()
        if a.mode == "boxes":
            for (x1, y1, x2, y2), cid, conf in dets_per_frame[i]:
                if conf >= a.conf and cid in CLASS_STYLE:
                    label, color = CLASS_STYLE[cid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            if a.mode == "compare":
                pts = [raw[f] for f in sorted(raw) if f <= i]
                for p0, p1 in zip(pts, pts[1:]):
                    cv2.line(frame, p0, p1, RAW_COLOR, 2)
            upto = path[: max(0, i - start + 1)]
            for p0, p1 in zip(upto, upto[1:]):
                cv2.line(frame, p0, p1, PATH_COLOR, 3)
            if upto:
                cv2.circle(frame, upto[-1], 7, PATH_COLOR, -1)
        writer.write(frame)
    writer.release()

    covered = len(heads)
    print(f"head detected in {covered}/{len(images)} frames "
          f"({100 * covered / len(images):.0f}%), {len(raw) - covered} filled from shaft, "
          f"path spans {len(path)} frames")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
