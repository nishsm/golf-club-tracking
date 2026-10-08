"""Run detection on a video, rebuild the club-head path, and draw it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2

from .detect import CLASS_STYLE, HEAD, SHAFT, FrameResult
from .trajectory import bbox_center, fill_head_from_shaft, reconstruct_path

MODES = ("boxes", "path", "both", "compare")
PATH_COLOR = (0, 255, 0)   # smoothed path (green)
RAW_COLOR = (0, 255, 255)  # raw detections + shaft estimates (yellow)


@dataclass
class TrackResult:
    out: str
    frames: int
    head_frames: int
    filled_from_shaft: int
    path_frames: int

    @property
    def head_pct(self) -> float:
        return 100 * self.head_frames / max(1, self.frames)

    def summary(self) -> str:
        return (f"head detected in {self.head_frames}/{self.frames} frames ({self.head_pct:.0f}%), "
                f"{self.filled_from_shaft} filled from shaft, path spans {self.path_frames} frames")


def render(
    frames: Iterable[FrameResult],
    out: str,
    fps: float = 30,
    mode: str = "both",
    conf: float = 0.7,
    window: int = 5,
) -> TrackResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect detections.
    images, dets_per_frame, heads, shafts = [], [], {}, {}
    for i, img, dets in frames:
        images.append(img)
        dets_per_frame.append(dets)
        for bbox, cid, c in dets:
            if c < conf:
                continue
            if cid == HEAD:
                heads[i] = bbox_center(bbox)
            elif cid == SHAFT:
                shafts[i] = bbox
    if not images:
        raise SystemExit("no frames read")

    start, path = reconstruct_path(heads, shafts, window)
    raw = fill_head_from_shaft(heads, shafts)
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Pass 2: draw.
    for i, img in enumerate(images):
        frame = img.copy()
        if mode in ("boxes", "both"):
            for (x1, y1, x2, y2), cid, c in dets_per_frame[i]:
                if c >= conf and cid in CLASS_STYLE:
                    label, color = CLASS_STYLE[cid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {c:.2f}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if mode != "boxes":
            if mode == "compare":
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

    return TrackResult(out, len(images), len(heads), len(raw) - len(heads), len(path))


def track(
    video: str,
    out: Optional[str] = None,
    mode: str = "both",
    weights: Optional[str] = None,
    device: Optional[str] = None,
    conf: float = 0.7,
    window: int = 5,
) -> TrackResult:
    """Python API: ``swingtrace.track("swing.mp4", mode="both")``."""
    from .detect import local_frames
    from .weights import resolve_weights

    out = out or f"out/{Path(video).stem}_{mode}.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    frames = local_frames(video, resolve_weights(weights), conf=min(conf, 0.25), device=device)
    return render(frames, out, fps=fps, mode=mode, conf=conf, window=window)
