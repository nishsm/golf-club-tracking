"""Club-head trajectory reconstruction from per-frame detections.

The detector does not find the club head in every frame: it blurs out at the
top of the backswing and through impact. These helpers turn sparse,
noisy detections into a continuous swing path:

1. When the head is missed but the shaft is found, estimate the head as the
   shaft-box corner closest to the last known head position.
2. Fill the remaining gaps with a cubic spline over frame index.
3. Smooth the result with a centered moving average.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def shaft_end_point(bbox: BBox, prev_head: Optional[Point] = None) -> Point:
    """Estimate the club-head end of a shaft bounding box.

    The shaft runs corner to corner across its box, so the head sits at one
    of the four corners. Pick the corner nearest the previous head position;
    with no history, fall back to bottom-right.
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    if prev_head is None:
        return corners[-1]
    dists = [np.hypot(c[0] - prev_head[0], c[1] - prev_head[1]) for c in corners]
    return corners[int(np.argmin(dists))]


def fill_head_from_shaft(
    heads: Dict[int, Point], shafts: Dict[int, BBox]
) -> Dict[int, Point]:
    """Return a copy of ``heads`` with shaft-based estimates for missed frames."""
    out = dict(heads)
    last: Optional[Point] = None
    for f in sorted(set(heads) | set(shafts)):
        if f in out:
            last = out[f]
        elif f in shafts:
            out[f] = shaft_end_point(shafts[f], last)
            last = out[f]
    return out


def interpolate_gaps(known: Sequence[Tuple[int, Point]]) -> List[Point]:
    """Cubic-spline the (frame, point) samples onto every frame in their span."""
    if len(known) < 2:
        return [p for _, p in known]
    known = sorted(known)
    frames = np.array([f for f, _ in known])
    pts = np.array([p for _, p in known], dtype=float)
    cs_x = CubicSpline(frames, pts[:, 0])
    cs_y = CubicSpline(frames, pts[:, 1])
    full = np.arange(frames[0], frames[-1] + 1)
    return [(int(cs_x(f)), int(cs_y(f))) for f in full]


def smooth(points: Sequence[Point], window: int = 5) -> List[Point]:
    """Centered moving average over x and y."""
    if len(points) < window:
        return list(points)
    arr = np.asarray(points, dtype=float)
    half = window // 2
    out = []
    for i in range(len(arr)):
        seg = arr[max(0, i - half) : i + half + 1]
        out.append((int(seg[:, 0].mean()), int(seg[:, 1].mean())))
    return out


def reconstruct_path(
    heads: Dict[int, Point], shafts: Dict[int, BBox], window: int = 5
) -> Tuple[int, List[Point]]:
    """Full pipeline. Returns (first_frame_index, smoothed path)."""
    filled = fill_head_from_shaft(heads, shafts)
    if not filled:
        return 0, []
    start = min(filled)
    return start, smooth(interpolate_gaps(list(filled.items())), window)
