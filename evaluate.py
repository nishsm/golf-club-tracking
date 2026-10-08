"""Measure how often the club is found on real swing videos.

Validation mAP comes from single frames that sit right next to training frames
(see README, "How solid is 0.918?"), so it overstates performance on new
videos. This script measures what matters for tracking: on whole, unseen
swing videos, in what share of frames do we see the head, and how long are
the gaps the spline has to bridge?

    python evaluate.py --videos path/to/folder --device mps
    # writes results/video_eval.csv and prints a summary table
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from golfclub.detect import HEAD, SHAFT, local_frames  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def longest_gap(hit: list[bool]) -> int:
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1
        best = max(best, cur)
    return best


def eval_video(path: Path, weights: str, conf: float, device: str | None) -> dict:
    head, shaft, either = [], [], []
    for _, _, dets in local_frames(str(path), weights, conf=conf, device=device):
        h = any(c == HEAD and s >= conf for _, c, s in dets)
        sh = any(c == SHAFT and s >= conf for _, c, s in dets)
        head.append(h)
        shaft.append(sh)
        either.append(h or sh)
    n = len(head) or 1
    return {
        "video": path.name,
        "frames": len(head),
        "head_pct": round(100 * sum(head) / n, 1),
        "shaft_pct": round(100 * sum(shaft) / n, 1),
        "head_or_shaft_pct": round(100 * sum(either) / n, 1),
        "longest_head_gap": longest_gap(head),
        "longest_head_or_shaft_gap": longest_gap(either),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="folder of swing videos (not used in training)")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/video_eval.csv")
    a = p.parse_args()

    vids = sorted(v for v in Path(a.videos).iterdir() if v.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit(f"no videos in {a.videos}")
    rows = []
    for v in vids:
        r = eval_video(v, a.weights, a.conf, a.device)
        rows.append(r)
        print(f"{r['video'][:40]:40s} frames {r['frames']:4d}  head {r['head_pct']:5.1f}%  "
              f"head|shaft {r['head_or_shaft_pct']:5.1f}%  longest head gap {r['longest_head_gap']}")

    tot = sum(r["frames"] for r in rows) or 1
    w = lambda k: sum(r[k] * r["frames"] for r in rows) / tot  # noqa: E731
    print(f"\n{len(rows)} videos, {tot} frames, conf {a.conf}: head {w('head_pct'):.1f}%, "
          f"head or shaft {w('head_or_shaft_pct'):.1f}% of frames")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
