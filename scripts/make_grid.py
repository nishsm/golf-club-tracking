"""Combine the 4 track.py modes for one video into a labelled 2x2 GIF.

    for m in boxes path both compare; do python track.py --video swing.mp4 --mode $m; done
    python scripts/make_grid.py out/swing assets/modes_grid.gif

Labels are drawn with OpenCV, so any ffmpeg build works (no drawtext needed).
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

MODES = ["boxes", "path", "both", "compare"]


def label(img, text):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 400, max(1, img.shape[1] // 200)
    (w, h), _ = cv2.getTextSize(text, font, scale, th)
    cv2.rectangle(img, (6, 6), (18 + w, 18 + h + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 12 + h), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stem", help="out/<video_stem> (expects <stem>_boxes.mp4 etc.)")
    p.add_argument("out", nargs="?", default="assets/modes_grid.gif")
    p.add_argument("--width", type=int, default=220, help="width of each tile")
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (brew install ffmpeg)")
    caps = [cv2.VideoCapture(f"{a.stem}_{m}.mp4") for m in MODES]
    for m, c in zip(MODES, caps):
        if not c.isOpened():
            raise SystemExit(f"missing {a.stem}_{m}.mp4")
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

    with tempfile.TemporaryDirectory() as tmp:
        grid_mp4 = str(Path(tmp) / "grid.mp4")
        writer = None
        while True:
            frames = []
            for c in caps:
                ok, f = c.read()
                if not ok:
                    break
                h = int(f.shape[0] * a.width / f.shape[1]) // 2 * 2
                frames.append(cv2.resize(f, (a.width, h), interpolation=cv2.INTER_AREA))
            if len(frames) < 4:
                break
            frames = [label(f, m) for f, m in zip(frames, MODES)]
            grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
            if writer is None:
                writer = cv2.VideoWriter(grid_mp4, cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                         (grid.shape[1], grid.shape[0]))
            writer.write(grid)
        if writer is None:
            raise SystemExit("no frames read")
        writer.release()

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        vf = f"fps={a.fps},split[x][y];[x]palettegen=max_colors=128[p];[y][p]paletteuse=dither=bayer"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", grid_mp4, "-filter_complex", vf,
                        "-loop", "0", a.out], check=True)
    print(f"saved {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
