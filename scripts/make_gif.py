"""Make a labelled GIF from one or more track.py output videos, side by side.

    python track.py swing.mp4 --mode all
    python scripts/make_gif.py out/swing boxes path assets/boxes_vs_path.gif   # two panels
    python scripts/make_gif.py out/swing compare assets/compare.gif --width 420  # one panel

Pass the stem (out/<video_stem>), then one or more modes, then the output path.
Labels are drawn with OpenCV, so any ffmpeg build works.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

LABELS = {"boxes": "Detections", "path": "Club path", "both": "Boxes + path",
          "compare": "Raw vs smoothed"}


def label(img, text):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 480, max(1, img.shape[1] // 240)
    (w, h), _ = cv2.getTextSize(text, font, scale, th)
    cv2.rectangle(img, (0, 0), (w + 20, h + 20), (0, 0, 0), -1)
    cv2.putText(img, text, (10, h + 8), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stem")
    p.add_argument("args", nargs="+", help="one or more modes, then the output .gif path")
    p.add_argument("--width", type=int, default=320, help="width of each panel")
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()
    *modes, out = a.args

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (brew install ffmpeg)")
    caps = [cv2.VideoCapture(f"{a.stem}_{m}.mp4") for m in modes]
    for m, c in zip(modes, caps):
        if not c.isOpened():
            raise SystemExit(f"missing {a.stem}_{m}.mp4")
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

    with tempfile.TemporaryDirectory() as tmp:
        mp4, writer = str(Path(tmp) / "tmp.mp4"), None
        while True:
            frames = []
            for c in caps:
                ok, f = c.read()
                if not ok:
                    break
                h = int(f.shape[0] * a.width / f.shape[1]) // 2 * 2
                frames.append(cv2.resize(f, (a.width, h), interpolation=cv2.INTER_AREA))
            if len(frames) < len(caps):
                break
            frames = [label(f, LABELS.get(m, m)) for f, m in zip(frames, modes)]
            row = np.hstack(frames)
            if writer is None:
                writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                         (row.shape[1], row.shape[0]))
            writer.write(row)
        if writer is None:
            raise SystemExit("no frames read")
        writer.release()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        vf = f"fps={a.fps},split[x][y];[x]palettegen=max_colors=64[p];[y][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-filter_complex", vf,
                        "-loop", "0", out], check=True)
    print(f"saved {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
