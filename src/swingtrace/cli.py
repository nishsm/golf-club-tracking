"""SwingTrace command line.

    swingtrace swing.mp4                     # boxes + club path (default)
    swingtrace swing.mp4 --mode path
    swingtrace swing.mp4 --mode all          # writes all four modes
    swingtrace swing.mp4 --backend roboflow  # hosted model, needs ROBOFLOW_API_KEY
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import __version__
from .render import MODES, render, track


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="swingtrace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--mode", choices=[*MODES, "all"], default="both",
                   help="boxes | path | both (default) | compare | all")
    p.add_argument("--out", default=None, help="output .mp4 (default: out/<video>_<mode>.mp4)")
    p.add_argument("--weights", default=None, help="local .pt; default downloads from Hugging Face")
    p.add_argument("--device", default=None, help="cpu, 0 (CUDA) or mps (Apple silicon)")
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--window", type=int, default=5, help="smoothing window in frames")
    p.add_argument("--backend", choices=["local", "roboflow"], default="local")
    p.add_argument("--workspace", default="trial-z8ely")
    p.add_argument("--workflow", default="custom-workflow-2")
    p.add_argument("--version", action="version", version=f"swingtrace {__version__}")
    a = p.parse_args(argv)

    modes = MODES if a.mode == "all" else (a.mode,)
    for m in modes:
        out = a.out if (a.out and len(modes) == 1) else f"out/{Path(a.video).stem}_{m}.mp4"
        if a.backend == "local":
            r = track(a.video, out, mode=m, weights=a.weights, device=a.device,
                      conf=a.conf, window=a.window)
        else:
            from .detect import roboflow_frames
            cap = cv2.VideoCapture(a.video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()
            r = render(roboflow_frames(a.video, a.workspace, a.workflow), out, fps=fps,
                       mode=m, conf=a.conf, window=a.window)
        print(f"[{m}] {r.summary()}\nsaved {r.out}")


if __name__ == "__main__":
    main()
