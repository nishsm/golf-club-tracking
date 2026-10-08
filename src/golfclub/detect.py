"""Detection backends.

* ``local``: run the trained YOLO11m weights with Ultralytics (CPU, CUDA or Apple MPS).
* ``roboflow``: run the same model through a hosted Roboflow workflow.
  Needs ``ROBOFLOW_API_KEY`` in the environment; the key is never stored in code.

Both yield, per frame: (frame_index, frame_bgr, [(bbox, class_id, confidence), ...]).
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import numpy as np

Det = Tuple[Tuple[int, int, int, int], int, float]
FrameResult = Tuple[int, np.ndarray, List[Det]]

# Class ids in the golf-club-tracking dataset (Roboflow, v2)
# (Roboflow exports the names as "0", "1", "3"; ids are 0, 1, 2.)
SHAFT, HEAD, HANDS = 0, 1, 2
CLASS_STYLE = {
    SHAFT: ("shaft", (0, 0, 255)),
    HEAD: ("head", (0, 255, 0)),
    HANDS: ("hands", (255, 0, 0)),
}


def local_frames(
    video: str, weights: str, conf: float = 0.25, imgsz: int = 640, device: str | None = None
) -> Iterator[FrameResult]:
    from ultralytics import YOLO

    model = YOLO(weights)
    stream = model.predict(
        source=video, stream=True, conf=conf, imgsz=imgsz, device=device, verbose=False
    )
    for i, r in enumerate(stream):
        dets: List[Det] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            dets = [(tuple(b), int(c), float(s)) for b, c, s in zip(xyxy, cls, cf)]
        yield i, r.orig_img, dets


def roboflow_frames(
    video: str, workspace: str, workflow_id: str, max_fps: int = 30
) -> Iterator[FrameResult]:
    """Collect predictions from a Roboflow InferencePipeline workflow."""
    from inference import InferencePipeline

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (see .env.example) to use the roboflow backend.")

    results: List[FrameResult] = []

    def sink(result, video_frame):
        dets: List[Det] = []
        preds = result.get("predictions")
        if preds is not None and len(preds):
            for b, c, s in zip(preds.xyxy, preds.class_id, preds.confidence):
                dets.append((tuple(int(v) for v in b), int(c), float(s)))
        results.append((len(results), video_frame.image.copy(), dets))

    pipe = InferencePipeline.init_with_workflow(
        api_key=api_key,
        workspace_name=workspace,
        workflow_id=workflow_id,
        video_reference=video,
        max_fps=max_fps,
        on_prediction=sink,
    )
    pipe.start()
    pipe.join()
    yield from results
