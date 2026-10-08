---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
base_model: Ultralytics/YOLO11
tags:
  - golf
  - sports
  - sports-analytics
  - yolo11
  - ultralytics
  - object-detection
  - object-tracking
  - swing-analysis
---

# ⛳ SwingTrace: golf club detector (YOLO11m)

Detects the **club shaft, club head and hands** in golf swing videos. Used by [SwingTrace](https://github.com/nishsm/swingtrace) to rebuild the club-head path through the whole swing, including the blurry frames at the top and at impact.

**Try it:** [live demo](https://huggingface.co/spaces/nishsm/swingtrace) · **Code:** [github.com/nishsm/swingtrace](https://github.com/nishsm/swingtrace)

## Use it

```bash
pip install git+https://github.com/nishsm/swingtrace.git
swingtrace my_swing.mp4            # boxes + club path
```

Or just the detector, with Ultralytics:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("nishsm/swingtrace", "swingtrace-yolo11m.pt"))
results = model.predict("my_swing.mp4", conf=0.25)
```

## Classes

| id | label | meaning |
|---|---|---|
| 0 | `0` | shaft |
| 1 | `1` | club head |
| 2 | `3` | hands / grip |

The raw label strings come from the source dataset.

## Results

**Unseen phone videos (7 clips, 3,018 frames, conf 0.5):** club head detected in **71.6%** of frames; head or shaft (so the head can be recovered from the shaft) in **93.4%**. These clips have no hand labels, so the counts include occasional false positives.

**Validation set:** precision 0.930 · recall 0.849 · mAP@50 0.918 · mAP@50-95 0.674.

Treat the validation mAP as an upper bound. The dataset's frames come from videos and were split at random, and 91% of validation frames have a training frame within 2 frames of them.

## Training

- Base: YOLO11m (Ultralytics 8.3)
- Data: [golf-club-tracking](https://universe.roboflow.com/club-head-tracking/golf-club-tracking) by club-head-tracking on Roboflow Universe (CC BY 4.0)
- 640 px, batch 2, mixed precision, early stopping (patience 100): best epoch 456 of 556, about 48 h on an RTX 4060 laptop GPU (8 GB)

## Limitations

Trained and tested on single-golfer clips from a static phone camera. Expect worse results with moving cameras, several people in frame, or unusual angles. It sometimes boxes thin bright objects (for example a ceiling light) as a shaft.

## License

AGPL-3.0, following Ultralytics YOLO, which these weights are fine-tuned from.
