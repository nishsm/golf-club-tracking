---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
base_model: Ultralytics/YOLO11
tags:
  - golf
  - golf-club-detection
  - golf-club-tracking
  - golf-swing
  - golf-swing-analysis
  - club-head-detection
  - club-shaft-detection
  - swing-path
  - object-detection
  - object-tracking
  - computer-vision
  - sports
  - sports-analytics
  - yolo
  - yolo11
  - ultralytics
model-index:
  - name: swingtrace-yolo11m
    results:
      - task:
          type: object-detection
        dataset:
          name: golf-club-tracking (validation split)
          type: roboflow-universe
        metrics:
          - type: precision
            value: 0.930
          - type: recall
            value: 0.849
          - type: mAP@50
            value: 0.918
          - type: mAP@50-95
            value: 0.674
---

# ⛳ SwingTrace: Golf Club, Club Head & Shaft Detection (YOLO11m)

**Open-source golf swing tracer. Drop in any phone video, get the club path.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nishsm/golf-club-detection-swingtrace/blob/main/notebooks/swingtrace_quickstart.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-nishsm%2Fgolf--club--detection--swingtrace-black?logo=github)](https://github.com/nishsm/golf-club-detection-swingtrace)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

<p align="center">
  <img src="assets/boxes_vs_path.gif" width="520" alt="Same swing: detections on the left, reconstructed club path on the right">
</p>

*Same swing, two views. Left: what the model detects (shaft, head, hands). Right: the club-head path rebuilt from those detections. Face blurred for privacy.*

<details>
<summary><b>See what the reconstruction adds</b></summary>

<p align="center">
  <img src="assets/compare.gif" width="300" alt="Raw detections (yellow) vs smoothed club path (green)">
</p>

*Raw per-frame detections (yellow) vs the smoothed, gap-filled path (green).*

</details>

This model finds the **club shaft, club head and hands** in every frame of a golf swing video. The [SwingTrace](https://github.com/nishsm/golf-club-detection-swingtrace) package uses it to rebuild the **club-head path through the whole swing**, including the top of the backswing and impact, where the head turns into a blur and normal detectors lose it. Use it for golf club detection, club head tracking and golf swing path analysis from a single phone video.

- 🎯 Tracks the club in **93%** of frames on phone videos it has never seen
- 📱 One phone camera. No sensors, no launch monitor, no markers
- ⚡ Runs on a laptop: CPU, NVIDIA GPU or Apple silicon
- 🆓 Weights, code, training setup and evaluation are all open

## 🚀 Try it in 10 seconds

**1. In your browser (no install):** open the **[Colab notebook](https://colab.research.google.com/github/nishsm/golf-club-detection-swingtrace/blob/main/notebooks/swingtrace_quickstart.ipynb)**, upload a swing clip, press run. Free GPU, nothing to install.

**2. On your machine:**

```bash
pip install git+https://github.com/nishsm/golf-club-detection-swingtrace.git
swingtrace my_swing.mp4                # boxes + club path → out/my_swing_both.mp4
swingtrace my_swing.mp4 --mode all     # all four modes
```

The 40 MB weights download automatically from this repo the first time you run it.

**3. From Python:**

```python
import swingtrace

result = swingtrace.track("my_swing.mp4", mode="path", device="mps")  # "cpu", "cuda" or "mps"
print(result.summary())
```

**4. Just the detector, with Ultralytics:**

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("nishsm/golf-club-detection-swingtrace", "swingtrace-yolo11m.pt"))
results = model.predict("my_swing.mp4", conf=0.25)
```

Raw detections give you boxes only. For the reconstructed club path, use the `swingtrace` package or the Colab notebook.

### Output modes

| Mode | What you get |
|---|---|
| `both` (default) | Boxes for shaft, head and hands, plus the club-head path |
| `path` | Just the smoothed club-head path |
| `boxes` | Just the detections |
| `compare` | Raw detections (yellow) next to the smoothed path (green) |

**Tips for best results:** one golfer, a full swing, a phone on a tripod or held still, filmed face-on or from behind the golfer.

## 🏷️ Classes

| id | label in model | meaning |
|---|---|---|
| 0 | `0` | club shaft |
| 1 | `1` | club head |
| 2 | `3` | hands / grip |

The raw label strings come from the source dataset.

## 📊 How good is it?

On **7 unseen phone videos (3,018 frames)**:

| | Share of frames |
|---|---|
| Club head detected directly | **71.6%** |
| Head or shaft detected, so the head can be recovered from the shaft | **93.4%** |

(conf 0.5. These clips have no hand labels, so counts include occasional false positives.)

**Validation set:** precision 0.930 · recall 0.849 · mAP@50 0.918 · mAP@50-95 0.674 (YOLO11m, 640 px, best of 556 epochs).

> Treat the validation numbers as an **upper bound**, not a headline. The dataset is frames from swing videos split at random, and 91% of validation frames have a near-identical neighbour within 2 frames in the training set. That's why the unseen-video test above exists.

## 🧠 How the path is built

```
video ──► YOLO11m (shaft / head / hands) ──► per-frame boxes
                                                │
                head missed, shaft found? ──────┤  estimate head = shaft-box corner
                                                │  nearest the last known head
                                                ▼
                         cubic spline over frame index (fills remaining gaps)
                                                ▼
                         moving-average smoothing (5 frames)
                                                ▼
                         club path drawn on the video
```

1. **Detect.** This model finds the shaft, club head and hands.
2. **Recover.** The head is small and moves fastest at the top and at impact, so that's where detection drops out. The shaft is bigger and easier to see. When only the shaft is found, the head is placed at the corner of the shaft box closest to where the head was last seen.
3. **Fill and smooth.** A cubic spline bridges remaining gaps, and a moving average smooths the result into one continuous path.

## 🏋️ Training

- **Base:** YOLO11m (Ultralytics 8.3)
- **Data:** golf-club-tracking by club-head-tracking on Roboflow Universe (CC BY 4.0)
- **Setup:** 640 px, batch 2, mixed precision, early stopping (patience 100). Best epoch 456 of 556, about 48 h on an RTX 4060 laptop GPU (8 GB)

![Training curves](assets/training_curves.png)

## ⚠️ Limitations

- Built for **one golfer and one swing per clip, from a static camera**. Moving cameras and multiple people haven't been tested.
- It sometimes boxes thin bright objects (for example a ceiling light) as a shaft.
- The shaft-corner head estimate breaks down when the shaft is close to vertical or horizontal.
- Frames are held in memory for the drawing pass: fine for swing clips, not for long videos.

## 🗺️ Roadmap

- Cut the path off when the swing ends, instead of tracing the walk-off
- Swing metrics from the path: tempo, swing plane, club speed estimate
- Live tracking with a Kalman filter
- ONNX / CoreML export for on-phone use
- Ball tracking

## 🙌 Credits & license

- Dataset: golf-club-tracking by club-head-tracking on Roboflow Universe, CC BY 4.0.
- Base model: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).
- License: **AGPL-3.0**, following Ultralytics YOLO, which these weights are fine-tuned from.

If this is useful, a ❤️ here or a ⭐ on [GitHub](https://github.com/nishsm/golf-club-detection-swingtrace) helps other people find it. Issues and PRs welcome, especially clips where it fails.
