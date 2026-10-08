# Golf Club Tracking

Detects the golf club shaft, club head and hands in swing videos and draws the club-head swing path, even through the frames where the head blurs out and the detector loses it. On 7 unseen phone videos, the head or shaft is found in **93% of frames**.

An open-source project I built to analyse golf swings from ordinary phone video. Free to use: weights, code and evaluation are all here.

<!-- DEMO: add GIFs made from openly licensed swing videos (see assets/README.md) -->

## Results

| Model | Input | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| YOLO11m (fine-tuned) | 640 px | 0.930 | 0.849 | **0.918** | **0.674** |

Validation metrics at the best epoch (456 of 556; training stopped early with patience 100). About 48 hours of training on a laptop RTX 4060 (8 GB), using batch size 2 and mixed precision to fit in memory.

![Training curves](assets/training_curves.png)

The full per-epoch history is in [`results/training_history.json`](results/training_history.json).

### How solid is 0.918?

Treat the validation number as an upper bound. The dataset is made of frames pulled from swing videos and split at random, so most validation frames have a near-identical neighbour in the training set. In the local copy of the dataset, **91% of validation frames have a training frame within 2 frame numbers**, and 98% have one within 5. A score on those frames mostly measures how well the model handles footage like what it already saw.

The fair test is whole swing videos the model never trained on. [`evaluate.py`](evaluate.py) measures how often the head (and the head or shaft) is found in each video, and the longest gap the spline has to bridge:

```bash
python evaluate.py --videos path/to/unseen_swings --device mps
```

**Result on 7 unseen phone videos (3,018 frames, conf 0.5):**

| | Share of frames |
|---|---|
| Club head detected directly | **71.6%** |
| Head **or** shaft detected (so the head can be recovered from the shaft) | **93.4%** |

| Video | Frames | Head | Head or shaft | Longest head gap (frames) |
|---|---|---|---|---|
| Indoor studio, face-on* | 644 | 48.4% | 99.4% | 62 |
| Range 1 | 278 | 37.8% | 86.0% | 49 |
| Range 2 | 399 | 89.0% | 95.2% | 11 |
| Range 3 | 527 | 87.7% | 92.8% | 19 |
| Range 4 | 468 | 87.4% | 96.8% | 14 |
| Simulator bay | 391 | 71.4% | 84.1% | 40 |
| Outdoor, backlit | 311 | 77.2% | 92.9% | 20 |

These are real phone recordings of golf swings, and none of them are in the training dataset. The gap between the first two numbers is why the shaft fallback matters: the head alone is missing in more than a quarter of frames, but the shaft covers most of those. \*In the indoor studio video, the model also boxes a ceiling light as a shaft, so its head-or-shaft figure is inflated. One more caveat: "detected" means a box above the confidence threshold. These videos have no hand labels, so the count includes the occasional false positive. Per-video numbers are in [`results/video_eval.csv`](results/video_eval.csv).

## How it works

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
                         swing path drawn on the video
```

1. **Detection.** I fine-tuned YOLO11m on the open [golf-club-tracking dataset](https://universe.roboflow.com/club-head-tracking/golf-club-tracking) (CC BY 4.0, by the club-head-tracking workspace on Roboflow Universe), with three classes: shaft, club head and hands. Early YOLOv8n runs on a MacBook (Apple MPS, a few epochs each) topped out around 0.53 mAP@50. Moving to YOLO11m on a GPU and training much longer is what got it to 0.92.
2. **Recovering missed heads.** The club head is small and moves fastest at the top of the backswing and through impact, so that's where it drops out. The shaft is longer and easier to see. When the head is missing but the shaft is detected, the head is estimated as the corner of the shaft box closest to the head's last known position.
3. **Gap filling and smoothing.** The remaining gaps are filled with a cubic spline over frame index, then smoothed with a centered moving average, so the result is one continuous swing path.
4. **Two ways to run it.** Use the local weights through Ultralytics, or call the same model deployed as a hosted Roboflow workflow. The original scripts used the hosted Roboflow workflow.

## Quick start

```bash
git clone https://github.com/nishsm/golf-club-tracking.git
cd golf-club-tracking
pip install -r requirements.txt

# boxes + swing path, using the included weights (CPU, CUDA, or --device mps on Apple silicon)
python track.py --video path/to/swing.mp4

# raw detections (yellow) vs the interpolated + smoothed path (green)
python track.py --video path/to/swing.mp4 --mode compare

# just the boxes
python track.py --video path/to/swing.mp4 --mode boxes

# just the path
python track.py --video path/to/swing.mp4 --mode path
```

Output goes to `out/<video>_<mode>.mp4`, and the script prints how many frames had a direct head detection and how many were filled in from the shaft.

**Hosted model (Roboflow):**

```bash
pip install inference
cp .env.example .env   # add your key
export ROBOFLOW_API_KEY=...
python track.py --video path/to/swing.mp4 --backend roboflow
```

**Retrain:** export the dataset from Roboflow in YOLOv11 format into `data/`, then run:

```bash
python train.py --data data/data.yaml --device 0
```

## Repo layout

```
track.py                  CLI: detect, reconstruct and draw the swing path
evaluate.py               per-video detection coverage on unseen swings
train.py                  YOLO11 fine-tuning with the settings used for best.pt
src/golfclub/detect.py    local (Ultralytics) and Roboflow detection backends
src/golfclub/trajectory.py  shaft-based head estimate, spline gap fill, smoothing
tests/                    unit tests for the trajectory logic (pytest)
weights/best.pt           trained YOLO11m weights (40 MB)
results/                  training history and final metrics
```

## Limitations and next steps

- Works on one swing per clip, filmed by a single static phone camera. Moving cameras and multiple golfers in frame haven't been tested.
- The shaft-corner estimate assumes the shaft runs corner to corner across its box. It breaks down when the shaft is close to vertical or horizontal.
- The local backend keeps every frame in memory for a second drawing pass. That's fine for a few-second swing clip, but not for long videos.
- Next steps: a motion model such as a Kalman filter in place of offline splines so the path can be drawn live, and swing metrics (plane, tempo) computed from the path.

## Tech

Python, Ultralytics YOLO11, PyTorch, OpenCV, SciPy, Roboflow (labeling, hosted inference).

## Credits

- Dataset: [golf-club-tracking](https://universe.roboflow.com/club-head-tracking/golf-club-tracking) by club-head-tracking on Roboflow Universe, CC BY 4.0.
- Base model: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

## License

Ultralytics YOLO is AGPL-3.0, and these weights are fine-tuned from it, so this repo is released under AGPL-3.0 as well.
