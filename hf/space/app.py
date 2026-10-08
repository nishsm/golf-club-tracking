"""SwingTrace demo: upload a golf swing, get the club path back."""

import os
import subprocess
import tempfile

import cv2
import gradio as gr

import swingtrace
from swingtrace.weights import resolve_weights

MAX_SECONDS = 12
WEIGHTS = resolve_weights()  # downloads once from the Hugging Face Hub
EXAMPLES = [p for p in ["examples/swing.mp4"] if os.path.exists(p)]


def _trim_and_resize(src: str, dst: str, max_side: int = 960) -> None:
    """Keep the first MAX_SECONDS and cap resolution so the free CPU stays fast."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    s = min(1.0, max_side / max(w, h))
    size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(fps * MAX_SECONDS)):
        ok, f = cap.read()
        if not ok:
            break
        out.write(cv2.resize(f, size) if s < 1 else f)
    out.release()


def run(video, mode):
    if video is None:
        raise gr.Error("Upload a swing video first.")
    tmp = tempfile.mkdtemp()
    clip = os.path.join(tmp, "clip.mp4")
    raw = os.path.join(tmp, "traced_raw.mp4")
    final = os.path.join(tmp, "traced.mp4")
    _trim_and_resize(video, clip)
    r = swingtrace.track(clip, out=raw, mode=mode, weights=WEIGHTS, device="cpu")
    # re-encode to H.264 so browsers can play it
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", final], check=True)
    return final, r.summary()


with gr.Blocks(title="SwingTrace") as demo:
    gr.Markdown(
        "# ⛳ SwingTrace\n"
        "Upload a golf swing filmed on a phone. SwingTrace finds the club shaft, head and hands "
        "and draws the club-head path through the whole swing. "
        f"The first {MAX_SECONDS} seconds are processed on a free CPU, so give it about a minute.\n\n"
        "[GitHub](https://github.com/nishsm/swingtrace) · [Model](https://huggingface.co/nishsm/swingtrace)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Video(label="Your swing", sources=["upload"])
            mode = gr.Radio(["both", "path", "boxes", "compare"], value="both", label="Mode")
            btn = gr.Button("Trace my swing", variant="primary")
        with gr.Column():
            out = gr.Video(label="Traced")
            info = gr.Textbox(label="Stats", interactive=False)
    btn.click(run, [inp, mode], [out, info])
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=[inp])
    gr.Markdown("Tips: one golfer, a full swing, phone held still, filmed face-on or from behind.")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
