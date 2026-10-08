"""Backward-compatible wrapper. Prefer the installed CLI: `swingtrace swing.mp4`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from swingtrace.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--video" in args:  # old flag style: track.py --video X
        i = args.index("--video")
        args = [args[i + 1]] + args[:i] + args[i + 2:]
    main(args)
