import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from golfclub.trajectory import (  # noqa: E402
    bbox_center,
    fill_head_from_shaft,
    interpolate_gaps,
    reconstruct_path,
    shaft_end_point,
    smooth,
)


def test_bbox_center():
    assert bbox_center((0, 0, 10, 20)) == (5, 10)


def test_shaft_end_picks_corner_nearest_previous_head():
    assert shaft_end_point((0, 0, 100, 200), prev_head=(5, 5)) == (0, 0)
    assert shaft_end_point((0, 0, 100, 200), prev_head=(95, 190)) == (100, 200)
    assert shaft_end_point((0, 0, 100, 200)) == (100, 200)


def test_fill_head_from_shaft_only_fills_missing_frames():
    heads = {0: (10, 10), 3: (40, 40)}
    shafts = {1: (0, 0, 12, 12), 3: (0, 0, 1, 1)}
    out = fill_head_from_shaft(heads, shafts)
    assert out[0] == (10, 10) and out[3] == (40, 40)
    assert out[1] == (12, 12)
    assert 2 not in out


def test_interpolate_fills_every_frame_on_a_line():
    pts = interpolate_gaps([(0, (0, 0)), (2, (20, 20)), (5, (50, 50))])
    assert len(pts) == 6
    assert pts[3] == (30, 30)


def test_smooth_keeps_length_and_flattens_spike():
    pts = [(0, 0)] * 4 + [(50, 50)] + [(0, 0)] * 4
    out = smooth(pts, window=5)
    assert len(out) == len(pts)
    assert out[4] == (10, 10)


def test_reconstruct_path_starts_at_first_detection():
    start, path = reconstruct_path({10: (0, 0), 14: (40, 0)}, {})
    assert start == 10 and len(path) == 5
