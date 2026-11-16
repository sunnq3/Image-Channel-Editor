"""Fast known-answer tests:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import evaluate                                     # noqa: E402
from cases import build_cases, synthetic_photo, to_img, wave       # noqa: E402
from patternscan.analyze import analyze, block_boundary_test, lattice_of   # noqa: E402
from patternscan.report import build_report                        # noqa: E402

BASE = synthetic_photo(3, 576, 448)
CASES = {name.split("/", 1)[1]: (data, ext, expect)
         for name, data, ext, expect in build_cases(BASE, "t")}


def run(tmp_path, key):
    data, ext, expect = CASES[key]
    path = tmp_path / f"case.{ext}"
    path.write_bytes(data)
    return evaluate(str(path), expect)


def test_clean_image_reports_nothing(tmp_path):
    hits, extra, res = run(tmp_path, "clean")
    assert res["patterns"] == []


@pytest.mark.parametrize("key", [
    "faint ripple, period 5 px, +-2 levels",
    "faint ripple, period 12.5 px, +-4 levels",
    "visible stripes, period 9 px, +-16 levels",
    "faint 2-D grid (two ripples)",
    "stripes in one region only",
    "stripes in one region, exactly 8 px wide",
    "JPEG quality 50",
    "JPEG quality 95",
    "JPEG q75 re-saved as PNG",
    "enlarged 1.5x (bicubic)",
    "enlarged exactly 2x (bicubic)",
    "rotated 7 degrees",
    "faint ripple +-3 then JPEG q85",
])
def test_planted_pattern_is_found_and_named(tmp_path, key):
    hits, extra, res = run(tmp_path, key)
    assert all(hits), [(p.category, p._rep.period) for p in res["patterns"]]
    assert not extra, [(p.category, p._rep.period) for p in extra]


def test_jpeg_grid_is_not_mistaken_for_an_overlay(tmp_path):
    hits, extra, res = run(tmp_path, "JPEG quality 75")
    cats = {p.category for p in res["patterns"]}
    assert "jpeg_grid" in cats
    assert not cats & {"global_faint", "global_visible", "regional"}


def test_crop_after_jpeg_shifts_the_block_grid(tmp_path):
    hits, extra, res = run(tmp_path, "JPEG q75, cropped by (3,5), saved as PNG")
    jp = res["jpeg"]
    assert jp["detected"]
    assert (jp["offset_x"], jp["offset_y"]) == ((8 - 3) % 8, (8 - 5) % 8)


def test_amplitude_and_period_are_measured(tmp_path):
    ripple, (u, v) = wave(BASE.shape, 7.0, 25, 3.0)
    path = tmp_path / "r.png"
    to_img(BASE + ripple[..., None]).save(path)
    res = analyze(str(path))
    pat = next(p for p in res["patterns"] if p.category == "global_faint")
    assert abs(pat._rep.period - 7.0) < 0.05
    assert abs(pat._rep.angle - 25) < 1.0
    assert abs(pat.measure["amplitude"] - 3.0) < 0.5
    assert pat.measure["is_global"]


def test_ripple_on_a_flat_image(tmp_path):
    flat = np.full((400, 400, 3), 250.0)
    ripple, _ = wave(flat.shape, 9.0, 30, 1.0)
    path = tmp_path / "flat.png"
    to_img(flat + ripple[..., None]).save(path)
    res = analyze(str(path))
    top = res["patterns"][0]
    assert top.category == "global_faint"
    assert abs(top._rep.period - 9.0) < 0.2


def test_lattice_of():
    assert lattice_of(0.125, 0.0, 512, 512) == 8
    assert lattice_of(0.5, 0.25, 512, 512) == 4
    assert lattice_of(-0.5, 0.0, 512, 512) == 2
    assert lattice_of(0.0625, 0.0, 512, 512) == 16
    assert lattice_of(0.137, 0.0, 512, 512) == 0


def test_block_test_ignores_uncompressed_pictures():
    y = BASE @ np.array([0.299, 0.587, 0.114])
    assert not block_boundary_test(y)["detected"]


def test_report_is_written_for_odd_inputs(tmp_path):
    variants = {
        "gray.png": to_img(BASE).convert("L"),
        "alpha.png": to_img(BASE).convert("RGBA"),
        "white.png": Image.new("RGB", (200, 200), (255, 255, 255)),
    }
    for name, im in variants.items():
        path = tmp_path / name
        im.save(path)
        out = tmp_path / (name + ".html")
        build_report(analyze(str(path)), str(out), str(out) + ".json")
        assert out.stat().st_size > 2000


def test_too_small_image_is_rejected(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (40, 40)).save(path)
    with pytest.raises(ValueError):
        analyze(str(path))
