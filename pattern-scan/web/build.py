"""Assemble the single-file browser tool:  python web/build.py  ->  index.html"""
import base64
import io
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests"))
from cases import synthetic_photo, to_img, wave   # noqa: E402


def sample_jpeg():
    """A small generated picture with three planted patterns (the page's opening example)."""
    base = synthetic_photo(11, 640, 480)
    ripple, _ = wave(base.shape, 6.3, 37, 2.0)                 # faint, whole frame
    stripes, _ = wave(base.shape, 10.4, 100, 24.0)             # visible, one region
    mask = np.zeros(base.shape[:2])
    mask[250:430, 370:600] = 1
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(4)), dtype=float) / 255
    buf = io.BytesIO()
    to_img(base + ripple[..., None] + (stripes * mask)[..., None]).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def read(name):
    with open(os.path.join(here, name), encoding="utf-8") as fh:
        return fh.read()


worker = "\n".join([read("core.js"), read("render.js"), read("worker_glue.js")])
assert "</script" not in worker.lower() and "<!--" not in worker
sample = sample_jpeg()
page = read("template.html")
page = page.replace("/*__WORKER_SRC__*/", worker).replace("__SAMPLE_B64__", base64.b64encode(sample).decode())
out = os.path.join(here, "..", "index.html")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(page)
with open(os.path.join(here, "sample.jpg"), "wb") as fh:
    fh.write(sample)
print("wrote", os.path.normpath(out), f"({len(page) // 1024} KB, sample {len(sample) // 1024} KB)")
