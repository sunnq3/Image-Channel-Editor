"""Build the demo image: a generated picture with three planted patterns.

    python examples/make_demo.py && python -m patternscan examples/demo.jpg
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests"))
from cases import synthetic_photo, to_img, wave   # noqa: E402

base = synthetic_photo(11, 960, 720)
ripple, _ = wave(base.shape, 6.3, 37, 2.0)                 # faint, whole frame
stripes, _ = wave(base.shape, 10.4, 100, 24.0)             # visible, one region
mask = np.zeros(base.shape[:2])
mask[380:640, 560:900] = 1
mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
    ImageFilter.GaussianBlur(4)), dtype=float) / 255
img = to_img(base + ripple[..., None] + (stripes * mask)[..., None])
img.save(os.path.join(here, "demo.jpg"), quality=88)       # adds JPEG blocks
print("wrote", os.path.join(here, "demo.jpg"))
