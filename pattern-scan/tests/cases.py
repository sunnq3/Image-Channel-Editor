"""Test images with known answers.

Every case is (name, PIL image or file bytes, expectation). Base pictures are
generated from scratch so they carry no camera / compression history of their
own; anything the tool reports on a "clean" case is therefore a false alarm.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def synthetic_photo(seed, w=640, h=512):
    """Natural-looking picture: 1/f colour clouds, soft-edged shapes, sensor noise."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rad = np.hypot(fx, fy)
    rad[0, 0] = 1.0
    chans = []
    for _ in range(3):
        spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) / rad ** 1.3
        spec[0, 0] = 0
        f = np.fft.ifft2(spec).real
        chans.append(f / f.std())
    base = np.stack(chans, -1)
    base = 120 + 38 * (0.6 * base + 0.4 * base.mean(-1, keepdims=True))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    # soft-edged shapes, drawn 3x larger and reduced so edges are anti-aliased
    big = Image.new("RGBA", (w * 3, h * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for _ in range(14):
        col = tuple(int(c) for c in rng.integers(20, 235, 3)) + (int(rng.integers(120, 255)),)
        x0, y0 = rng.integers(0, w * 3), rng.integers(0, h * 3)
        rw, rh = rng.integers(60, w), rng.integers(60, h)
        if rng.random() < 0.5:
            d.ellipse([x0 - rw, y0 - rh, x0 + rw, y0 + rh], fill=col)
        else:
            pts = [(x0 + rng.integers(-rw, rw), y0 + rng.integers(-rh, rh)) for _ in range(4)]
            d.polygon(pts, fill=col)
    small = big.filter(ImageFilter.GaussianBlur(2.2)).reduce(3)
    img = Image.alpha_composite(img.convert("RGBA"), small).convert("RGB")
    a = np.asarray(img, dtype=np.float64)
    a += rng.normal(0, 1.4, a.shape)                          # sensor noise
    return np.clip(a, 0, 255)


def to_img(a):
    return Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))


def wave(shape, period, angle_deg, amp, phase=0.0):
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    t = np.deg2rad(angle_deg)
    u, v = np.cos(t) / period, np.sin(t) / period
    return amp * np.sin(2 * np.pi * (u * xx + v * yy) + phase), (u, v)


def jpeg_bytes(img, quality, subsampling=2):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=subsampling)
    return buf.getvalue()


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def reopen(b):
    return Image.open(io.BytesIO(b)).convert("RGB")


def build_cases(base, tag):
    """Yield (name, file_bytes, extension, expect) for one base picture (float array).

    expect = dict(must=[(category set, (u, v) or None)], allowed=set of categories)
    Anything reported outside `must` and `allowed` counts as a false alarm.
    """
    h, w = base.shape[:2]
    faint, vis = {"global_faint"}, {"global_visible"}
    JPEG = {"jpeg_grid", "chroma_grid", "trace"}
    out = []

    def add(name, data, ext, must, allowed=()):
        out.append((f"{tag}/{name}", data, ext, {"must": must, "allowed": set(allowed)}))

    add("clean", png_bytes(to_img(base)), "png", [])

    wv, f1 = wave(base.shape, 5.0, 33, 2.0)
    add("faint ripple, period 5 px, +-2 levels", png_bytes(to_img(base + wv[..., None])), "png",
        [(faint, f1)])
    wv, f2 = wave(base.shape, 12.5, 118, 4.0)
    add("faint ripple, period 12.5 px, +-4 levels", png_bytes(to_img(base + wv[..., None])), "png",
        [(faint, f2)])
    wv, f3 = wave(base.shape, 9.0, 60, 16.0)
    add("visible stripes, period 9 px, +-16 levels", png_bytes(to_img(base + wv[..., None])),
        "png", [(vis, f3)])

    wa, fa = wave(base.shape, 6.0, 20, 2.5)
    wb, fb = wave(base.shape, 7.0, 110, 2.5)
    add("faint 2-D grid (two ripples)", png_bytes(to_img(base + (wa + wb)[..., None])), "png",
        [(faint, fa), (faint, fb)])

    # repeating texture confined to one region (a "fence" in 1/5 of the frame)
    wv, f4 = wave(base.shape, 8.6, 12, 28.0)
    mask = np.zeros((h, w))
    mask[h // 4:h // 4 * 3, w // 8:w // 8 + int(w * 0.4)] = 1
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(3)), dtype=float) / 255
    add("stripes in one region only", png_bytes(to_img(base + (wv * mask)[..., None])), "png",
        [({"regional"}, f4)])
    wv, f4b = wave(base.shape, 8.0, 0, 28.0)
    add("stripes in one region, exactly 8 px wide", png_bytes(to_img(base + (wv * mask)[..., None])),
        "png", [({"regional"}, f4b)])

    clean = to_img(base)
    for q in (50, 75, 90, 95):
        add(f"JPEG quality {q}", jpeg_bytes(clean, q), "jpg", [({"jpeg_grid"}, None)], JPEG)
    add("JPEG q75 re-saved as PNG", png_bytes(reopen(jpeg_bytes(clean, 75))), "png",
        [({"jpeg_grid"}, None)], JPEG)
    cropped = reopen(jpeg_bytes(clean, 75)).crop((3, 5, w - 5, h - 3))
    add("JPEG q75, cropped by (3,5), saved as PNG", png_bytes(cropped), "png",
        [({"jpeg_grid"}, None)], JPEG)

    for s in (1.3, 1.5):
        up = clean.resize((round(w * s), round(h * s)), Image.BICUBIC)
        add(f"enlarged {s}x (bicubic)", png_bytes(up), "png", [({"resampling"}, None)],
            {"resampling", "trace"})
    up = clean.resize((w * 2, h * 2), Image.NEAREST)
    add("enlarged 2x (pixel doubling)", png_bytes(up), "png", [({"pixel_lattice"}, None)],
        {"pixel_lattice", "resampling", "trace"})
    up = clean.resize((w * 2, h * 2), Image.BICUBIC)
    add("enlarged exactly 2x (bicubic)", png_bytes(up), "png",
        [({"pixel_lattice", "resampling"}, None)], {"pixel_lattice", "resampling", "trace"})
    rot = clean.rotate(7, Image.BICUBIC, expand=False).crop((w // 8, h // 8, w - w // 8, h - h // 8))
    add("rotated 7 degrees", png_bytes(rot), "png", [({"resampling"}, None)], {"resampling", "trace"})

    wv, f5 = wave(base.shape, 5.0, 33, 3.0)
    add("faint ripple +-3 then JPEG q85", jpeg_bytes(to_img(base + wv[..., None]), 85), "jpg",
        [(faint, f5), ({"jpeg_grid"}, None)], JPEG)
    return out
