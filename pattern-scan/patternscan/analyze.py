"""Find repeating (periodic) patterns in a single image and say what causes them.

Two "views" of every colour channel are scanned at all frequencies:

  view A  the pixel values themselves
          -> finds patterns that are ADDED to the picture (overlays, stripes,
             moire, banding, halftone, faint watermark-like ripples)
  view B  the local roughness |Laplacian| of the pixel values
          -> finds patterns in the picture's STATISTICS (JPEG block edges,
             traces left by resizing / rotating, sensor / generator lattices)

Each significant spectral peak is then measured back in the image (where it is,
how strong, whether its phase is locked across the whole frame) and classified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy.ndimage import maximum_filter, uniform_filter

LN2 = math.log(2.0)

# ----------------------------------------------------------------------------
# tunables (calibrated by tests/benchmark.py, see README)
# ----------------------------------------------------------------------------
ALPHA = 0.01          # nominal per-spectrum false-alarm level before safety factor
SAFETY = 1.6          # multiplies the theoretical threshold (real images have heavy tails)
ISOLATION = 5.0       # peak must be this many times the 85th percentile of its surroundings
MIN_BINS_FROM_DC = 3  # ignore patterns with fewer than 3 cycles across the image
MAX_PEAKS_PER_SPECTRUM = 60
LATTICE_TOL_BINS = 1.5
GLOBAL_SIGN_CONSISTENCY = 0.85
MAX_ANALYSIS_SIDE = 2048       # default; --full raises it to FULL_ANALYSIS_SIDE
FULL_ANALYSIS_SIDE = 4096


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


def _estimate_jpeg_quality(qtable) -> int | None:
    """Closest libjpeg quality setting for a luminance quantisation table."""
    try:
        q = np.sort(np.asarray(list(qtable), dtype=float))
    except Exception:
        return None
    if q.size != 64:
        return None
    best, best_err = None, 1e18
    ref = np.sort(_STD_LUMA_Q)
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        t = np.clip(np.floor((ref * scale + 50) / 100), 1, 255)
        err = float(np.abs(np.sort(t) - q).sum())
        if err < best_err:
            best, best_err = quality, err
    return best


def load_image(path, max_side=MAX_ANALYSIS_SIDE):
    """Return (channels dict, RGB array, file-info dict). Pixels are never rescaled."""
    im = Image.open(path)
    info = {
        "path": str(path),
        "format": im.format or "unknown",
        "mode": im.mode,
        "width": im.width,
        "height": im.height,
        "is_jpeg": (im.format == "JPEG"),
        "jpeg_quality": None,
        "jpeg_subsampling": None,
        "notes": [],
    }
    if info["is_jpeg"]:
        qt = getattr(im, "quantization", None)
        if qt:
            info["jpeg_quality"] = _estimate_jpeg_quality(qt[min(qt.keys())])
        try:
            from PIL import JpegImagePlugin
            ss = JpegImagePlugin.get_sampling(im)
            info["jpeg_subsampling"] = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(ss)
        except Exception:
            pass
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba).convert("RGB")
        info["notes"].append("Transparency was flattened onto white before analysis.")
    elif im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        hi = a.max() if a.max() > 0 else 1.0
        a = a * (255.0 / (65535.0 if hi > 255 else 255.0))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
        info["notes"].append("High bit-depth image was reduced to 8 bits for analysis.")
    gray = im.mode in ("L", "1")
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64)

    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        ch, cw = min(h, max_side), min(w, max_side)
        # keep the crop on the 16-pixel grid so block patterns stay aligned
        y0 = ((h - ch) // 2) // 16 * 16
        x0 = ((w - cw) // 2) // 16 * 16
        rgb = rgb[y0:y0 + ch, x0:x0 + cw]
        info["notes"].append(
            f"Image is large; analysed the central {cw}x{ch} pixels at full resolution "
            f"(starting at {x0},{y0}). Pictures and positions in this report refer to that "
            f"region. Run with --full to analyse up to {FULL_ANALYSIS_SIDE} px per side.")
    info["analysed_width"] = rgb.shape[1]
    info["analysed_height"] = rgb.shape[0]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chans = {"Y": 0.299 * r + 0.587 * g + 0.114 * b}
    if not gray and (np.abs(r - g).max() > 0 or np.abs(g - b).max() > 0):
        chans["Cb"] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        chans["Cr"] = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return chans, rgb, info


def _smooth3(a, size):
    """Three box blurs: a cheap, close approximation of a Gaussian blur."""
    for _ in range(3):
        a = uniform_filter(a, size=size, mode="nearest")
    return a


# ----------------------------------------------------------------------------
# spectra
# ----------------------------------------------------------------------------
def roughness_lap(x):
    """|Laplacian|: symmetric local roughness (view B, first flavour)."""
    lap = np.zeros_like(x)
    lap[1:-1, 1:-1] = (4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1]
                       - x[1:-1, :-2] - x[1:-1, 2:])
    return np.abs(lap)


def roughness_step(x):
    """|step to the right| + |step down|: one-sided roughness (view B, second flavour).

    After an exact 2x enlargement every pixel has the same symmetric
    neighbourhood, so the Laplacian shows nothing; a one-sided measure still
    shows the 2-pixel rhythm.
    """
    r = np.zeros_like(x)
    r[:, :-1] += np.abs(x[:, 1:] - x[:, :-1])
    r[:-1, :] += np.abs(x[1:, :] - x[:-1, :])
    r[:, -1] = r[:, -2]
    r[-1, :] = r[-2, :]
    return r


def _block_median_interp(p, cell, axis):
    """Median over 1 x cell strips along `axis`, linearly interpolated back."""
    p = np.moveaxis(p, axis, -1)
    n = p.shape[-1]
    nc = max(1, n // cell)
    used = nc * cell
    med = np.median(p[..., :used].reshape(p.shape[:-1] + (nc, cell)), axis=-1)
    if nc == 1:
        out = np.repeat(med, n, axis=-1)
    else:
        pos = (np.arange(n) - (cell - 1) / 2.0) / cell
        pos = np.clip(pos, 0, nc - 1)
        i0 = np.minimum(np.floor(pos).astype(int), nc - 2)
        wgt = pos - i0
        out = med[..., i0] * (1 - wgt) + med[..., i0 + 1] * wgt
    return np.moveaxis(out, -1, axis)


def flatness_weight(x, scale=3):
    """Weight in (0, 1]: near 1 in smooth areas, near 0 in busy detail.

    A faint pattern is only measurable where the picture itself is quiet, so the
    sensitive scans weight every pixel by the inverse of its local variance
    (the same idea as the weighting used for sensor-noise fingerprints).
    """
    d = x - (uniform_filter(x, size=3, mode="nearest") if scale <= 3 else _smooth3(x, scale))
    v = _smooth3(d * d, max(7, scale))
    v0 = max(0.5, 0.25 * float(np.median(v)))
    return v0 / (v + v0)


def whitened_spectrum(x, weight=None):
    """Return E, an fftshift-ed map where pure noise gives Exp(1)-like values.

    E = ln2 * power / local background, so a value of 20 means the bin holds
    ~29x the typical power of its surroundings.
    """
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    if weight is not None:
        win = win * weight
    xw = (x - (x * win).sum() / win.sum()) * win
    F = sfft.fftshift(sfft.fft2(xw, workers=-1))
    P = (F.real ** 2 + F.imag ** 2)
    P = np.maximum(P, 1e-20)

    fy = sfft.fftshift(sfft.fftfreq(h))[:, None]
    fx = sfft.fftshift(sfft.fftfreq(w))[None, :]
    rad = np.hypot(fx, fy)
    # 1) radial 1/f-type slope
    nb = 256
    idx = np.minimum((rad / 0.7072 * nb).astype(np.int32), nb - 1)
    logp = np.log(P)
    s = np.bincount(idx.ravel(), weights=logp.ravel(), minlength=nb)
    c = np.bincount(idx.ravel(), minlength=nb)
    prof = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    good = ~np.isnan(prof)
    prof = np.interp(np.arange(nb), np.flatnonzero(good), prof[good])
    pos = rad / 0.7072 * nb - 0.5
    Pn = P / np.exp(np.interp(pos.ravel(), np.arange(nb), prof).reshape(P.shape))
    # 2) local background that also follows horizontal / vertical ridges
    cell = int(np.clip(min(h, w) // 16, 8, 32))
    B = np.maximum(_block_median_interp(Pn, cell, 1), _block_median_interp(Pn, cell, 0))
    E = LN2 * Pn / np.maximum(B, 1e-20)
    return E, P, win.sum()


@dataclass
class Peak:
    view: str            # "A" (pixel values) or "B" (roughness)
    channel: str
    u: float             # cycles / pixel, horizontal
    v: float             # cycles / pixel, vertical
    score: float         # whitened strength E at the peak
    bx: float = 0.0      # bin offset from DC (x)
    by: float = 0.0
    lattice: int = 0     # 0 = none, else 2, 4, 8 or 16 (pixel period of the lattice)
    weighted: bool = False   # found by the flat-area-weighted (sensitive) scan
    src: str = "val"         # which map it was found in: val, lap or step
    channels: dict = field(default_factory=dict)

    @property
    def freq(self):
        return math.hypot(self.u, self.v)

    @property
    def period(self):
        return 1.0 / max(self.freq, 1e-9)

    @property
    def angle(self):
        """Direction the pattern varies along, degrees in [0, 180)."""
        return math.degrees(math.atan2(self.v, self.u)) % 180.0


def find_peaks(E, view, channel, P=None):
    """Isolated, statistically significant maxima of a whitened spectrum.

    The spectrum is periodic, so the search wraps around its edges: patterns at
    exactly 2 pixels per cycle sit on the border of the map.
    """
    h, w = E.shape
    cy, cx = h // 2, w // 2
    n_half = h * w / 2
    thr = SAFETY * math.log(n_half / ALPHA)
    pad = 30
    Ep = np.pad(E, pad, mode="wrap")
    mx = maximum_filter(E, size=7, mode="wrap")
    cand = (E == mx) & (E > thr)
    # keep one representative of each mirror pair (the spectrum is symmetric)
    cand[1:cy, :] = False
    cand[cy, 1:cx + 1] = False
    cand[0, 1:cx + 1] = False
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return [], thr
    order = np.argsort(E[ys, xs])[::-1][:MAX_PEAKS_PER_SPECTRUM * 3]
    peaks = []
    yy, xx = np.mgrid[-10:11, -10:11]
    ring = (np.hypot(yy, xx) > 3.5) & (np.hypot(yy, xx) <= 10)

    def refine(a_, b_, c_):
        a_, b_, c_ = math.log(a_), math.log(b_), math.log(c_)
        d = a_ - 2 * b_ + c_
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a_ - c_) / d, -0.5, 0.5))

    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if math.hypot(y - cy, x - cx) < MIN_BINS_FROM_DC:
            continue
        yp, xp = y + pad, x + pad
        ref = np.percentile(Ep[yp - 10:yp + 11, xp - 10:xp + 11][ring], 85)
        if E[y, x] < ISOLATION * ref:
            continue
        # A long straight edge in the picture puts a streak through the centre of
        # the spectrum, and noise along a streak makes local maxima. A real
        # pattern is a dot: it must also stand out along the line through the
        # centre, not just from its surroundings in general.
        ry, rx = y - cy, x - cx
        rr = math.hypot(ry, rx)
        along = []
        for dist in range(5, 27, 2):
            for sgn in (1, -1):
                t = (rr + sgn * dist) / rr
                if rr + sgn * dist < MIN_BINS_FROM_DC + 2:
                    continue
                sy = int(round(cy + ry * t)) + pad
                sx = int(round(cx + rx * t)) + pad
                if 1 <= sy < Ep.shape[0] - 1 and 1 <= sx < Ep.shape[1] - 1:
                    along.append(Ep[sy - 1:sy + 2, sx - 1:sx + 2].max())
        if len(along) >= 6 and E[y, x] < ISOLATION * 2.0 * float(np.median(along)):
            continue
        if P is not None:
            # locate on the raw power: the whitening can skew the apex slightly
            nb = np.take(np.take(P, range(y - 2, y + 3), axis=0, mode="wrap"),
                         range(x - 2, x + 3), axis=1, mode="wrap")
            j, i = np.unravel_index(int(np.argmax(nb)), nb.shape)
            if 0 < j < 4 and 0 < i < 4:
                dx = refine(nb[j, i - 1], nb[j, i], nb[j, i + 1]) + (i - 2)
                dy = refine(nb[j - 1, i], nb[j, i], nb[j + 1, i]) + (j - 2)
            else:
                dx, dy = float(i - 2), float(j - 2)
        else:
            dx = refine(Ep[yp, xp - 1], Ep[yp, xp], Ep[yp, xp + 1])
            dy = refine(Ep[yp - 1, xp], Ep[yp, xp], Ep[yp + 1, xp])
        bx, by = x + dx - cx, y + dy - cy
        peaks.append(Peak(view, channel, bx / w, by / h, float(E[y, x]), bx, by))
        if len(peaks) >= MAX_PEAKS_PER_SPECTRUM:
            break
    return peaks, thr


def _wrap(d):
    """Frequency difference in cycles/pixel, wrapped to [-0.5, 0.5)."""
    return (d + 0.5) % 1.0 - 0.5


def lattice_of(u, v, w, h):
    """Smallest pixel period (2, 4, 8, 16) whose lattice this frequency sits on."""
    for base in (2, 4, 8, 16):
        lu, lv = round(u * base) / base, round(v * base) / base
        # An enlarged image has a spectral null at 2 px/cycle; the peak beside a
        # null is located less precisely, so allow more slack there.
        tu = (3.5 if abs(lu) == 0.5 else LATTICE_TOL_BINS) / w
        tv = (3.5 if abs(lv) == 0.5 else LATTICE_TOL_BINS) / h
        if abs(u - lu) < tu and abs(v - lv) < tv and (lu, lv) != (0.0, 0.0):
            return base
    return 0


# ----------------------------------------------------------------------------
# measuring a pattern back in the image
# ----------------------------------------------------------------------------
def _box_for(period, h, w):
    return int(np.clip(round(1.25 * period), 8, max(8, min(h, w) // 8)))


def demodulate_all(x, u, v, weight=None):
    """Like demodulate(), but also returns the decoy maps (for noise estimates)."""
    h, w = x.shape
    period = 1.0 / max(math.hypot(u, v), 1e-9)
    box = _box_for(period, h, w)
    hp = x - uniform_filter(x, size=box, mode="nearest")   # remove local mean (stops leakage)
    cell = max(2, box // 4)
    hh, ww = (h // cell) * cell, (w // cell) * cell
    yy = np.arange(hh, dtype=np.float64)[:, None]
    xx = np.arange(ww, dtype=np.float64)[None, :]
    if weight is None:
        wc = np.ones((hh // cell, ww // cell))
        hpw = hp[:hh, :ww].astype(np.float32)
    else:
        wc = weight[:hh, :ww].reshape(hh // cell, cell, ww // cell, cell).mean(axis=(1, 3))
        wc = _smooth3(wc, 4)
        hpw = (hp[:hh, :ww] * weight[:hh, :ww]).astype(np.float32)
    d = 2.0 / box                                # decoy offset, outside the passband
    out = []
    for (fu, fv) in [(u, v), (u + d, v), (u - d, v), (u, v + d), (u, v - d)]:
        # separable carrier: average along x first, then along y (cheap and exact)
        ex = np.exp(-2j * np.pi * fu * xx).astype(np.complex64)
        ey = np.exp(-2j * np.pi * fv * yy).astype(np.complex64)
        zx = (hpw * ex).reshape(hh, ww // cell, cell).mean(axis=2)
        z = (zx * ey).reshape(hh // cell, cell, ww // cell).mean(axis=1).astype(np.complex128)
        out.append((_smooth3(z.real, 4) + 1j * _smooth3(z.imag, 4)) / np.maximum(wc, 1e-9))
    return out[0], out[1:], wc, cell, box


def measure(x, u, v, weight=None):
    """Measure one wave back in the image.

    Besides amplitude and a where-is-it map, this decides whether the pattern is
    GLOBAL (same strength and alignment in every part of the frame that is quiet
    enough to tell) or REGIONAL (present in some parts and provably absent in
    others). The frame is split into 4x4 zones; a zone only votes when its noise
    is low enough to distinguish "present" from "absent".
    """
    A, decoys, wc, cell, box = demodulate_all(x, u, v, weight)
    # The peak position is known to a fraction of a bin; a residual error shows
    # up as a slow phase drift across the frame. Measure that drift from
    # neighbouring cells and take it out (at most one bin's worth).
    # (Zone averages are used because noise in separate zones is independent;
    # neighbouring cells share noise and would bias the estimate.)
    nzd = 8
    if A.shape[0] >= 2 * nzd and A.shape[1] >= 2 * nzd:
        gh0, gw0 = A.shape
        sy, sx = gh0 // nzd, gw0 // nzd
        Z = (A * wc)[:sy * nzd, :sx * nzd].reshape(nzd, sy, nzd, sx).sum(axis=(1, 3))
        gx = np.angle((Z[:, 1:] * np.conj(Z[:, :-1])).sum()) / sx     # radians per cell
        gy = np.angle((Z[1:, :] * np.conj(Z[:-1, :])).sum()) / sy
        lim_x = 1.5 * 2 * np.pi * cell / x.shape[1]                   # 1.5 bins
        lim_y = 1.5 * 2 * np.pi * cell / x.shape[0]
        if abs(gx) <= lim_x and abs(gy) <= lim_y:
            jj = np.arange(gh0)[:, None]
            ii = np.arange(gw0)[None, :]
            A = A * np.exp(-1j * (gx * ii + gy * jj))
    mag = np.abs(A)
    noise = np.median(np.abs(np.stack(decoys)), axis=0)
    wn = wc / wc.sum()
    total = (A * wn).sum()                       # reliability-weighted mean amplitude
    phi = np.angle(total)
    rot = np.exp(-1j * phi)
    proj = (A * rot).real
    sign_consistency = float(((proj > 0) * wn).sum())
    detect = mag > 3.0 * np.maximum(noise, 1e-9)
    coverage = float(detect.mean())
    amp_global = 2.0 * float(np.abs(total))

    nz = 4
    gh, gw = A.shape
    ye = np.linspace(0, gh, nz + 1).astype(int)
    xe = np.linspace(0, gw, nz + 1).astype(int)
    P, SE = [], []
    for j in range(nz):
        for i in range(nz):
            sl = (slice(ye[j], ye[j + 1]), slice(xe[i], xe[i + 1]))
            wz = wc[sl]
            if wz.size == 0 or wz.sum() <= 0:
                continue
            wz = wz / wz.sum()
            P.append(float((proj[sl] * wz).sum()))
            dz = [abs((d_[sl] * wz).sum()) for d_ in decoys]
            SE.append(float(np.sqrt(np.mean(np.square(dz)) / 2.0)))
    P, SE = np.array(P), np.array(SE)
    ref = float(np.mean(np.sort(P)[-3:])) if P.size else 0.0   # strongest zones
    present = absent = 0
    if ref > 0:
        for pz, se in zip(P, SE):
            if pz > 0.4 * ref and pz > 2.0 * se:
                present += 1
            elif pz + 2.0 * se < 0.4 * ref:
                absent += 1
    voted = present + absent
    uniformity = present / voted if voted else None
    if uniformity is None:
        is_global = sign_consistency >= GLOBAL_SIGN_CONSISTENCY
    else:
        is_global = (uniformity >= 0.8) and (voted >= 3 or sign_consistency >= 0.7)
    amp_local = (2.0 * float(np.percentile(mag[detect], 90)) if detect.sum() >= 4
                 else 2.0 * max(ref, 0.0))
    env = np.clip(mag - 1.5 * noise, 0, None)
    return {
        "sign_consistency": sign_consistency,
        "uniformity": uniformity,
        "zones_present": present,
        "zones_absent": absent,
        "is_global": bool(is_global),
        "coverage": coverage,
        "amp_global": amp_global,
        "amp_local": amp_local,
        "envelope": env,
        "cell": cell,
        "box": box,
    }


def reconstruct(x, peaks_uv, y0=0, x0=0, hgt=None, wid=None):
    """Isolate the pattern (sum of the given waves) inside a rectangle of x.

    Each wave is band-passed around its own frequency, so what comes back is the
    pattern as it actually sits in the image (its local strength and alignment),
    in the same units as x.
    """
    h, w = x.shape
    hgt = h - y0 if hgt is None else hgt
    wid = w - x0 if wid is None else wid
    out = np.zeros((hgt, wid))
    for (u, v) in peaks_uv:
        period = 1.0 / max(math.hypot(u, v), 1e-9)
        box = _box_for(period, h, w)
        pad = 2 * box
        ya, yb = max(0, y0 - pad), min(h, y0 + hgt + pad)
        xa, xb = max(0, x0 - pad), min(w, x0 + wid + pad)
        sub = x[ya:yb, xa:xb]
        hp = sub - _smooth3(sub, box)
        yy = np.arange(ya, yb, dtype=np.float64)[:, None]
        xx = np.arange(xa, xb, dtype=np.float64)[None, :]
        car = np.exp(2j * np.pi * u * xx) * np.exp(2j * np.pi * v * yy)
        z = hp * np.conj(car)
        A = _smooth3(z.real, box) + 1j * _smooth3(z.imag, box)
        wave = 2.0 * (A * car).real
        out += wave[y0 - ya:y0 - ya + hgt, x0 - xa:x0 - xa + wid]
    return out


def reconstruct_crop(x, peaks_uv, y0, x0, size):
    return reconstruct(x, peaks_uv, y0, x0, size, size)


# ----------------------------------------------------------------------------
# JPEG block-boundary test (pixel domain, independent of the spectra)
# ----------------------------------------------------------------------------
def block_boundary_test(y, weight=None):
    """Look for an 8-pixel grid of small steps, the signature of JPEG blocks.

    Returns z-scores for the horizontal and vertical direction and the grid
    offset. A fresh JPEG has offset (0, 0); another offset means the image was
    cropped (or pasted) after it was compressed.
    """
    res = {}
    if weight is None:
        weight = np.ones_like(y)
    for name, a, wt in (("x", y, weight), ("y", y.T, weight.T)):
        d = np.abs(np.diff(a, axis=1))                       # |I(x+1) - I(x)|
        s = d[:, 1:-1] - 0.5 * (d[:, :-2] + d[:, 2:])        # step at x+1|x+2 vs neighbours
        ww = wt[:, 1:-2]
        col = (s * ww).sum(axis=0) / np.maximum(ww.sum(axis=0), 1e-9)
        n = (col.size // 8) * 8
        if n < 64:
            res[name] = (0.0, 0)
            continue
        ph = col[:n].reshape(-1, 8)
        m = ph.mean(axis=0)
        k = int(np.argmax(m))
        others = np.delete(m, k)
        # per-phase scatter across the image, for a proper standard error
        se = ph.std(axis=0, ddof=1).mean() / math.sqrt(ph.shape[0])
        z = float((m[k] - others.mean()) / max(se, 1e-9))
        # one boundary per 8 pixels must stand out; a 2- or 4-pixel rhythm has several
        z = min(z, float((m[k] - others.max()) / max(se, 1e-9)) * 1.5)
        res[name] = (z, (k + 2) % 8)     # first pixel of a block, modulo 8
    zx, ox = res["x"]
    zy, oy = res["y"]
    return {"z_x": zx, "z_y": zy, "offset_x": ox, "offset_y": oy,
            "detected": bool(zx > 6.0 and zy > 6.0)}


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------
CATEGORIES = {
    "jpeg_grid":      ("JPEG compression blocks", (255, 150, 30)),
    "grid8":          ("8-pixel lattice (not confirmed as JPEG)", (235, 190, 60)),
    "chroma_grid":    ("16-pixel colour blocks", (190, 120, 60)),
    "pixel_lattice":  ("2/4-pixel lattice", (170, 110, 255)),
    "resampling":     ("Resize / rotate trace", (60, 150, 255)),
    "global_faint":   ("Faint global overlay (watermark-like)", (255, 50, 120)),
    "global_visible": ("Strong global pattern", (255, 225, 40)),
    "regional":       ("Repeating texture in part of the image", (60, 210, 110)),
    "trace":          ("Sub-level processing trace", (150, 160, 170)),
}

EXPLAIN = {
    "jpeg_grid": "JPEG stores the picture in 8x8-pixel blocks. Tiny steps at the block "
                 "edges repeat every 8 pixels across the whole image.",
    "grid8": "A pattern that repeats every 8 pixels, but without the small steps at block "
             "edges that JPEG leaves. Possible causes: an AI generator's upsampling stage, "
             "heavy processing after a JPEG save, or a pattern placed on that lattice.",
    "chroma_grid": "Repeats every 16 pixels, mainly in colour. JPEG usually stores colour at "
                   "half resolution, which makes colour blocks 16 pixels wide.",
    "pixel_lattice": "Repeats every 2 or 4 pixels exactly. Typical causes: the camera "
                     "sensor's colour mosaic, enlarging by a whole factor, the upsampling "
                     "layers of an AI image generator, or a pattern placed on that lattice.",
    "resampling": "Resizing or rotating recomputes every pixel from its neighbours, which "
                  "leaves a regular rhythm in how rough the image is. It is not a drawn "
                  "pattern; it is a statistical trace.",
    "global_faint": "A regular ripple added to the pixel values over the whole frame with "
                    "the same alignment everywhere, too faint to see, and not explained by "
                    "JPEG blocks, pixel lattices or resizing. Invisible watermarks, "
                    "electrical interference and scanner noise look like this.",
    "global_visible": "A regular pattern covering the whole frame with the same alignment "
                      "everywhere, strong enough to see: stripes, scan lines, a photographed "
                      "screen, halftone print dots, or moire.",
    "regional": "A repeating pattern that is present in part of the image or changes "
                "alignment from place to place. That is how real subjects behave: fabric, "
                "fences, bricks, blinds, text lines.",
    "trace": "A regular ripple smaller than a third of one brightness step. Colour "
             "conversion, resizing and compression leave such residues; it is far below "
             "anything visible and is listed for completeness.",
}

TRACE_AMPLITUDE = 0.3          # grey levels; global ripples below this are "traces"
MIN_REGIONAL_AMPLITUDE = 1.5   # grey levels; local rhythms below this are not reported


@dataclass
class Pattern:
    category: str
    peaks: list
    strength: float                 # best whitened score in the group
    channel: str                    # channel with the strongest response
    view: str
    measure: dict | None = None
    notes: list = field(default_factory=list)
    weighted: bool = False


def _merge_channels(peaks, w, h):
    """Merge the same frequency seen in several channels / scans into one peak."""
    merged = []
    for p in sorted(peaks, key=lambda q: -q.score):
        for m in merged:
            if (m.view == p.view and abs(_wrap(m.u - p.u)) * w <= 1.5
                    and abs(_wrap(m.v - p.v)) * h <= 1.5):
                m.channels[p.channel] = max(m.channels.get(p.channel, 0), p.score)
                break
        else:
            p.channels = {p.channel: p.score}
            merged.append(p)
    return merged


def _near(du, dv, w, h, tol=2.0):
    return abs(_wrap(du)) * w <= tol and abs(_wrap(dv)) * h <= tol


def _on_lattice(du, dv, w, h, step, tol=2.0):
    """Is the offset (du, dv) a point of the 1/step frequency lattice?"""
    return (abs(_wrap(du * step) / step) * w <= tol and abs(_wrap(dv * step) / step) * h <= tol)


_HARMONICS = [s_ * n_ for n_ in range(1, 21) for s_ in (1, -1)]


def _derived_from(p, bases, w, h, lattice_step):
    """Index of the base pattern that explains peak p, or None.

    A peak is explained by a base wave f when it is a harmonic (n*f), a mix of
    two base waves (m*f1 + n*f2), or -- when the image carries a pixel-block
    lattice -- an echo of the wave shifted by a lattice frequency (n*f + k/8),
    which is what block-wise compression does to an overlaid pattern.
    """
    for i, b in enumerate(bases):
        for n in _HARMONICS:
            du, dv = p.u - n * b.u, p.v - n * b.v
            if _near(du, dv, w, h, min(2.0 + 0.75 * (abs(n) - 1), 5.0)):  # base error grows n-fold
                return i
            if lattice_step and abs(n) <= 3 and _on_lattice(du, dv, w, h, lattice_step):
                return i
    nb = min(len(bases), 4)
    for i in range(nb):
        for j in range(i + 1, nb):
            for m in (1, -1, 2, -2):
                for n in (1, -1, 2, -2):
                    du = p.u - m * bases[i].u - n * bases[j].u
                    dv = p.v - m * bases[i].v - n * bases[j].v
                    if _near(du, dv, w, h) or (lattice_step and abs(m) + abs(n) <= 2
                                               and _on_lattice(du, dv, w, h, lattice_step)):
                        return i
    return None


def analyze(path, full=False):
    chans, rgb, info = load_image(path, FULL_ANALYSIS_SIDE if full else MAX_ANALYSIS_SIDE)
    h, w = chans["Y"].shape
    if min(h, w) < 64:
        raise ValueError("Image is too small to analyse (needs at least 64x64 pixels).")

    sources, weights, spectra, thresholds = {}, {}, {}, {}
    all_peaks = []
    for name, x in chans.items():
        wt_fine = flatness_weight(x, 3)       # quiet at the pixel scale (for view B)
        wt_coarse = flatness_weight(x, 15)    # quiet over ~15 px (for faint ripples)
        weights[("A", name)] = wt_coarse
        weights[("B", name)] = wt_fine
        sources[("val", name)] = x
        sources[("lap", name)] = roughness_lap(x)
        sources[("step", name)] = roughness_step(x)
        for view, kind, wt in (("A", "val", None), ("A", "val", wt_coarse),
                               ("B", "lap", wt_fine), ("B", "step", wt_fine)):
            E, P, _ = whitened_spectrum(sources[(kind, name)], wt)
            pk, thr = find_peaks(E, view, name, P)
            del P
            for p in pk:
                p.weighted = wt is not None
                p.src = kind
            thresholds[(view, name)] = thr
            all_peaks += pk
            if name == "Y" and kind in ("val", "lap") and wt is not None:
                spectra[view] = E
            del E

    peaks = _merge_channels(all_peaks, w, h)
    for p in peaks:
        p.lattice = lattice_of(p.u, p.v, w, h)

    def best_channel(p):
        return max(p.channels, key=p.channels.get)

    def measure_peak(p):
        c = best_channel(p)
        return measure(sources[(p.src, c)], p.u, p.v,
                       weights[(p.view, c)] if p.weighted else None)

    # A real subject can repeat at exactly 2/4/8/16 px by coincidence. Processing
    # lattices are frame-wide; a lattice-frequency pattern that is provably absent
    # from parts of the frame is treated as an ordinary pattern instead.
    cache = {}
    coincidental = []
    for p in sorted((q for q in peaks if q.lattice and q.view == "A"),
                    key=lambda q: -q.score)[:5]:
        if True:
            ms = cache[id(p)] = measure_peak(p)
            # (2-px stripes are excluded: no lens resolves a real subject that finely)
            if (p.lattice >= 4 and not ms["is_global"] and ms["zones_absent"] >= 3
                    and ms["amp_local"] >= 2.0):
                p.lattice = 0
                coincidental.append(p)
    for p in peaks:                       # their harmonics follow them
        if p.lattice and any(_near(p.u - n * b.u, p.v - n * b.v, w, h)
                             for b in coincidental for n in (2, -2, 3, -3, 4, -4)):
            p.lattice = 0

    jpeg = block_boundary_test(chans["Y"], weights[("B", "Y")])
    jpeg_evidence = info["is_jpeg"] or jpeg["detected"]
    has8 = any(p.lattice in (8, 16) for p in peaks)
    lattice_step = 8 if (jpeg_evidence or has8) else 0

    groups, patterns = {}, []

    def add_lattice(cat, p):
        if cat not in groups:
            groups[cat] = Pattern(cat, [], 0.0, "Y", p.view)
            patterns.append(groups[cat])
        groups[cat].peaks.append(p)

    # An enlarged image has spectral nulls along the 2-px/cycle lines; the ridges
    # beside a null can pass as peaks. When a 2-px lattice is present, peaks hugging
    # those lines belong to it.
    has2 = any(p.lattice == 2 for p in peaks)
    if has2:
        for p in peaks:
            if not p.lattice:
                du = (0.5 - abs(_wrap(p.u))) * w
                dv = (0.5 - abs(_wrap(p.v))) * h
                near_u_edge = du <= 6 and (abs(_wrap(p.v)) * h <= 6 or dv <= 6)
                near_v_edge = dv <= 6 and (abs(_wrap(p.u)) * w <= 6 or du <= 6)
                if near_u_edge or near_v_edge:
                    p.lattice = 2

    free = []
    for p in peaks:
        if p.lattice in (2, 4):
            # 1/4 and 1/2 are also harmonics of an 8-pixel grid when one is present
            add_lattice("jpeg_grid" if jpeg_evidence else ("grid8" if has8 else "pixel_lattice"), p)
        elif p.lattice == 8:
            add_lattice("jpeg_grid" if jpeg_evidence else "grid8", p)
        elif p.lattice == 16:
            if jpeg_evidence and "Y" not in p.channels:
                add_lattice("chroma_grid", p)
            else:
                add_lattice("jpeg_grid" if jpeg_evidence else "grid8", p)
        else:
            free.append(p)

    # Additive (view A) patterns first, strongest first; everything they explain
    # (harmonics, mixes, block-lattice echoes, roughness footprints) joins them.
    free_a = sorted((p for p in free if p.view == "A"), key=lambda q: -q.score)
    free_b = sorted((p for p in free if p.view == "B"), key=lambda q: -q.score)
    bases, members = [], []
    for p in free_a:
        i = _derived_from(p, bases, w, h, lattice_step)
        if i is None:
            bases.append(p)
            members.append([p])
        else:
            members[i].append(p)
    b_bases, b_members = [], []
    for p in free_b:
        i = _derived_from(p, bases, w, h, lattice_step)
        if i is not None:
            members[i].append(p)
            continue
        j = _derived_from(p, b_bases, w, h, lattice_step)
        if j is None:
            b_bases.append(p)
            b_members.append([p])
        else:
            b_members[j].append(p)

    for base, mem in zip(bases, members):
        pat = Pattern("?", mem, base.score, best_channel(base), "A")
        pat._rep = base
        patterns.append(pat)
    for base, mem in zip(b_bases, b_members):
        pat = Pattern("resampling", mem, base.score, best_channel(base), "B")
        pat._rep = base
        patterns.append(pat)
    for pat in patterns:
        if pat.category in groups:
            # prefer the true grid frequency (1/8, 1/16) as the representative
            pat._rep = max(pat.peaks, key=lambda q: (q.view == "B", q.score))
            pat.strength = max(q.score for q in pat.peaks)
            pat.channel = best_channel(pat._rep)
            pat.view = pat._rep.view

    patterns.sort(key=lambda q: -q.strength)

    final = []
    dropped = 0
    for n_measured, pat in enumerate(patterns):
        if len(final) >= 12 or n_measured >= 24:
            break
        m = pat._rep
        ms = cache.get(id(m)) or measure_peak(m)
        pat.measure = ms
        pat.weighted = bool(m.weighted)
        on_axis = abs(_wrap(m.u)) * w <= 1.5 or abs(_wrap(m.v)) * h <= 1.5
        if pat.category == "?":
            has_b_twin = any(q.view == "B" and _near(q.u - m.u, q.v - m.v, w, h)
                             for q in pat.peaks)
            if ms["is_global"]:
                ms["amplitude"] = ms["amp_global"]
                if has_b_twin and ms["amplitude"] < 1.0:
                    # A sub-level ripple cannot shake the roughness map by itself;
                    # resizing can do both (interpolation rhythm + rounding bias).
                    pat.category = "resampling"
                else:
                    pat.category = "global_faint" if ms["amplitude"] < 8.0 else "global_visible"
            else:
                ms["amplitude"] = ms["amp_local"]
                if ms["amplitude"] < MIN_REGIONAL_AMPLITUDE:
                    # a sub-visible rhythm confined to one spot: the staircase of a
                    # straight edge after compression or resizing, not a texture
                    dropped += 1
                    continue
                pat.category = "regional"
            if pat.category == "global_faint" and ms["amplitude"] < TRACE_AMPLITUDE:
                pat.category = "trace"
            if pat.category == "global_faint" and on_axis:
                pat.notes.append(
                    "It runs exactly horizontally or vertically, which favours a processing "
                    "cause (a sensor or compression lattice that was later resized, scan "
                    "lines, banding) over a deliberately placed pattern.")
        else:
            ms["amplitude"] = ms["amp_global"] if m.view == "A" else None
            if (pat.category in ("pixel_lattice", "grid8") and ms["amplitude"] is not None
                    and all(q.view == "A" for q in pat.peaks)
                    and ms["amplitude"] < TRACE_AMPLITUDE):
                pat.category = "trace"
        if (pat.category in ("pixel_lattice", "grid8") and not ms["is_global"]
                and ms["zones_absent"] >= 3):
            dropped += 1          # a lattice-frequency rhythm confined to one area
            continue
        if pat.category == "resampling" and pat.view == "B" and not ms["is_global"]:
            # A rhythm in roughness that is missing from parts of the frame is the
            # staircase of a long straight edge or a local texture, not processing.
            dropped += 1
            continue
        if pat.category == "resampling":
            f = m.freq
            if on_axis and 0.02 < f < 0.5:
                pat.notes.append(
                    f"The rhythm repeats every {1 / f:.2f} px. Enlarging by about "
                    f"{1 / (1 - f):.2f}x or {1 / f:.2f}x would do this; so would an 8- or "
                    f"16-px compression grid shrunk to {100 / (8 * f):.0f}% or "
                    f"{100 / (16 * f):.0f}% of its size.")
            elif not on_axis:
                pat.notes.append("It is not aligned with the pixel rows or columns, which "
                                 "points to rotation or skew rather than a plain resize.")
        final.append(pat)

    return {
        "info": info,
        "rgb": rgb,
        "channels": chans,
        "sources": sources,
        "weights": weights,
        "spectra": spectra,
        "patterns": final,
        "jpeg": jpeg,
        "thresholds": thresholds,
        "n_peaks": len(peaks),
        "n_patterns_total": len(patterns) - dropped,
        "n_edge_streaks_dropped": dropped,
    }
