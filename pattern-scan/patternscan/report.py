"""Turn an analysis into pictures and a single self-contained HTML report."""
from __future__ import annotations

import base64
import html
import io
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from .analyze import CATEGORIES, EXPLAIN, LN2, reconstruct, reconstruct_crop

MAX_DISPLAY = 1400
MAX_CARDS = 8
CROP = 128
ZOOM = 3
LATTICE_CATS = ("jpeg_grid", "grid8", "chroma_grid", "pixel_lattice")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _b64(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _u8(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def _period_text(pat):
    m = pat._rep
    if pat.category in ("jpeg_grid", "grid8"):
        return "8 px (grid)"
    if pat.category == "chroma_grid":
        return "16 px (grid)"
    if pat.category == "pixel_lattice":
        return f"{m.lattice or 2} px (lattice)"
    return f"{m.period:.2f} px"


def _direction_text(pat):
    if pat.category in LATTICE_CATS:
        return "rows and columns"
    a = pat._rep.angle
    if a < 3 or a > 177:
        return "vertical stripes (changes left to right)"
    if abs(a - 90) < 3:
        return "horizontal stripes (changes top to bottom)"
    tilt = (a + 90) % 180
    return f"diagonal stripes, running at {tilt:.0f}° from horizontal"


def _where_text(pat):
    ms = pat.measure
    p, a = ms["zones_present"], ms["zones_absent"]
    if pat.category in ("jpeg_grid", "chroma_grid") and not ms["is_global"]:
        return (f"Whole frame by construction; block edges show most in smooth areas "
                f"(clear in {p} of 16 zones).")
    if ms["is_global"]:
        if a == 0 and p > 0:
            return (f"Whole frame. Confirmed in {p} of 16 zones; the others are too busy "
                    f"to tell either way.") if p < 16 else "Whole frame (all 16 zones)."
        return f"Whole frame (present in {p} zones, absent in {a}, rest too busy to tell)."
    if p + a == 0:
        return "Could not be localised (too faint against the picture's detail)."
    return f"Part of the frame: present in {p} of 16 zones, clearly absent in {a}."


def _strength_text(pat):
    ms = pat.measure
    amp = ms.get("amplitude")
    if amp is None:
        return "Statistical rhythm (not an added brightness pattern)"
    if amp < 0.3:
        vis = "far below visibility"
    elif amp < 3:
        vis = "too faint to see"
    elif amp < 8:
        vis = "barely visible in smooth areas"
    else:
        vis = "visible"
    return f"±{amp:.2f} of 255 brightness levels ({vis})"


def _confidence(pat, thr):
    times = pat.strength / LN2
    ratio = pat.strength / thr
    word = "very high" if ratio > 5 else "high" if ratio > 2 else "moderate"
    return f"{word} ({times:,.0f}× the surrounding background; detection limit {thr / LN2:.0f}×)"


def _channels_text(pat):
    names = {"Y": "brightness", "Cb": "blue-yellow colour", "Cr": "red-green colour"}
    seen = []
    for p in pat.peaks:
        for c in p.channels:
            if c not in seen:
                seen.append(c)
    return ", ".join(names[c] for c in ("Y", "Cb", "Cr") if c in seen)


# ----------------------------------------------------------------------------
# pictures
# ----------------------------------------------------------------------------
def _display_base(rgb):
    h, w = rgb.shape[:2]
    s = min(1.0, MAX_DISPLAY / max(h, w))
    img = Image.fromarray(_u8(rgb))
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img, s


def _envelope_alpha(pat, size):
    env = pat.measure["envelope"]
    top = float(np.percentile(env, 99)) if env.size else 0.0
    if top <= 0:
        return np.zeros((size[1], size[0]))
    a = np.clip(env / top, 0, 1) ** 0.8
    im = Image.fromarray(_u8(a * 255)).resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def _stripes_alpha(res, pat, size, scale):
    """The pattern itself (its crests) as a 0..1 map at display size.

    Falls back to the where-is-it envelope when the stripes would be too fine to
    survive the display reduction.
    """
    m = pat._rep
    if m.period * scale < 3.0:
        return _envelope_alpha(pat, size)
    src = res["sources"][(m.src, pat.channel)]
    same = sorted((p for p in pat.peaks if p.view == m.view and p.period * scale >= 3.0),
                  key=lambda q: -q.score)[:4]
    wave = reconstruct(src, [(p.u, p.v) for p in same])
    top = float(np.percentile(np.abs(wave), 97)) or 1.0
    a = np.clip(wave / top, 0, 1)
    im = Image.fromarray(_u8(a * 255))
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _tint(base_img, layers):
    """layers: list of (alpha map 0..1, rgb colour, max opacity)."""
    a = np.asarray(base_img.convert("RGB"), dtype=np.float64)
    gray = a.mean(axis=2, keepdims=True)
    out = 0.45 * a + 0.35 * gray + 12          # muted so the highlight colours read
    for alpha, col, op in layers:
        al = (alpha * op)[..., None]
        out = out * (1 - al) + np.array(col, dtype=np.float64) * al
    return Image.fromarray(_u8(out))


def _crop_origin(pat, shape):
    h, w = shape
    env = pat.measure["envelope"]
    cell = pat.measure["cell"]
    j, i = np.unravel_index(int(np.argmax(env)), env.shape)
    cy, cx = int((j + 0.5) * cell), int((i + 0.5) * cell)
    size = min(CROP, h, w)
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return y0, x0, size


def _zoom_pair(res, pat):
    """Left: the original crop. Right: the isolated pattern, amplified."""
    rgb = res["rgb"]
    m = pat._rep
    src = res["sources"][(m.src, pat.channel)]
    y0, x0, size = _crop_origin(pat, src.shape)
    same = sorted((p for p in pat.peaks if p.view == m.view), key=lambda q: -q.score)[:8]
    wave = reconstruct_crop(src, [(p.u, p.v) for p in same], y0, x0, size)
    top = float(np.percentile(np.abs(wave), 97)) or 1.0
    gain = 100.0 / top
    col = np.array(CATEGORIES[pat.category][1], dtype=np.float64)
    t = np.clip(wave * gain / 100.0, -1, 1)[..., None]
    pat_img = 128 + t * 110 * (0.35 + 0.65 * col / 255.0)
    left = Image.fromarray(_u8(rgb[y0:y0 + size, x0:x0 + size]))
    right = Image.fromarray(_u8(pat_img))
    left = left.resize((size * ZOOM, size * ZOOM), Image.NEAREST)
    right = right.resize((size * ZOOM, size * ZOOM), Image.NEAREST)
    if pat.category in ("jpeg_grid", "chroma_grid"):
        step = 16 if pat.category == "chroma_grid" else 8
        ox = res["jpeg"]["offset_x"] if res["jpeg"]["detected"] else 0
        oy = res["jpeg"]["offset_y"] if res["jpeg"]["detected"] else 0
        d = ImageDraw.Draw(left)
        for gx in range((ox - x0) % step, size, step):
            d.line([(gx * ZOOM, 0), (gx * ZOOM, size * ZOOM)], fill=(255, 150, 30), width=1)
        for gy in range((oy - y0) % step, size, step):
            d.line([(0, gy * ZOOM), (size * ZOOM, gy * ZOOM)], fill=(255, 150, 30), width=1)
    gap = 8
    pair = Image.new("RGB", (size * ZOOM * 2 + gap, size * ZOOM), (30, 30, 34))
    pair.paste(left, (0, 0))
    pair.paste(right, (size * ZOOM + gap, 0))
    return pair, gain, (x0, y0, size)


def _spectrum_image(E, patterns, view, max_side=720):
    """Log-scaled whitened spectrum with the detected peaks circled."""
    h, w = E.shape
    f = max(1, math.ceil(max(h, w) / max_side))
    hh, ww = (h // f) * f, (w // f) * f
    y0, x0 = (h - hh) // 2, (w - ww) // 2
    pooled = E[y0:y0 + hh, x0:x0 + ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3))
    g = np.log1p(pooled)
    g = np.clip(g / math.log1p(60.0), 0, 1) ** 0.9
    img = Image.fromarray(_u8(g * 255)).convert("RGB")
    d = ImageDraw.Draw(img)
    cy, cx = h // 2, w // 2
    for pat in patterns:
        col = CATEGORIES[pat.category][1]
        for p in pat.peaks:
            if p.view != view:
                continue
            for sgn in (1, -1):
                px = (cx + sgn * p.bx - x0) / f
                py = (cy + sgn * p.by - y0) / f
                r = 6
                d.ellipse([px - r, py - r, px + r, py + r], outline=col, width=2)
    return img


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
def _summary_lines(res):
    pats = res["patterns"]
    info, jp = res["info"], res["jpeg"]
    lines = []
    main = [p for p in pats if p.category != "trace"]
    traces = [p for p in pats if p.category == "trace"]
    if not pats:
        lines.append("No repeating pattern stands out from the picture's own detail at any "
                     "frequency, in brightness or colour.")
    else:
        n = len(main)
        lines.append(f"{n} repeating pattern{'s' if n != 1 else ''} found"
                     + (f", plus {len(traces)} sub-level trace{'s' if len(traces) != 1 else ''}."
                        if traces else "."))
    for p in main:
        label = CATEGORIES[p.category][0]
        bit = f"{label}: repeats every {_period_text(p)}"
        amp = p.measure.get("amplitude")
        if amp is not None and p.category not in LATTICE_CATS:
            bit += f", ±{amp:.1f} levels"
        if p.category in ("jpeg_grid", "chroma_grid"):
            bit += "."
        else:
            bit += ", whole frame." if p.measure["is_global"] else ", part of the frame."
        lines.append(bit)
    if info["is_jpeg"]:
        q = f", quality about {info['jpeg_quality']}" if info["jpeg_quality"] else ""
        ss = f", colour stored at {info['jpeg_subsampling']}" if info["jpeg_subsampling"] else ""
        lines.append(f"The file is a JPEG{q}{ss}.")
    elif jp["detected"]:
        off = (jp["offset_x"], jp["offset_y"])
        extra = ("" if off == (0, 0) else
                 f" The block grid starts at pixel {off}, not (0, 0), so the picture was "
                 f"cropped or shifted after that compression.")
        lines.append("The file is not a JPEG, but its pixels carry JPEG block edges: it was "
                     "JPEG-compressed at some point before being saved in this format." + extra)
    return lines


def _history_text(res):
    info, jp = res["info"], res["jpeg"]
    rows = [f"Block-edge test: horizontal z = {jp['z_x']:.1f}, vertical z = {jp['z_y']:.1f} "
            f"(above 6 in both directions counts as detected)."]
    if jp["detected"]:
        rows.append(f"8-pixel block grid found, starting at pixel "
                    f"({jp['offset_x']}, {jp['offset_y']}).")
        if info["is_jpeg"] and (jp["offset_x"], jp["offset_y"]) != (0, 0):
            rows.append("This file's own JPEG grid starts at (0, 0), so the grid found here "
                        "is left over from an EARLIER compression, before a crop or shift.")
    else:
        rows.append("No block edges found in the pixels" + (
            " (normal for a high-quality JPEG)." if info["is_jpeg"] else "."))
    return rows


def build_report(res, out_html, out_json=None):
    info = res["info"]
    pats = res["patterns"]
    rgb = res["rgb"]
    base, scale = _display_base(rgb)
    thr = max(res["thresholds"].values())

    shown = pats[:MAX_CARDS]
    # main highlight picture: every pattern in its category colour
    layers = []
    for pat in sorted(shown, key=lambda p: p.category not in LATTICE_CATS):
        if pat.category == "trace" and any(p.category != "trace" for p in shown):
            continue
        op = 0.22 if pat.category in LATTICE_CATS or pat.category == "resampling" else 0.85
        layers.append((_stripes_alpha(res, pat, base.size, scale), CATEGORIES[pat.category][1], op))
    overlay = _tint(base, layers) if layers else base

    cards = []
    for i, pat in enumerate(shown, 1):
        label, col = CATEGORIES[pat.category]
        where = _tint(base, [(_envelope_alpha(pat, base.size), col, 0.85)])
        pair, gain, (cx0, cy0, csz) = _zoom_pair(res, pat)
        kind = ("pattern isolated and amplified" if pat._rep.view == "A"
                else "roughness rhythm isolated and amplified")
        notes = "".join(f"<p class='note'>{html.escape(n)}</p>" for n in pat.notes)
        cards.append(f"""
<section class="card">
  <h3><span class="chip" style="background:rgb{col}"></span>{i}. {html.escape(label)}</h3>
  <p>{html.escape(EXPLAIN[pat.category])}</p>{notes}
  <dl>
    <dt>Repeats every</dt><dd>{_period_text(pat)}</dd>
    <dt>Direction</dt><dd>{_direction_text(pat)}</dd>
    <dt>Strength</dt><dd>{_strength_text(pat)}</dd>
    <dt>Where</dt><dd>{_where_text(pat)}</dd>
    <dt>Found in</dt><dd>{_channels_text(pat)}</dd>
    <dt>Confidence</dt><dd>{_confidence(pat, thr)}</dd>
    <dt>Spectral peaks</dt><dd>{len(pat.peaks)}</dd>
  </dl>
  <figure><img src="{_b64(where, 'JPEG', quality=85)}" alt="where the pattern is">
    <figcaption>Where it is: brighter colour = the pattern is measured more clearly there.</figcaption></figure>
  <figure><img class="pix" src="{_b64(pair)}" alt="zoom">
    <figcaption>Zoom on a {csz}×{csz}-pixel patch at ({cx0}, {cy0}), shown {ZOOM}×.
    Left: the picture{' with the block grid drawn in' if pat.category in ('jpeg_grid', 'chroma_grid') else ''}.
    Right: {kind} {gain:.0f}×.</figcaption></figure>
</section>""")

    legend = "".join(
        f"<li><span class='chip' style='background:rgb{CATEGORIES[c][1]}'></span>"
        f"{html.escape(CATEGORIES[c][0])}</li>"
        for c in dict.fromkeys(p.category for p in shown))

    spec_html = ""
    for view, title, text in (
            ("A", "Frequency map of the pixel values",
             "Each point is one stripe width and direction: the centre is slow change, the "
             "edges are 2-pixel stripes. The picture's own content has been divided out, so "
             "an even grey means nothing regular; a bright dot is a repeating pattern."),
            ("B", "Frequency map of the roughness",
             "The same map for how rough the image is from pixel to pixel. Compression "
             "blocks and resizing show up here even though they add no visible pattern.")):
        if view in res["spectra"]:
            img = _spectrum_image(res["spectra"][view], shown, view)
            spec_html += (f"<figure><img src='{_b64(img)}' alt='{title}'>"
                          f"<figcaption><b>{title}.</b> {text} Circles mark the detected "
                          f"peaks in the colours above.</figcaption></figure>")

    rows = "".join(
        f"<tr><td>{i}</td><td>{html.escape(CATEGORIES[p.category][0])}</td>"
        f"<td>{_period_text(p)}</td><td>{p._rep.angle:.0f}°</td>"
        f"<td>{'%.2f' % p.measure['amplitude'] if p.measure.get('amplitude') is not None else '–'}</td>"
        f"<td>{'whole' if (p.measure['is_global'] or p.category in ('jpeg_grid', 'chroma_grid')) else 'part'}</td>"
        f"<td>{p.strength / LN2:,.0f}×</td><td>{len(p.peaks)}</td></tr>"
        for i, p in enumerate(pats, 1))
    table = (f"<table><thead><tr><th>#</th><th>Cause</th><th>Repeat</th><th>Angle</th>"
             f"<th>±Levels</th><th>Extent</th><th>Strength</th><th>Peaks</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>") if pats else ""

    summary = "".join(f"<li>{html.escape(s)}</li>" for s in _summary_lines(res))
    history = "".join(f"<li>{html.escape(s)}</li>" for s in _history_text(res))
    notes = "".join(f"<li>{html.escape(s)}</li>" for s in info["notes"])
    if res.get("n_edge_streaks_dropped"):
        notes += (f"<li>{res['n_edge_streaks_dropped']} roughness rhythm(s) were set aside "
                  f"because they are confined to part of the frame (long straight edges and "
                  f"local textures do this).</li>")
    if len(pats) > len(shown):
        notes += f"<li>Only the {len(shown)} strongest patterns are illustrated; all are in the table.</li>"
    fmt = info["format"] + (f", quality ≈ {info['jpeg_quality']}" if info["jpeg_quality"] else "")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pattern scan: {html.escape(os.path.basename(info['path']))}</title>
<style>
:root{{color-scheme:light dark;--bg:#fbfaf8;--fg:#1d1d1f;--mut:#63636b;--line:#dcdad5;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececee;--mut:#a0a0a8;--line:#303036;--card:#1e1e22}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:860px;margin:0 auto;padding:20px 16px 60px}}
h1{{font-size:1.4rem;margin:0 0 2px}} h2{{font-size:1.15rem;margin:34px 0 10px}} h3{{font-size:1.02rem;margin:0 0 8px}}
.meta{{color:var(--mut);font-size:.9rem;margin:0 0 16px;overflow-wrap:anywhere}}
img{{max-width:100%;height:auto;display:block;border-radius:6px}}
img.pix{{image-rendering:pixelated}}
figure{{margin:14px 0}} figcaption{{color:var(--mut);font-size:.86rem;margin-top:6px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}}
.chip{{display:inline-block;width:.8em;height:.8em;border-radius:3px;margin-right:.5em;vertical-align:baseline}}
ul{{padding-left:1.2em;margin:8px 0}} li{{margin:4px 0}}
ul.legend{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.9rem}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:12px 0;font-size:.93rem}}
dt{{color:var(--mut)}} dd{{margin:0}}
.note{{font-size:.93rem;border-left:3px solid var(--line);padding-left:10px;color:var(--mut)}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;font-size:.88rem;width:100%}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--mut);font-weight:600}}
</style></head><body><main>
<h1>Pattern scan</h1>
<p class="meta">{html.escape(os.path.basename(info['path']))} · {info['width']}×{info['height']} px · {html.escape(fmt)}</p>

<figure><img src="{_b64(overlay, 'JPEG', quality=88)}" alt="image with patterns highlighted">
<figcaption>The image, muted, with each detected pattern drawn over it in its colour,
amplified so it can be seen. Stripes too fine for this view are shown as a glow where they occur.</figcaption></figure>
<ul class="legend">{legend}</ul>

<h2>Report</h2>
<ul>{summary}</ul>

{'<h2>Each pattern</h2>' + ''.join(cards) if cards else ''}

<h2>Compression history</h2>
<ul>{history}</ul>

<h2>Frequency maps</h2>
{spec_html}

{'<h2>All patterns</h2><div class="scroll">' + table + '</div>' if table else ''}

<h2>How to read this</h2>
<ul>
<li>Every stripe width from 2 pixels up to a third of the image, in every direction, was tested
in brightness and in both colour channels, {len(res['thresholds'])} scans in all.</li>
<li>A pattern is reported only when it stands far above what the picture's own detail produces
at neighbouring frequencies. "Strength" is that ratio.</li>
<li>"Whole frame" means the pattern has the same strength and alignment in every zone quiet
enough to measure. Real subjects (fabric, fences) fail that test; overlays and processing pass it.</li>
<li>One image has limits: a very faint or very slow ripple can hide under busy detail, an exact
2× enlargement can leave no rhythm, and a cause is named from the pattern's geometry, which
cannot prove who or what put it there.</li>
{notes}
</ul>
</main></body></html>"""
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(doc)

    if out_json:
        data = {
            "file": {k: v for k, v in info.items()},
            "jpeg_block_test": res["jpeg"],
            "summary": _summary_lines(res),
            "patterns": [{
                "category": p.category,
                "label": CATEGORIES[p.category][0],
                "period_px": round(p._rep.period, 3),
                "angle_deg": round(p._rep.angle, 1),
                "frequency_cycles_per_px": [round(p._rep.u, 5), round(p._rep.v, 5)],
                "amplitude_levels": p.measure.get("amplitude"),
                "whole_frame": p.measure["is_global"],
                "zones_present": p.measure["zones_present"],
                "zones_absent": p.measure["zones_absent"],
                "strength_x_background": round(p.strength / LN2, 1),
                "channels": sorted({c for q in p.peaks for c in q.channels}),
                "n_peaks": len(p.peaks),
                "notes": p.notes,
            } for p in pats],
        }
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    return out_html
