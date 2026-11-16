/* Pattern scan: report text and pictures (no DOM). Port of patternscan/report.py.
   Pictures are returned as {w, h, data: Uint8ClampedArray RGBA}. */
'use strict';

(function () {
  const C = (typeof module !== 'undefined' && module.exports) ? require('./core.js') : self.PatternCore;
  const { CATEGORIES, EXPLAIN, LN2, reconstruct, peakPeriod, peakAngle, percentile } = C;
  const MAX_DISPLAY = 1100, MAX_WHERE = 640, MAX_CARDS = 8, CROP = 128, ZOOM = 3;
  const LATTICE_CATS = ['jpeg_grid', 'grid8', 'chroma_grid', 'pixel_lattice'];
  const u8 = v => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

  // ------------------------------------------------------------ text
  function periodText(pat) {
    if (pat.category === 'jpeg_grid' || pat.category === 'grid8') return '8 px (grid)';
    if (pat.category === 'chroma_grid') return '16 px (grid)';
    if (pat.category === 'pixel_lattice') return `${pat.rep.lattice || 2} px (lattice)`;
    return `${peakPeriod(pat.rep).toFixed(2)} px`;
  }
  function directionText(pat) {
    if (LATTICE_CATS.includes(pat.category)) return 'rows and columns';
    const a = peakAngle(pat.rep);
    if (a < 3 || a > 177) return 'vertical stripes (changes left to right)';
    if (Math.abs(a - 90) < 3) return 'horizontal stripes (changes top to bottom)';
    return `diagonal stripes, running at ${((a + 90) % 180).toFixed(0)}° from horizontal`;
  }
  function whereText(pat) {
    const ms = pat.measure, p = ms.zones_present, a = ms.zones_absent;
    if ((pat.category === 'jpeg_grid' || pat.category === 'chroma_grid') && !ms.is_global) {
      return `Whole frame by construction; block edges show most in smooth areas (clear in ${p} of 16 zones).`;
    }
    if (ms.is_global) {
      if (a === 0 && p > 0) {
        return p < 16 ? `Whole frame. Confirmed in ${p} of 16 zones; the others are too busy to tell either way.`
                      : 'Whole frame (all 16 zones).';
      }
      return `Whole frame (present in ${p} zones, absent in ${a}, rest too busy to tell).`;
    }
    if (p + a === 0) return 'Could not be localised (too faint against the picture\'s detail).';
    return `Part of the frame: present in ${p} of 16 zones, clearly absent in ${a}.`;
  }
  function strengthText(pat) {
    const amp = pat.measure.amplitude;
    if (amp === null || amp === undefined) return 'Statistical rhythm (not an added brightness pattern)';
    const vis = amp < 0.3 ? 'far below visibility' : amp < 3 ? 'too faint to see'
      : amp < 8 ? 'barely visible in smooth areas' : 'visible';
    return `±${amp.toFixed(2)} of 255 brightness levels (${vis})`;
  }
  const fmt0 = v => Math.round(v).toLocaleString('en-US');
  function confidenceText(pat, thr) {
    const ratio = pat.strength / thr;
    const word = ratio > 5 ? 'very high' : ratio > 2 ? 'high' : 'moderate';
    return `${word} (${fmt0(pat.strength / LN2)}× the surrounding background; detection limit ${fmt0(thr / LN2)}×)`;
  }
  function channelsText(pat) {
    const names = { Y: 'brightness', Cb: 'blue-yellow colour', Cr: 'red-green colour' };
    const seen = new Set();
    for (const p of pat.peaks) for (const c of Object.keys(p.channels)) seen.add(c);
    return ['Y', 'Cb', 'Cr'].filter(c => seen.has(c)).map(c => names[c]).join(', ');
  }
  function acrossText(pat) {
    const a = pat.across;
    if (!a) return null;
    const agree = `agreement ${a.coherence.toFixed(2)}; unrelated images would give about ${a.chance.toFixed(2)}`;
    if (!a.judged) return `Same alignment in ${a.aligned} of ${a.n} images. At least 5 images are needed to judge whether they share it.`;
    if (a.shared) return `Every image carries it: same alignment in ${a.aligned} of ${a.n} images (${agree}).`;
    return `Not shared: the alignment agrees in only ${a.aligned} of ${a.n} images (${agree}). It comes from one or a few of the images.`;
  }
  function summaryLines(res) {
    const pats = res.patterns, info = res.info, jp = res.jpeg, lines = [];
    const main = pats.filter(p => p.category !== 'trace'), traces = pats.filter(p => p.category === 'trace');
    if (res.stack) {
      lines.push(`${res.stack.n} images of ${res.w}\u00d7${res.h} px were stacked: averaged directly, and as noise residuals. Each picture's own content fades in the stack; whatever all of them carry stays.`);
    }
    if (!pats.length) {
      lines.push(res.stack ? 'Nothing repeating survives the stack: the images share no regular pattern that stands out at any frequency.'
        : 'No repeating pattern stands out from the picture\'s own detail at any frequency, in brightness or colour.');
    } else {
      const n = main.length;
      lines.push(`${n} repeating pattern${n !== 1 ? 's' : ''} found`
        + (traces.length ? `, plus ${traces.length} sub-level trace${traces.length !== 1 ? 's' : ''}.` : '.'));
    }
    for (const p of main) {
      let bit = `${CATEGORIES[p.category][0]}: repeats every ${periodText(p)}`;
      const amp = p.measure.amplitude;
      if (amp !== null && amp !== undefined && !LATTICE_CATS.includes(p.category)) bit += `, ±${amp.toFixed(1)} levels`;
      if (p.category === 'jpeg_grid' || p.category === 'chroma_grid') bit += '.';
      else bit += p.measure.is_global ? ', whole frame.' : ', part of the frame.';
      if (p.across && p.across.judged) bit += p.across.shared ? ' Shared by all the images.' : ' Not shared by the images.';
      lines.push(bit);
    }
    if (info.is_jpeg) {
      const q = info.jpeg_quality ? `, quality about ${info.jpeg_quality}` : '';
      const ss = info.jpeg_subsampling ? `, colour stored at ${info.jpeg_subsampling}` : '';
      lines.push(`The file is a JPEG${q}${ss}.`);
    } else if (jp.detected) {
      const shifted = !(jp.offset_x === 0 && jp.offset_y === 0);
      lines.push('The file is not a JPEG, but its pixels carry JPEG block edges: it was JPEG-compressed at some point before being saved in this format.'
        + (shifted ? ` The block grid starts at pixel (${jp.offset_x}, ${jp.offset_y}), not (0, 0), so the picture was cropped or shifted after that compression.` : ''));
    }
    return lines;
  }
  function historyLines(res) {
    const info = res.info, jp = res.jpeg;
    const rows = [`Block-edge test: horizontal z = ${jp.z_x.toFixed(1)}, vertical z = ${jp.z_y.toFixed(1)} (above 6 in both directions counts as detected).`];
    if (jp.detected) {
      rows.push(`8-pixel block grid found, starting at pixel (${jp.offset_x}, ${jp.offset_y}).`);
      if (info.is_jpeg && !(jp.offset_x === 0 && jp.offset_y === 0)) {
        rows.push('This file\'s own JPEG grid starts at (0, 0), so the grid found here is left over from an EARLIER compression, before a crop or shift.');
      }
    } else {
      rows.push('No block edges found in the pixels' + (info.is_jpeg ? ' (normal for a high-quality JPEG).' : '.'));
    }
    return rows;
  }

  // ------------------------------------------------------------ pictures
  function downscaleRGB(rgb, w, h, maxSide) {       // area average; returns Float64 RGB
    const s = Math.min(1, maxSide / Math.max(w, h));
    if (s >= 1) return { rgb, w, h, scale: 1 };
    const dw = Math.max(1, Math.round(w * s)), dh = Math.max(1, Math.round(h * s));
    const out = new Float64Array(dw * dh * 3), cnt = new Float64Array(dw * dh);
    for (let y = 0; y < h; y++) {
      const dy = Math.min(dh - 1, Math.floor(y * dh / h));
      for (let x = 0; x < w; x++) {
        const d = dy * dw + Math.min(dw - 1, Math.floor(x * dw / w)), sI = (y * w + x) * 3;
        out[d * 3] += rgb[sI]; out[d * 3 + 1] += rgb[sI + 1]; out[d * 3 + 2] += rgb[sI + 2]; cnt[d]++;
      }
    }
    for (let d = 0; d < dw * dh; d++) { out[d * 3] /= cnt[d]; out[d * 3 + 1] /= cnt[d]; out[d * 3 + 2] /= cnt[d]; }
    return { rgb: out, w: dw, h: dh, scale: s };
  }
  function downscaleGray(a, w, h, dw, dh) {
    if (dw === w && dh === h) return a;
    const out = new Float64Array(dw * dh), cnt = new Float64Array(dw * dh);
    for (let y = 0; y < h; y++) {
      const dy = Math.min(dh - 1, Math.floor(y * dh / h));
      for (let x = 0; x < w; x++) {
        const d = dy * dw + Math.min(dw - 1, Math.floor(x * dw / w));
        out[d] += a[y * w + x]; cnt[d]++;
      }
    }
    for (let d = 0; d < out.length; d++) out[d] /= cnt[d];
    return out;
  }
  function resizeBilinear(src, sw, sh, dw, dh) {
    const out = new Float64Array(dw * dh);
    for (let y = 0; y < dh; y++) {
      const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sh / dh - 0.5));
      const y0 = Math.min(sh - 2 < 0 ? 0 : sh - 2, Math.floor(fy)), ty = fy - y0, y1 = Math.min(sh - 1, y0 + 1);
      for (let x = 0; x < dw; x++) {
        const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sw / dw - 0.5));
        const x0 = Math.min(sw - 2 < 0 ? 0 : sw - 2, Math.floor(fx)), tx = fx - x0, x1 = Math.min(sw - 1, x0 + 1);
        out[y * dw + x] = (src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx) * (1 - ty)
                        + (src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx) * ty;
      }
    }
    return out;
  }
  function envelopeAlpha(pat, dw, dh) {
    const ms = pat.measure, env = ms.envelope;
    const top = env.length ? percentile(env, 99) : 0;
    if (!(top > 0)) return new Float64Array(dw * dh);
    const a = Float64Array.from(env, v => Math.pow(Math.min(1, Math.max(0, v / top)), 0.8));
    return resizeBilinear(a, ms.gw, ms.gh, dw, dh);
  }
  function stripesAlpha(res, pat, base) {
    const m = pat.rep;
    if (peakPeriod(m) * base.scale < 3.0) return envelopeAlpha(pat, base.w, base.h);
    const src = res.sources[m.src + ':' + pat.channel];
    const lim = res.w * res.h > 1.2e6 ? 2 : 4;
    const same = pat.peaks.filter(p => p.view === m.view && peakPeriod(p) * base.scale >= 3.0)
      .sort((a, b) => b.score - a.score).slice(0, lim);
    const wave = reconstruct(src, res.w, res.h, same.map(p => [p.u, p.v]), 0, 0, res.h, res.w);
    const abs = Float64Array.from(wave, Math.abs);
    const top = percentile(abs, 97) || 1;
    for (let i = 0; i < wave.length; i++) wave[i] = Math.min(1, Math.max(0, wave[i] / top));
    return downscaleGray(wave, res.w, res.h, base.w, base.h);
  }
  function tint(base, layers) {
    const n = base.w * base.h, data = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      const r = base.rgb[i * 3], g = base.rgb[i * 3 + 1], b = base.rgb[i * 3 + 2];
      const gray = (r + g + b) / 3;
      let o0 = 0.45 * r + 0.35 * gray + 12, o1 = 0.45 * g + 0.35 * gray + 12, o2 = 0.45 * b + 0.35 * gray + 12;
      for (const [alpha, col, op] of layers) {
        const al = alpha[i] * op;
        o0 = o0 * (1 - al) + col[0] * al; o1 = o1 * (1 - al) + col[1] * al; o2 = o2 * (1 - al) + col[2] * al;
      }
      data[i * 4] = u8(o0); data[i * 4 + 1] = u8(o1); data[i * 4 + 2] = u8(o2); data[i * 4 + 3] = 255;
    }
    return { w: base.w, h: base.h, data };
  }
  function plain(base) {
    const n = base.w * base.h, data = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      data[i * 4] = u8(base.rgb[i * 3]); data[i * 4 + 1] = u8(base.rgb[i * 3 + 1]);
      data[i * 4 + 2] = u8(base.rgb[i * 3 + 2]); data[i * 4 + 3] = 255;
    }
    return { w: base.w, h: base.h, data };
  }
  function zoomPair(res, pat) {
    const m = pat.rep, src = res.sources[m.src + ':' + pat.channel];
    const ms = pat.measure, env = ms.envelope;
    let best = 0;
    for (let i = 1; i < env.length; i++) if (env[i] > env[best]) best = i;
    const j = Math.floor(best / ms.gw), i0 = best % ms.gw;
    const cy = Math.trunc((j + 0.5) * ms.cell), cx = Math.trunc((i0 + 0.5) * ms.cell);
    const size = Math.min(CROP, res.h, res.w);
    const y0 = Math.min(Math.max(cy - (size >> 1), 0), res.h - size);
    const x0 = Math.min(Math.max(cx - (size >> 1), 0), res.w - size);
    const same = pat.peaks.filter(p => p.view === m.view).sort((a, b) => b.score - a.score).slice(0, 8);
    const wave = reconstruct(src, res.w, res.h, same.map(p => [p.u, p.v]), y0, x0, size, size);
    const top = percentile(Float64Array.from(wave, Math.abs), 97) || 1;
    const gain = 100 / top;
    const col = CATEGORIES[pat.category][1];
    const Z = size * ZOOM;
    const left = new Uint8ClampedArray(Z * Z * 4), right = new Uint8ClampedArray(Z * Z * 4);
    for (let y = 0; y < Z; y++) for (let x = 0; x < Z; x++) {
      const sy = Math.floor(y / ZOOM), sx = Math.floor(x / ZOOM), d = (y * Z + x) * 4;
      const s = ((sy + y0) * res.w + (sx + x0)) * 3;
      left[d] = u8(res.rgb[s]); left[d + 1] = u8(res.rgb[s + 1]); left[d + 2] = u8(res.rgb[s + 2]); left[d + 3] = 255;
      const t = Math.min(1, Math.max(-1, wave[sy * size + sx] * gain / 100));
      for (let k = 0; k < 3; k++) right[d + k] = u8(128 + t * 110 * (0.35 + 0.65 * col[k] / 255));
      right[d + 3] = 255;
    }
    let grid = false;
    if (pat.category === 'jpeg_grid' || pat.category === 'chroma_grid') {
      grid = true;
      const step = pat.category === 'chroma_grid' ? 16 : 8;
      const ox = res.jpeg.detected ? res.jpeg.offset_x : 0, oy = res.jpeg.detected ? res.jpeg.offset_y : 0;
      const put = (x, y) => { const d = (y * Z + x) * 4; left[d] = 255; left[d + 1] = 150; left[d + 2] = 30; };
      for (let gx = (((ox - x0) % step) + step) % step; gx < size; gx += step) for (let y = 0; y < Z; y++) put(gx * ZOOM, y);
      for (let gy = (((oy - y0) % step) + step) % step; gy < size; gy += step) for (let x = 0; x < Z; x++) put(x, gy * ZOOM);
    }
    return { left: { w: Z, h: Z, data: left }, right: { w: Z, h: Z, data: right }, gain, x0, y0, size, grid };
  }
  function spectrumImage(E, w, h, patterns, view, maxSide) {
    const f = Math.max(1, Math.ceil(Math.max(h, w) / maxSide));
    const ph = Math.floor(h / f), pw = Math.floor(w / f);
    const hh = ph * f, ww = pw * f, y0 = (h - hh) >> 1, x0 = (w - ww) >> 1;
    const data = new Uint8ClampedArray(pw * ph * 4);
    const norm = Math.log1p(60);
    for (let py = 0; py < ph; py++) for (let px = 0; px < pw; px++) {
      let m = 0;
      for (let a = 0; a < f; a++) for (let b = 0; b < f; b++) {
        const v = E[(y0 + py * f + a) * w + x0 + px * f + b]; if (v > m) m = v;
      }
      const g = u8(Math.pow(Math.min(1, Math.log1p(m) / norm), 0.9) * 255), d = (py * pw + px) * 4;
      data[d] = g; data[d + 1] = g; data[d + 2] = g; data[d + 3] = 255;
    }
    const cy = h >> 1, cx = w >> 1, marks = [];
    for (const pat of patterns) {
      const col = CATEGORIES[pat.category][1];
      for (const p of pat.peaks) {
        if (p.view !== view) continue;
        for (const sgn of [1, -1]) marks.push({ x: (cx + sgn * p.bx - x0) / f, y: (cy + sgn * p.by - y0) / f, color: col });
      }
    }
    return { img: { w: pw, h: ph, data }, marks };
  }

  // ------------------------------------------------------------ assemble
  function buildReport(res, progress) {
    progress = progress || (() => {});
    const info = res.info, pats = res.patterns;
    const base = downscaleRGB(res.rgb, res.w, res.h, MAX_DISPLAY);
    const small = downscaleRGB(res.rgb, res.w, res.h, MAX_WHERE);
    const shown = pats.slice(0, MAX_CARDS);
    const anyReal = shown.some(p => p.category !== 'trace');

    const layers = [];
    const ordered = shown.slice().sort((a, b) => (LATTICE_CATS.includes(a.category) ? 0 : 1) - (LATTICE_CATS.includes(b.category) ? 0 : 1));
    let k = 0;
    for (const pat of ordered) {
      if (pat.category === 'trace' && anyReal) continue;
      progress(`Drawing patterns (${++k})`, 0.86 + 0.06 * k / Math.max(1, ordered.length));
      const op = (LATTICE_CATS.includes(pat.category) || pat.category === 'resampling') ? 0.22 : 0.85;
      layers.push([stripesAlpha(res, pat, base), CATEGORIES[pat.category][1], op]);
    }
    const overlay = layers.length ? tint(base, layers) : plain(base);

    const cards = [];
    shown.forEach((pat, i) => {
      progress(`Preparing pattern ${i + 1}`, 0.92 + 0.07 * i / Math.max(1, shown.length));
      const [label, col] = CATEGORIES[pat.category];
      const where = tint(small, [[envelopeAlpha(pat, small.w, small.h), col, 0.85]]);
      const zp = zoomPair(res, pat);
      const kind = pat.rep.view === 'A' ? 'pattern isolated and amplified' : 'roughness rhythm isolated and amplified';
      cards.push({
        n: i + 1, label, color: col, explain: EXPLAIN[pat.category], notes: pat.notes,
        fields: [['Repeats every', periodText(pat)], ['Direction', directionText(pat)],
                 ['Strength', strengthText(pat)]].concat(acrossText(pat) ? [['Across images', acrossText(pat)]] : []).concat([['Where', whereText(pat)],
                 ['Found in', channelsText(pat)], ['Confidence', confidenceText(pat, res.thr)],
                 ['Spectral peaks', String(pat.peaks.length)]]),
        where, zoomLeft: zp.left, zoomRight: zp.right,
        zoomCaption: `Zoom on a ${zp.size}×${zp.size}-pixel patch at (${zp.x0}, ${zp.y0}), shown ${ZOOM}×. Left: the picture${zp.grid ? ' with the block grid drawn in' : ''}. Right: ${kind} ${zp.gain.toFixed(0)}×.`,
      });
    });

    const legend = [];
    for (const p of shown) if (!legend.some(l => l[0] === CATEGORIES[p.category][0])) legend.push(CATEGORIES[p.category]);

    const spectra = [];
    for (const [view, title, text] of [
      ['A', 'Frequency map of the pixel values', 'Each point is one stripe width and direction: the centre is slow change, the edges are 2-pixel stripes. The picture\'s own content has been divided out, so an even grey means nothing regular; a bright dot is a repeating pattern.'],
      ['B', 'Frequency map of the roughness', 'The same map for how rough the image is from pixel to pixel. Compression blocks and resizing show up here even though they add no visible pattern.']]) {
      if (res.spectra[view]) {
        const sp = spectrumImage(res.spectra[view], res.w, res.h, shown, view, 640);
        spectra.push({ title, text, img: sp.img, marks: sp.marks });
      }
    }

    // what the stack itself looks like: the two classic pictures
    const shared = [];
    if (res.stack && res.stackRes) {
      const cw = Math.min(res.w, MAX_DISPLAY), ch = Math.min(res.h, MAX_DISPLAY);
      const x0 = (res.w - cw) >> 1, y0 = (res.h - ch) >> 1;
      const panel = (a, title, text) => {
        const crop = new Float64Array(cw * ch);
        for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = a[(y + y0) * res.w + x + x0];
        const top = percentile(Float64Array.from(crop, Math.abs), 99) || 1e-9;
        const gain = 110 / top, data = new Uint8ClampedArray(cw * ch * 4);
        for (let i = 0; i < cw * ch; i++) {
          const g = u8(128 + crop[i] * gain);
          data[i * 4] = g; data[i * 4 + 1] = g; data[i * 4 + 2] = g; data[i * 4 + 3] = 255;
        }
        shared.push({ title, img: { w: cw, h: ch, data },
          text: `${text} Brightness differences are amplified ${gain >= 10 ? gain.toFixed(0) : gain.toFixed(1)}\u00d7.` });
      };
      const mY = res.chans.Y, lp = C.smooth3(mY, res.w, res.h, 31), det = new Float64Array(mY.length);
      for (let i = 0; i < det.length; i++) det[i] = mY[i] - lp[i];
      panel(res.stackRes.res.Y, 'Stacked noise residual',
        `Each image minus a smoothed copy of itself, averaged over all ${res.stack.n} images with more weight where an image is smooth. The pictures cancel; a pattern they share does not. An even grey field means they share nothing fine.`);
      panel(det, 'Average of the images',
        `The plain average of all ${res.stack.n} images with slow shading removed (flat-image averaging). On flat or plain images this shows a shared pattern directly, including wide stripes the residual cannot hold.`);
    }

    const table = pats.map((p, i) => ({
      n: i + 1, cause: CATEGORIES[p.category][0], repeat: periodText(p), angle: `${peakAngle(p.rep).toFixed(0)}°`,
      levels: (p.measure.amplitude !== null && p.measure.amplitude !== undefined) ? p.measure.amplitude.toFixed(2) : '–',
      extent: (p.measure.is_global || p.category === 'jpeg_grid' || p.category === 'chroma_grid') ? 'whole' : 'part',
      shared: p.across ? (p.across.judged ? (p.across.shared ? 'yes' : 'no') : '?') : null,
      strength: `${fmt0(p.strength / LN2)}×`, peaks: p.peaks.length,
    }));

    const notes = info.notes.slice();
    if (res.dropped) notes.push(`${res.dropped} weak rhythm(s) were set aside because they are confined to one part of the frame (long straight edges and local textures do this).`);
    if (pats.length > shown.length) notes.push(`Only the ${shown.length} strongest patterns are illustrated; all are in the table.`);

    const fmt = info.format + (info.jpeg_quality ? `, quality ≈ ${info.jpeg_quality}` : '');
    const report = {
      meta: { name: info.path, width: info.width, height: info.height, format: fmt, scans: res.n_scans,
              stack: res.stack || null },
      overlay, legend, summary: summaryLines(res), cards, history: historyLines(res), spectra, table, notes, shared,
    };
    const transfer = [overlay.data.buffer];
    for (const sh of shared) transfer.push(sh.img.data.buffer);
    for (const c of cards) transfer.push(c.where.data.buffer, c.zoomLeft.data.buffer, c.zoomRight.data.buffer);
    for (const s of spectra) transfer.push(s.img.data.buffer);
    return { report, transfer };
  }

  const PatternRender = { buildReport, summaryLines, historyLines, periodText };
  if (typeof module !== 'undefined' && module.exports) module.exports = PatternRender;
  if (typeof self !== 'undefined') self.PatternRender = PatternRender;
})();
