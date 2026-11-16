/* Pattern scan: analysis core (no DOM). Runs in a Web Worker and in Node.
   A line-for-line port of patternscan/analyze.py; tests/compare_js.py checks
   that both give the same findings on the same pixels. */
'use strict';

const LN2 = Math.LN2;
const CFG = {
  ALPHA: 0.01, SAFETY: 1.6, ISOLATION: 5.0, MIN_BINS_FROM_DC: 3,
  MAX_PEAKS: 60, LATTICE_TOL: 1.5, GLOBAL_SIGN: 0.85,
  TRACE_AMPLITUDE: 0.3, MIN_REGIONAL_AMPLITUDE: 1.5,
};

const CATEGORIES = {
  jpeg_grid: ['JPEG compression blocks', [255, 150, 30]],
  grid8: ['8-pixel lattice (not confirmed as JPEG)', [235, 190, 60]],
  chroma_grid: ['16-pixel colour blocks', [190, 120, 60]],
  pixel_lattice: ['2/4-pixel lattice', [170, 110, 255]],
  resampling: ['Resize / rotate trace', [60, 150, 255]],
  global_faint: ['Faint global overlay (watermark-like)', [255, 50, 120]],
  global_visible: ['Strong global pattern', [255, 225, 40]],
  regional: ['Repeating texture in part of the image', [60, 210, 110]],
  trace: ['Sub-level processing trace', [150, 160, 170]],
};

const EXPLAIN = {
  jpeg_grid: 'JPEG stores the picture in 8x8-pixel blocks. Tiny steps at the block edges repeat every 8 pixels across the whole image.',
  grid8: 'A pattern that repeats every 8 pixels, but without the small steps at block edges that JPEG leaves. Possible causes: an AI generator\'s upsampling stage, heavy processing after a JPEG save, or a pattern placed on that lattice.',
  chroma_grid: 'Repeats every 16 pixels, mainly in colour. JPEG usually stores colour at half resolution, which makes colour blocks 16 pixels wide.',
  pixel_lattice: 'Repeats every 2 or 4 pixels exactly. Typical causes: the camera sensor\'s colour mosaic, enlarging by a whole factor, the upsampling layers of an AI image generator, or a pattern placed on that lattice.',
  resampling: 'Resizing or rotating recomputes every pixel from its neighbours, which leaves a regular rhythm in how rough the image is. It is not a drawn pattern; it is a statistical trace.',
  global_faint: 'A regular ripple added to the pixel values over the whole frame with the same alignment everywhere, too faint to see, and not explained by JPEG blocks, pixel lattices or resizing. Invisible watermarks, electrical interference and scanner noise look like this.',
  global_visible: 'A regular pattern covering the whole frame with the same alignment everywhere, strong enough to see: stripes, scan lines, a photographed screen, halftone print dots, or moire.',
  regional: 'A repeating pattern that is present in part of the image or changes alignment from place to place. That is how real subjects behave: fabric, fences, bricks, blinds, text lines.',
  trace: 'A regular ripple smaller than a third of one brightness step. Colour conversion, resizing and compression leave such residues; it is far below anything visible and is listed for completeness.',
};

// ---------------------------------------------------------------- numerics
function hyp(a, b) { return Math.sqrt(a * a + b * b); }   // Math.hypot is slow in hot loops
function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function quickselect(a, k) {      // k-th smallest, partially sorts a in place
  let lo = 0, hi = a.length - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j; else if (k >= i) lo = i; else break;
  }
  return a[k];
}

function median(arr) {           // arr: plain array or typed array (not modified)
  const a = Float64Array.from(arr);
  const n = a.length;
  if (n === 0) return NaN;
  if (n % 2) return quickselect(a, (n - 1) >> 1);
  const hiV = quickselect(a, n / 2);
  let loV = -Infinity;
  for (let i = 0; i < n / 2; i++) if (a[i] > loV) loV = a[i];
  return 0.5 * (loV + hiV);
}

function smallMedian(buf, n) {   // median of buf[0..n), n small; sorts buf in place
  for (let i = 1; i < n; i++) {
    const v = buf[i];
    let j = i - 1;
    while (j >= 0 && buf[j] > v) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = v;
  }
  return n % 2 ? buf[(n - 1) >> 1] : 0.5 * (buf[n / 2 - 1] + buf[n / 2]);
}

function percentile(arr, q) {    // numpy default (linear interpolation)
  const a = Float64Array.from(arr).sort();
  const n = a.length;
  if (n === 0) return NaN;
  const pos = (q / 100) * (n - 1);
  const i = Math.floor(pos), f = pos - i;
  return i + 1 < n ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

/* Box blur with edge replication; same window placement as
   scipy.ndimage.uniform_filter(size, mode="nearest"). */
function boxFilter(src, w, h, size) {
  const half = size >> 1, inv = 1 / size;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  // horizontal: sliding sum over [x - half, x - half + size - 1], clamped
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let sum = 0;
    for (let k = 0; k < size; k++) sum += src[o + clamp(k - half, 0, w - 1)];
    tmp[o] = sum * inv;
    for (let x = 1; x < w; x++) {
      sum += src[o + clamp(x - half + size - 1, 0, w - 1)] - src[o + clamp(x - 1 - half, 0, w - 1)];
      tmp[o + x] = sum * inv;
    }
  }
  // vertical: one running sum per column, updated row by row
  const col = new Float64Array(w);
  for (let k = 0; k < size; k++) {
    const o = clamp(k - half, 0, h - 1) * w;
    for (let x = 0; x < w; x++) col[x] += tmp[o + x];
  }
  for (let x = 0; x < w; x++) out[x] = col[x] * inv;
  for (let y = 1; y < h; y++) {
    const oa = clamp(y - half + size - 1, 0, h - 1) * w, os = clamp(y - 1 - half, 0, h - 1) * w, o = y * w;
    for (let x = 0; x < w; x++) {
      col[x] += tmp[oa + x] - tmp[os + x];
      out[o + x] = col[x] * inv;
    }
  }
  return out;
}

function smooth3(a, w, h, size) {
  return boxFilter(boxFilter(boxFilter(a, w, h, size), w, h, size), w, h, size);
}

// ---------------------------------------------------------------- FFT
const _plans = new Map();
function fftPlan(n) {
  let p = _plans.get(n);
  if (p) return p;
  if ((n & (n - 1)) === 0) {
    const rev = new Uint32Array(n);
    let bits = 0;
    while ((1 << bits) < n) bits++;
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos(2 * Math.PI * i / n);
      sin[i] = Math.sin(2 * Math.PI * i / n);
    }
    p = { n, pow2: true, rev, cos, sin };
  } else {
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;
    const wr = new Float64Array(n), wi = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const ang = Math.PI * ((k * k) % (2 * n)) / n;
      wr[k] = Math.cos(ang); wi[k] = -Math.sin(ang);      // exp(-i*pi*k^2/n)
    }
    const br = new Float64Array(m), bi = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      br[k] = wr[k]; bi[k] = -wi[k];
      if (k) { br[m - k] = wr[k]; bi[m - k] = -wi[k]; }
    }
    const sub = fftPlan(m);
    fftRadix2(sub, br, bi, false);
    p = { n, pow2: false, m, wr, wi, br, bi, sub,
          ar: new Float64Array(m), ai: new Float64Array(m) };
  }
  _plans.set(n, p);
  return p;
}

function fftRadix2(p, re, im, inverse) {
  const n = p.n, rev = p.rev, cos = p.cos, sin = p.sin;
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const c = cos[k * step], s = inverse ? sin[k * step] : -sin[k * step];
        const a = i + k, b = a + half;
        const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function fft1d(p, re, im) {            // forward, in place, any length
  if (p.pow2) { fftRadix2(p, re, im, false); return; }
  const n = p.n, m = p.m, ar = p.ar, ai = p.ai;
  ar.fill(0); ai.fill(0);
  for (let k = 0; k < n; k++) {
    ar[k] = re[k] * p.wr[k] - im[k] * p.wi[k];
    ai[k] = re[k] * p.wi[k] + im[k] * p.wr[k];
  }
  fftRadix2(p.sub, ar, ai, false);
  for (let k = 0; k < m; k++) {
    const r = ar[k] * p.br[k] - ai[k] * p.bi[k];
    ai[k] = ar[k] * p.bi[k] + ai[k] * p.br[k];
    ar[k] = r;
  }
  fftRadix2(p.sub, ar, ai, true);
  for (let k = 0; k < n; k++) {
    re[k] = ar[k] * p.wr[k] - ai[k] * p.wi[k];
    im[k] = ar[k] * p.wi[k] + ai[k] * p.wr[k];
  }
}

function fft2d(re, im, w, h) {
  const pw = fftPlan(w), ph = fftPlan(h);
  const tr = new Float64Array(w), ti = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    tr.set(re.subarray(o, o + w)); ti.set(im.subarray(o, o + w));
    fft1d(pw, tr, ti);
    re.set(tr, o); im.set(ti, o);
  }
  const cr = new Float64Array(h), ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) { cr[y] = re[y * w + x]; ci[y] = im[y * w + x]; }
    fft1d(ph, cr, ci);
    for (let y = 0; y < h; y++) { re[y * w + x] = cr[y]; im[y * w + x] = ci[y]; }
  }
}

// ---------------------------------------------------------------- file header
const STD_LUMA_Q = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];

function estimateJpegQuality(table) {
  if (!table || table.length !== 64) return null;
  const q = Float64Array.from(table).sort();
  const ref = Float64Array.from(STD_LUMA_Q).sort();
  let best = null, bestErr = 1e18;
  for (let quality = 1; quality <= 100; quality++) {
    const scale = quality < 50 ? 5000 / quality : 200 - 2 * quality;
    const t = Float64Array.from(ref, r => clamp(Math.floor((r * scale + 50) / 100), 1, 255)).sort();
    let err = 0;
    for (let i = 0; i < 64; i++) err += Math.abs(t[i] - q[i]);
    if (err < bestErr) { best = quality; bestErr = err; }
  }
  return best;
}

/* Format, JPEG quality tables and colour subsampling, read from the file bytes. */
function parseFileHeader(bytes, name) {
  const info = { path: name || 'image', format: 'unknown', is_jpeg: false,
                 jpeg_quality: null, jpeg_subsampling: null };
  const b = bytes;
  if (!b || b.length < 12) return info;
  if (b[0] === 0xFF && b[1] === 0xD8 && b[2] === 0xFF) {
    info.format = 'JPEG'; info.is_jpeg = true;
    const tables = {};
    let i = 2;
    while (i + 4 <= b.length) {
      if (b[i] !== 0xFF) { i++; continue; }
      const marker = b[i + 1];
      if (marker === 0xFF) { i++; continue; }
      if (marker === 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker === 0x01) { i += 2; continue; }
      if (marker === 0xD9 || marker === 0xDA) break;
      const len = (b[i + 2] << 8) | b[i + 3];
      if (marker === 0xDB) {
        let j = i + 4;
        const end = i + 2 + len;
        while (j < end) {
          const pq = b[j] >> 4, tq = b[j] & 15;
          j++;
          const t = [];
          for (let k = 0; k < 64; k++) {
            if (pq) { t.push((b[j] << 8) | b[j + 1]); j += 2; } else { t.push(b[j]); j++; }
          }
          tables[tq] = t;
        }
      } else if (marker >= 0xC0 && marker <= 0xCF && marker !== 0xC4 && marker !== 0xC8 && marker !== 0xCC) {
        const nc = b[i + 9];
        if (nc >= 1) {
          const hv = b[i + 11];
          const hs = hv >> 4, vs = hv & 15;
          info.jpeg_subsampling = nc === 1 ? '4:4:4'
            : (hs === 1 && vs === 1) ? '4:4:4' : (hs === 2 && vs === 1) ? '4:2:2'
            : (hs === 2 && vs === 2) ? '4:2:0' : null;
        }
      }
      i += 2 + len;
    }
    const ids = Object.keys(tables).map(Number).sort((x, y) => x - y);
    if (ids.length) info.jpeg_quality = estimateJpegQuality(tables[ids[0]]);
  } else if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4E && b[3] === 0x47) info.format = 'PNG';
  else if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) info.format = 'GIF';
  else if (b[0] === 0x52 && b[1] === 0x49 && b[8] === 0x57 && b[9] === 0x45) info.format = 'WebP';
  else if (b[0] === 0x42 && b[1] === 0x4D) info.format = 'BMP';
  else if ((b[0] === 0x49 && b[1] === 0x49) || (b[0] === 0x4D && b[1] === 0x4D)) info.format = 'TIFF';
  else if (b[4] === 0x66 && b[5] === 0x74 && b[6] === 0x79 && b[7] === 0x70) info.format = 'AVIF/HEIC';
  return info;
}

// ---------------------------------------------------------------- loading
function loadPixels(rgba, W, H, info, maxSide) {
  info.width = W; info.height = H; info.notes = [];
  let x0 = 0, y0 = 0, cw = W, ch = H;
  if (Math.max(W, H) > maxSide) {
    ch = Math.min(H, maxSide); cw = Math.min(W, maxSide);
    y0 = Math.floor(Math.floor((H - ch) / 2) / 16) * 16;
    x0 = Math.floor(Math.floor((W - cw) / 2) / 16) * 16;
    info.notes.push(`Image is large; analysed the central ${cw}x${ch} pixels at full resolution (starting at ${x0},${y0}). Pictures and positions in this report refer to that region. Choose a larger analysis size to cover more.`);
  }
  info.analysed_width = cw; info.analysed_height = ch;
  const n = cw * ch;
  const rgb = new Float64Array(n * 3);
  let gray = true, hasAlpha = false;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const s = ((y + y0) * W + (x + x0)) * 4, d = (y * cw + x) * 3;
      let r = rgba[s], g = rgba[s + 1], b = rgba[s + 2];
      const a = rgba[s + 3];
      if (a < 255) {
        hasAlpha = true;
        const t = a / 255;
        r = Math.round(r * t + 255 * (1 - t));
        g = Math.round(g * t + 255 * (1 - t));
        b = Math.round(b * t + 255 * (1 - t));
      }
      rgb[d] = r; rgb[d + 1] = g; rgb[d + 2] = b;
      if (r !== g || g !== b) gray = false;
    }
  }
  if (hasAlpha) info.notes.push('Transparency was flattened onto white before analysis.');
  const chans = { Y: new Float64Array(n) };
  if (!gray) { chans.Cb = new Float64Array(n); chans.Cr = new Float64Array(n); }
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    chans.Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    if (!gray) {
      chans.Cb[i] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      chans.Cr[i] = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
  }
  return { chans, rgb, w: cw, h: ch };
}

// ---------------------------------------------------------------- spectra
function roughnessLap(x, w, h) {
  const out = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let xx = 1; xx < w - 1; xx++) {
      const i = y * w + xx;
      out[i] = Math.abs(4 * x[i] - x[i - w] - x[i + w] - x[i - 1] - x[i + 1]);
    }
  }
  return out;
}

function roughnessStep(x, w, h) {
  const r = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      const i = y * w + xx;
      let v = 0;
      if (xx < w - 1) v += Math.abs(x[i + 1] - x[i]);
      if (y < h - 1) v += Math.abs(x[i + w] - x[i]);
      r[i] = v;
    }
  }
  for (let y = 0; y < h; y++) r[y * w + w - 1] = r[y * w + w - 2];
  for (let xx = 0; xx < w; xx++) r[(h - 1) * w + xx] = r[(h - 2) * w + xx];
  return r;
}

function flatnessWeight(x, w, h, scale) {
  const sm = scale <= 3 ? boxFilter(x, w, h, 3) : smooth3(x, w, h, scale);
  const d2 = new Float64Array(w * h);
  for (let i = 0; i < d2.length; i++) { const d = x[i] - sm[i]; d2[i] = d * d; }
  const v = smooth3(d2, w, h, Math.max(7, scale));
  const v0 = Math.max(0.5, 0.25 * median(v));
  for (let i = 0; i < v.length; i++) v[i] = v0 / (v[i] + v0);
  return v;
}

function hanning(n) {
  const a = new Float64Array(n);
  if (n === 1) { a[0] = 1; return a; }
  for (let i = 0; i < n; i++) a[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
  return a;
}

/* E (fft-shifted): pure noise gives Exp(1)-like values; P: raw power (shifted). */
function whitenedSpectrum(x, w, h, weight) {
  const n = w * h;
  const wy = hanning(h), wx = hanning(w);
  const re = new Float64Array(n), im = new Float64Array(n);
  let sw = 0, sxw = 0;
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      const i = y * w + xx;
      const wv = wy[y] * wx[xx] * (weight ? weight[i] : 1);
      re[i] = wv;                       // keep the window for the second pass
      sw += wv; sxw += x[i] * wv;
    }
  }
  const mean = sxw / sw;
  for (let i = 0; i < n; i++) re[i] = (x[i] - mean) * re[i];
  fft2d(re, im, w, h);
  const hy = h >> 1, hx = w >> 1;
  const P = new Float64Array(n);
  for (let y = 0; y < h; y++) {
    const sy = (y + hy) % h;
    for (let xx = 0; xx < w; xx++) {
      const sx = (xx + hx) % w, i = y * w + xx;
      P[sy * w + sx] = Math.max(re[i] * re[i] + im[i] * im[i], 1e-20);
    }
  }
  // 1) radial 1/f-type slope
  const nb = 256;
  const s = new Float64Array(nb), c = new Float64Array(nb);
  const rad = im;                       // reuse memory
  for (let y = 0; y < h; y++) {
    const fy = (y - hy) / h;
    for (let xx = 0; xx < w; xx++) {
      const fx = (xx - hx) / w, i = y * w + xx;
      const r = hyp(fx, fy);
      rad[i] = r;
      const idx = Math.min(Math.floor(r / 0.7072 * nb), nb - 1);
      s[idx] += Math.log(P[i]); c[idx] += 1;
    }
  }
  const good = [];
  for (let k = 0; k < nb; k++) if (c[k] > 0) good.push(k);
  const prof = new Float64Array(nb);
  for (let k = 0, g = 0; k < nb; k++) {       // np.interp over the non-empty bins
    while (g + 1 < good.length && good[g + 1] <= k) g++;
    const k0 = good[g], v0 = s[k0] / c[k0];
    if (k <= good[0]) prof[k] = s[good[0]] / c[good[0]];
    else if (g + 1 >= good.length) prof[k] = v0;
    else {
      const k1 = good[g + 1], v1 = s[k1] / c[k1];
      prof[k] = v0 + (v1 - v0) * (k - k0) / (k1 - k0);
    }
  }
  const Pn = re;                        // reuse memory
  for (let i = 0; i < n; i++) {
    const pos = clamp(rad[i] / 0.7072 * nb - 0.5, 0, nb - 1);
    const k = Math.min(Math.floor(pos), nb - 2), f = pos - k;
    Pn[i] = P[i] / Math.exp(prof[k] * (1 - f) + prof[k + 1] * f);
  }
  // 2) local background that also follows horizontal / vertical ridges
  const cell = clamp(Math.floor(Math.min(h, w) / 16), 8, 32);
  const E = new Float64Array(n);
  {
    const nc = Math.max(1, Math.floor(w / cell));
    const med = new Float64Array(nc), buf = new Float64Array(cell);
    for (let y = 0; y < h; y++) {
      for (let k = 0; k < nc; k++) {
        for (let j = 0; j < cell; j++) buf[j] = Pn[y * w + k * cell + j];
        med[k] = smallMedian(buf, cell);
      }
      for (let xx = 0; xx < w; xx++) {
        let v;
        if (nc === 1) v = med[0];
        else {
          const pos = clamp((xx - (cell - 1) / 2) / cell, 0, nc - 1);
          const i0 = Math.min(Math.floor(pos), nc - 2), f = pos - i0;
          v = med[i0] * (1 - f) + med[i0 + 1] * f;
        }
        E[y * w + xx] = v;
      }
    }
  }
  {
    const nc = Math.max(1, Math.floor(h / cell));
    const med = new Float64Array(nc), buf = new Float64Array(cell);
    for (let xx = 0; xx < w; xx++) {
      for (let k = 0; k < nc; k++) {
        for (let j = 0; j < cell; j++) buf[j] = Pn[(k * cell + j) * w + xx];
        med[k] = smallMedian(buf, cell);
      }
      for (let y = 0; y < h; y++) {
        let v;
        if (nc === 1) v = med[0];
        else {
          const pos = clamp((y - (cell - 1) / 2) / cell, 0, nc - 1);
          const i0 = Math.min(Math.floor(pos), nc - 2), f = pos - i0;
          v = med[i0] * (1 - f) + med[i0 + 1] * f;
        }
        const i = y * w + xx;
        const B = Math.max(E[i], v);
        E[i] = LN2 * Pn[i] / Math.max(B, 1e-20);
      }
    }
  }
  return { E, P };
}

function peakFreq(p) { return hyp(p.u, p.v); }
function peakPeriod(p) { return 1 / Math.max(peakFreq(p), 1e-9); }
function peakAngle(p) {
  const a = Math.atan2(p.v, p.u) * 180 / Math.PI;
  return ((a % 180) + 180) % 180;
}

function findPeaks(E, P, w, h, view, channel) {
  const cy = h >> 1, cx = w >> 1;
  const thr = CFG.SAFETY * Math.log(h * w / 2 / CFG.ALPHA);
  const at = (y, x) => E[(((y % h) + h) % h) * w + (((x % w) + w) % w)];
  const pat = (y, x) => P[(((y % h) + h) % h) * w + (((x % w) + w) % w)];
  // 7x7 maximum filter with wrap-around (separable)
  const rowMax = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let m = -Infinity;
      for (let d = -3; d <= 3; d++) {
        let xx = x + d; if (xx < 0) xx += w; else if (xx >= w) xx -= w;
        const v = E[o + xx]; if (v > m) m = v;
      }
      rowMax[o + x] = m;
    }
  }
  const cands = [];
  for (let y = 0; y < h; y++) {
    if (y >= 1 && y < cy) continue;
    for (let x = 0; x < w; x++) {
      if ((y === cy || y === 0) && x >= 1 && x <= cx) continue;
      const v = E[y * w + x];
      if (v <= thr) continue;
      let m = -Infinity;
      for (let d = -3; d <= 3; d++) {
        let yy = y + d; if (yy < 0) yy += h; else if (yy >= h) yy -= h;
        const r = rowMax[yy * w + x]; if (r > m) m = r;
      }
      if (v === m) cands.push([v, y, x]);
    }
  }
  cands.sort((a, b) => b[0] - a[0]);
  const ring = [];
  for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
    const r = hyp(dy, dx);
    if (r > 3.5 && r <= 10) ring.push([dy, dx]);
  }
  const refine = (a, b, c) => {
    a = Math.log(a); b = Math.log(b); c = Math.log(c);
    const d = a - 2 * b + c;
    return d >= 0 ? 0 : clamp(0.5 * (a - c) / d, -0.5, 0.5);
  };
  const peaks = [];
  const lim = Math.min(cands.length, CFG.MAX_PEAKS * 3);
  const ringVals = new Float64Array(ring.length);
  for (let ci = 0; ci < lim; ci++) {
    const [val, y, x] = cands[ci];
    if (hyp(y - cy, x - cx) < CFG.MIN_BINS_FROM_DC) continue;
    for (let k = 0; k < ring.length; k++) ringVals[k] = at(y + ring[k][0], x + ring[k][1]);
    if (val < CFG.ISOLATION * percentile(ringVals, 85)) continue;
    // must also stand out along the line through the centre (edge streaks do not)
    const ry = y - cy, rx = x - cx, rr = hyp(ry, rx);
    const along = [];
    for (let dist = 5; dist < 27; dist += 2) {
      for (const sgn of [1, -1]) {
        if (rr + sgn * dist < CFG.MIN_BINS_FROM_DC + 2) continue;
        const t = (rr + sgn * dist) / rr;
        const sy = Math.round(cy + ry * t), sx = Math.round(cx + rx * t);
        if (sy >= -29 && sy < h + 29 && sx >= -29 && sx < w + 29) {
          let m = -Infinity;
          for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
            const v = at(sy + a, sx + b); if (v > m) m = v;
          }
          along.push(m);
        }
      }
    }
    if (along.length >= 6 && val < CFG.ISOLATION * 2.0 * median(along)) continue;
    // locate on the raw power
    let bi = 0, bj = 0, bm = -Infinity;
    for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) {
      const v = pat(y - 2 + j, x - 2 + i);
      if (v > bm) { bm = v; bj = j; bi = i; }
    }
    let dx, dy;
    if (bj > 0 && bj < 4 && bi > 0 && bi < 4) {
      const py = y - 2 + bj, px = x - 2 + bi;
      dx = refine(pat(py, px - 1), pat(py, px), pat(py, px + 1)) + (bi - 2);
      dy = refine(pat(py - 1, px), pat(py, px), pat(py + 1, px)) + (bj - 2);
    } else { dx = bi - 2; dy = bj - 2; }
    const bx = x + dx - cx, by = y + dy - cy;
    peaks.push({ view, channel, u: bx / w, v: by / h, score: val, bx, by,
                 lattice: 0, weighted: false, src: 'val', channels: {} });
    if (peaks.length >= CFG.MAX_PEAKS) break;
  }
  return { peaks, thr };
}

function wrap(d) { return ((d + 0.5) % 1 + 1) % 1 - 0.5; }

function latticeOf(u, v, w, h) {
  for (const base of [2, 4, 8, 16]) {
    const lu = Math.round(u * base) / base, lv = Math.round(v * base) / base;
    const tu = (Math.abs(lu) === 0.5 ? 3.5 : CFG.LATTICE_TOL) / w;
    const tv = (Math.abs(lv) === 0.5 ? 3.5 : CFG.LATTICE_TOL) / h;
    if (Math.abs(u - lu) < tu && Math.abs(v - lv) < tv && !(lu === 0 && lv === 0)) return base;
  }
  return 0;
}

// ---------------------------------------------------------------- measuring
function boxFor(period, h, w) {
  return Math.trunc(clamp(Math.round(1.25 * period), 8, Math.max(8, Math.floor(Math.min(h, w) / 8))));
}

function demodulateAll(x, w, h, u, v, weight) {
  const period = 1 / Math.max(hyp(u, v), 1e-9);
  const box = boxFor(period, h, w);
  const lp = boxFilter(x, w, h, box);
  const cell = Math.max(2, box >> 2);
  const gh = Math.floor(h / cell), gw = Math.floor(w / cell);
  const hh = gh * cell, ww = gw * cell;
  const gn = gw * gh;
  let wc = new Float64Array(gn);
  if (!weight) wc.fill(1);
  else {
    for (let y = 0; y < hh; y++) {
      const gy = Math.floor(y / cell);
      for (let xx = 0; xx < ww; xx++) wc[gy * gw + Math.floor(xx / cell)] += weight[y * w + xx];
    }
    for (let i = 0; i < gn; i++) wc[i] /= cell * cell;
    wc = smooth3(wc, gw, gh, 4);
  }
  const d = 2 / box;
  const freqs = [[u, v], [u + d, v], [u - d, v], [u, v + d], [u, v - d]];
  const out = [];
  const cosx = new Float64Array(ww), sinx = new Float64Array(ww);
  const rowRe = new Float64Array(gw), rowIm = new Float64Array(gw);
  for (const [fu, fv] of freqs) {
    for (let xx = 0; xx < ww; xx++) {
      cosx[xx] = Math.cos(2 * Math.PI * fu * xx); sinx[xx] = Math.sin(2 * Math.PI * fu * xx);
    }
    let zr = new Float64Array(gn), zi = new Float64Array(gn);
    for (let y = 0; y < hh; y++) {
      rowRe.fill(0); rowIm.fill(0);
      const o = y * w;
      for (let xx = 0; xx < ww; xx++) {
        let val = x[o + xx] - lp[o + xx];
        if (weight) val *= weight[o + xx];
        const g = Math.floor(xx / cell);
        rowRe[g] += val * cosx[xx]; rowIm[g] -= val * sinx[xx];
      }
      const cyv = Math.cos(2 * Math.PI * fv * y), syv = -Math.sin(2 * Math.PI * fv * y);
      const go = Math.floor(y / cell) * gw;
      for (let g = 0; g < gw; g++) {
        zr[go + g] += rowRe[g] * cyv - rowIm[g] * syv;
        zi[go + g] += rowRe[g] * syv + rowIm[g] * cyv;
      }
    }
    const norm = cell * cell;
    for (let i = 0; i < gn; i++) { zr[i] /= norm; zi[i] /= norm; }
    zr = smooth3(zr, gw, gh, 4); zi = smooth3(zi, gw, gh, 4);
    for (let i = 0; i < gn; i++) { const q = Math.max(wc[i], 1e-9); zr[i] /= q; zi[i] /= q; }
    out.push({ re: zr, im: zi });
  }
  return { A: out[0], decoys: out.slice(1), wc, cell, box, gw, gh };
}

function measure(x, w, h, u, v, weight) {
  const dm = demodulateAll(x, w, h, u, v, weight);
  const { A, decoys, wc, cell, gw, gh } = dm;
  const gn = gw * gh;
  // residual frequency error -> slow phase drift; estimate from zone averages
  const nzd = 8;
  if (gh >= 2 * nzd && gw >= 2 * nzd) {
    const sy = Math.floor(gh / nzd), sx = Math.floor(gw / nzd);
    const Zr = new Float64Array(nzd * nzd), Zi = new Float64Array(nzd * nzd);
    for (let j = 0; j < sy * nzd; j++) for (let i = 0; i < sx * nzd; i++) {
      const k = Math.floor(j / sy) * nzd + Math.floor(i / sx), g = j * gw + i;
      Zr[k] += A.re[g] * wc[g]; Zi[k] += A.im[g] * wc[g];
    }
    let xr = 0, xi = 0, yr = 0, yi = 0;
    for (let j = 0; j < nzd; j++) for (let i = 0; i < nzd; i++) {
      const k = j * nzd + i;
      if (i + 1 < nzd) {            // Z[j][i+1] * conj(Z[j][i])
        xr += Zr[k + 1] * Zr[k] + Zi[k + 1] * Zi[k];
        xi += Zi[k + 1] * Zr[k] - Zr[k + 1] * Zi[k];
      }
      if (j + 1 < nzd) {
        yr += Zr[k + nzd] * Zr[k] + Zi[k + nzd] * Zi[k];
        yi += Zi[k + nzd] * Zr[k] - Zr[k + nzd] * Zi[k];
      }
    }
    const gx = Math.atan2(xi, xr) / sx, gy = Math.atan2(yi, yr) / sy;
    const limx = 1.5 * 2 * Math.PI * cell / w, limy = 1.5 * 2 * Math.PI * cell / h;
    if (Math.abs(gx) <= limx && Math.abs(gy) <= limy) {
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const ang = -(gx * i + gy * j), c = Math.cos(ang), s = Math.sin(ang), g = j * gw + i;
        const r = A.re[g] * c - A.im[g] * s;
        A.im[g] = A.re[g] * s + A.im[g] * c; A.re[g] = r;
      }
    }
  }
  const mag = new Float64Array(gn), noise = new Float64Array(gn);
  let wsum = 0;
  for (let g = 0; g < gn; g++) {
    mag[g] = hyp(A.re[g], A.im[g]);
    let fs = 0, fmin = Infinity, fmax = -Infinity;
    for (let k = 0; k < 4; k++) {
      const f = hyp(decoys[k].re[g], decoys[k].im[g]);
      fs += f; if (f < fmin) fmin = f; if (f > fmax) fmax = f;
    }
    noise[g] = 0.5 * (fs - fmin - fmax);          // median of four
    wsum += wc[g];
  }
  let tr = 0, ti = 0;
  for (let g = 0; g < gn; g++) { tr += A.re[g] * wc[g] / wsum; ti += A.im[g] * wc[g] / wsum; }
  const phi = Math.atan2(ti, tr), cph = Math.cos(-phi), sph = Math.sin(-phi);
  const proj = new Float64Array(gn);
  let sign = 0, ndet = 0;
  const detVals = [];
  for (let g = 0; g < gn; g++) {
    proj[g] = A.re[g] * cph - A.im[g] * sph;
    if (proj[g] > 0) sign += wc[g] / wsum;
    if (mag[g] > 3 * Math.max(noise[g], 1e-9)) { ndet++; detVals.push(mag[g]); }
  }
  const ampGlobal = 2 * hyp(tr, ti);
  const nz = 4;
  const ye = [], xe = [];
  for (let k = 0; k <= nz; k++) { ye.push(Math.floor(gh * k / nz)); xe.push(Math.floor(gw * k / nz)); }
  const Pz = [], SE = [];
  for (let j = 0; j < nz; j++) for (let i = 0; i < nz; i++) {
    let sw = 0, cnt = 0;
    for (let y = ye[j]; y < ye[j + 1]; y++) for (let xx = xe[i]; xx < xe[i + 1]; xx++) { sw += wc[y * gw + xx]; cnt++; }
    if (cnt === 0 || sw <= 0) continue;
    let p = 0;
    const dr = [0, 0, 0, 0], di = [0, 0, 0, 0];
    for (let y = ye[j]; y < ye[j + 1]; y++) for (let xx = xe[i]; xx < xe[i + 1]; xx++) {
      const g = y * gw + xx, wz = wc[g] / sw;
      p += proj[g] * wz;
      for (let k = 0; k < 4; k++) { dr[k] += decoys[k].re[g] * wz; di[k] += decoys[k].im[g] * wz; }
    }
    let ms = 0;
    for (let k = 0; k < 4; k++) ms += dr[k] * dr[k] + di[k] * di[k];
    Pz.push(p); SE.push(Math.sqrt(ms / 4 / 2));
  }
  let ref = 0;
  if (Pz.length) {
    const sp = Float64Array.from(Pz).sort();
    const top = sp.slice(Math.max(0, sp.length - 3));
    ref = top.reduce((a, b) => a + b, 0) / top.length;
  }
  let present = 0, absent = 0;
  if (ref > 0) {
    for (let k = 0; k < Pz.length; k++) {
      if (Pz[k] > 0.4 * ref && Pz[k] > 2 * SE[k]) present++;
      else if (Pz[k] + 2 * SE[k] < 0.4 * ref) absent++;
    }
  }
  const voted = present + absent;
  const uniformity = voted ? present / voted : null;
  const isGlobal = uniformity === null ? sign >= CFG.GLOBAL_SIGN
    : (uniformity >= 0.8 && (voted >= 3 || sign >= 0.7));
  let magMax = 0;
  for (let g = 0; g < gn; g++) if (mag[g] > magMax) magMax = mag[g];
  const ampLocal = ndet >= 4 ? 2 * percentile(detVals, 90) : 2 * Math.max(ref, 0);
  const env = new Float64Array(gn);
  for (let g = 0; g < gn; g++) env[g] = Math.max(mag[g] - 1.5 * noise[g], 0);
  return { sign_consistency: sign, uniformity, zones_present: present, zones_absent: absent,
           is_global: isGlobal, coverage: ndet / gn, amp_global: ampGlobal, amp_local: ampLocal,
           envelope: env, gw, gh, cell, box: dm.box };
}

/* Isolate the pattern (sum of waves) inside a rectangle, in the units of x. */
function reconstruct(x, w, h, peaksUV, y0, x0, hgt, wid) {
  const out = new Float64Array(hgt * wid);
  for (const [u, v] of peaksUV) {
    const period = 1 / Math.max(hyp(u, v), 1e-9);
    const box = boxFor(period, h, w);
    const pad = 2 * box;
    const ya = Math.max(0, y0 - pad), yb = Math.min(h, y0 + hgt + pad);
    const xa = Math.max(0, x0 - pad), xb = Math.min(w, x0 + wid + pad);
    const sw = xb - xa, sh = yb - ya, sn = sw * sh;
    const sub = new Float64Array(sn);
    for (let y = 0; y < sh; y++) sub.set(x.subarray((y + ya) * w + xa, (y + ya) * w + xb), y * sw);
    const lp = smooth3(sub, sw, sh, box);
    const cosx = new Float64Array(sw), sinx = new Float64Array(sw);
    for (let i = 0; i < sw; i++) {
      cosx[i] = Math.cos(2 * Math.PI * u * (i + xa)); sinx[i] = Math.sin(2 * Math.PI * u * (i + xa));
    }
    let zr = new Float64Array(sn), zi = new Float64Array(sn);
    const cr = new Float64Array(sn), ci = new Float64Array(sn);
    for (let y = 0; y < sh; y++) {
      const cyv = Math.cos(2 * Math.PI * v * (y + ya)), syv = Math.sin(2 * Math.PI * v * (y + ya));
      for (let i = 0; i < sw; i++) {
        const k = y * sw + i;
        const c = cosx[i] * cyv - sinx[i] * syv, s = sinx[i] * cyv + cosx[i] * syv;   // carrier
        cr[k] = c; ci[k] = s;
        const hp = sub[k] - lp[k];
        zr[k] = hp * c; zi[k] = -hp * s;                                              // hp * conj(car)
      }
    }
    zr = smooth3(zr, sw, sh, box); zi = smooth3(zi, sw, sh, box);
    for (let y = 0; y < hgt; y++) for (let i = 0; i < wid; i++) {
      const k = (y + y0 - ya) * sw + (i + x0 - xa);
      out[y * wid + i] += 2 * (zr[k] * cr[k] - zi[k] * ci[k]);
    }
  }
  return out;
}

// ---------------------------------------------------------------- JPEG block test
function blockBoundaryTest(y, w, h, weight) {
  const res = {};
  for (const name of ['x', 'y']) {
    // direction "x": steps between horizontally adjacent pixels, averaged down columns
    const L = name === 'x' ? w : h, N = name === 'x' ? h : w;
    const get = name === 'x' ? (r, c) => y[r * w + c] : (r, c) => y[c * w + r];
    const wget = name === 'x' ? (r, c) => (weight ? weight[r * w + c] : 1)
                              : (r, c) => (weight ? weight[c * w + r] : 1);
    const ncol = L - 3;
    if (ncol < 64) { res[name] = [0, 0]; continue; }
    const num = new Float64Array(ncol), den = new Float64Array(ncol);
    for (let r = 0; r < N; r++) {
      let d0 = Math.abs(get(r, 1) - get(r, 0)), d1 = Math.abs(get(r, 2) - get(r, 1));
      for (let j = 0; j < ncol; j++) {
        const d2 = Math.abs(get(r, j + 3) - get(r, j + 2));
        const s = d1 - 0.5 * (d0 + d2);
        const wv = wget(r, j + 1);
        num[j] += s * wv; den[j] += wv;
        d0 = d1; d1 = d2;
      }
    }
    const n = Math.floor(ncol / 8) * 8;
    if (n < 64) { res[name] = [0, 0]; continue; }
    const rows = n / 8;
    const m = new Float64Array(8), sd = new Float64Array(8);
    for (let k = 0; k < 8; k++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += num[r * 8 + k] / Math.max(den[r * 8 + k], 1e-9);
      m[k] = sum / rows;
      let ss = 0;
      for (let r = 0; r < rows; r++) {
        const dv = num[r * 8 + k] / Math.max(den[r * 8 + k], 1e-9) - m[k];
        ss += dv * dv;
      }
      sd[k] = Math.sqrt(ss / (rows - 1));
    }
    let k = 0;
    for (let i = 1; i < 8; i++) if (m[i] > m[k]) k = i;
    let osum = 0, omax = -Infinity;
    for (let i = 0; i < 8; i++) if (i !== k) { osum += m[i]; if (m[i] > omax) omax = m[i]; }
    const se = (sd.reduce((a, b) => a + b, 0) / 8) / Math.sqrt(rows);
    let z = (m[k] - osum / 7) / Math.max(se, 1e-9);
    z = Math.min(z, (m[k] - omax) / Math.max(se, 1e-9) * 1.5);
    res[name] = [z, (k + 2) % 8];
  }
  return { z_x: res.x[0], z_y: res.y[0], offset_x: res.x[1], offset_y: res.y[1],
           detected: res.x[0] > 6 && res.y[0] > 6 };
}

// ---------------------------------------------------------------- grouping
function near(du, dv, w, h, tol) {
  if (tol === undefined) tol = 2.0;
  return Math.abs(wrap(du)) * w <= tol && Math.abs(wrap(dv)) * h <= tol;
}
function onLattice(du, dv, w, h, step) {
  return Math.abs(wrap(du * step) / step) * w <= 2.0 && Math.abs(wrap(dv * step) / step) * h <= 2.0;
}
const HARMONICS = [];
for (let n = 1; n <= 20; n++) { HARMONICS.push(n); HARMONICS.push(-n); }

function derivedFrom(p, bases, w, h, latticeStep) {
  for (let i = 0; i < bases.length; i++) {
    const b = bases[i];
    for (const n of HARMONICS) {
      const du = p.u - n * b.u, dv = p.v - n * b.v;
      if (near(du, dv, w, h, Math.min(2.0 + 0.75 * (Math.abs(n) - 1), 5.0))) return i;
      if (latticeStep && Math.abs(n) <= 3 && onLattice(du, dv, w, h, latticeStep)) return i;
    }
  }
  const nb = Math.min(bases.length, 4);
  for (let i = 0; i < nb; i++) for (let j = i + 1; j < nb; j++) {
    for (const m of [1, -1, 2, -2]) for (const n of [1, -1, 2, -2]) {
      const du = p.u - m * bases[i].u - n * bases[j].u;
      const dv = p.v - m * bases[i].v - n * bases[j].v;
      if (near(du, dv, w, h) || (latticeStep && Math.abs(m) + Math.abs(n) <= 2 && onLattice(du, dv, w, h, latticeStep))) return i;
    }
  }
  return null;
}

function mergeChannels(peaks, w, h) {
  const merged = [];
  const sorted = peaks.slice().sort((a, b) => b.score - a.score);
  for (const p of sorted) {
    let hit = null;
    for (const m of merged) {
      if (m.view === p.view && Math.abs(wrap(m.u - p.u)) * w <= 1.5 && Math.abs(wrap(m.v - p.v)) * h <= 1.5) { hit = m; break; }
    }
    if (hit) hit.channels[p.channel] = Math.max(hit.channels[p.channel] || 0, p.score);
    else { p.channels = { [p.channel]: p.score }; merged.push(p); }
  }
  return merged;
}

function bestChannel(p) {
  let best = null;
  for (const c of Object.keys(p.channels)) if (best === null || p.channels[c] > p.channels[best]) best = c;
  return best;
}

// ---------------------------------------------------------------- main
function analyzePixels(rgba, W, H, info, opts, progress) {
  opts = opts || {};
  const L = loadPixels(rgba, W, H, info, opts.maxSide || 2048);
  return analyzeChannels(L, info, opts, progress);
}

/* The analysis proper. L = {chans, rgb, w, h}. opts.res (optional) holds a stacked
   noise residual per channel, which is scanned as one more source. */
function analyzeChannels(L, info, opts, progress) {
  opts = opts || {};
  progress = progress || (() => {});
  const { chans, rgb, w, h } = L;
  if (Math.min(h, w) < 64) throw new Error('Image is too small to analyse (needs at least 64x64 pixels).');

  const sources = {}, weights = {}, spectra = {}, thresholds = {};
  let allPeaks = [];
  const names = Object.keys(chans);
  let step = 0;
  const total = names.length * (opts.res ? 6 : 4);
  if (opts.res) {
    for (const kind of ['res', 'resw']) for (const name of names) {
      if (!opts.res[kind] || !opts.res[kind][name]) continue;
      progress(`Scanning frequencies (${++step} of ${total})`, 0.05 + 0.6 * step / total);
      sources[kind + ':' + name] = opts.res[kind][name];
      const { E, P } = whitenedSpectrum(opts.res[kind][name], w, h, null);
      const fp = findPeaks(E, P, w, h, 'A', name);
      for (const p of fp.peaks) { p.weighted = false; p.src = kind; }
      thresholds['A:' + name + ':' + kind] = fp.thr;
      allPeaks = allPeaks.concat(fp.peaks);
      if (name === 'Y' && kind === 'resw') spectra.A = E;
    }
  }
  for (const name of names) {
    const x = chans[name];
    const wtFine = flatnessWeight(x, w, h, 3), wtCoarse = flatnessWeight(x, w, h, 15);
    weights['A:' + name] = wtCoarse; weights['B:' + name] = wtFine;
    sources['val:' + name] = x;
    sources['lap:' + name] = roughnessLap(x, w, h);
    sources['step:' + name] = roughnessStep(x, w, h);
    for (const [view, kind, wt] of [['A', 'val', null], ['A', 'val', wtCoarse], ['B', 'lap', wtFine], ['B', 'step', wtFine]]) {
      progress(`Scanning frequencies (${++step} of ${total})`, 0.05 + 0.6 * step / total);
      const { E, P } = whitenedSpectrum(sources[kind + ':' + name], w, h, wt);
      const fp = findPeaks(E, P, w, h, view, name);
      for (const p of fp.peaks) { p.weighted = wt !== null; p.src = kind; }
      thresholds[view + ':' + name + ':' + kind + (wt ? 'w' : '')] = fp.thr;
      allPeaks = allPeaks.concat(fp.peaks);
      if (name === 'Y' && (kind === 'val' || kind === 'lap') && wt !== null && !(view === 'A' && opts.res)) spectra[view] = E;
    }
  }
  progress('Measuring patterns', 0.7);

  const peaks = mergeChannels(allPeaks, w, h);
  for (const p of peaks) p.lattice = latticeOf(p.u, p.v, w, h);

  const measurePeak = (p) => {
    const c = bestChannel(p);
    const ms = measure(sources[p.src + ':' + c], w, h, p.u, p.v, p.weighted ? weights[p.view + ':' + c] : null);
    if (p.src === 'res' || p.src === 'resw') {   // undo the residual's high-pass: strength in real levels
      const g = Math.max(0.1, residualGain(p.u, p.v, p.src === 'res' ? RES_BOX : RES_BOX_WIDE));
      ms.amp_global /= g; ms.amp_local /= g;
    }
    return ms;
  };

  const cache = new Map();
  const coincidental = [];
  const latA = peaks.filter(q => q.lattice && q.view === 'A').sort((a, b) => b.score - a.score).slice(0, 5);
  for (const p of latA) {
    const ms = measurePeak(p);
    cache.set(p, ms);
    if (p.lattice >= 4 && !ms.is_global && ms.zones_absent >= 3 && ms.amp_local >= 2.0) {
      p.lattice = 0; coincidental.push(p);
    }
  }
  for (const p of peaks) {
    if (!p.lattice) continue;
    let hit = false;
    for (const b of coincidental) {
      for (const n of [2, -2, 3, -3, 4, -4]) if (near(p.u - n * b.u, p.v - n * b.v, w, h)) hit = true;
    }
    if (hit) p.lattice = 0;
  }

  const jpeg = blockBoundaryTest(chans.Y, w, h, weights['B:Y']);
  const jpegEvidence = info.is_jpeg || jpeg.detected;
  const has8 = peaks.some(p => p.lattice === 8 || p.lattice === 16);
  const latticeStep = (jpegEvidence || has8) ? 8 : 0;

  if (peaks.some(p => p.lattice === 2)) {
    for (const p of peaks) {
      if (p.lattice) continue;
      const du = (0.5 - Math.abs(wrap(p.u))) * w, dv = (0.5 - Math.abs(wrap(p.v))) * h;
      const nearU = du <= 6 && (Math.abs(wrap(p.v)) * h <= 6 || dv <= 6);
      const nearV = dv <= 6 && (Math.abs(wrap(p.u)) * w <= 6 || du <= 6);
      if (nearU || nearV) p.lattice = 2;
    }
  }

  const groups = {}, patterns = [];
  const addLattice = (cat, p) => {
    if (!groups[cat]) {
      groups[cat] = { category: cat, peaks: [], strength: 0, channel: 'Y', view: p.view, measure: null, notes: [], weighted: false, rep: null, isLatticeGroup: true };
      patterns.push(groups[cat]);
    }
    groups[cat].peaks.push(p);
  };
  const free = [];
  for (const p of peaks) {
    if (p.lattice === 2 || p.lattice === 4) addLattice(jpegEvidence ? 'jpeg_grid' : (has8 ? 'grid8' : 'pixel_lattice'), p);
    else if (p.lattice === 8) addLattice(jpegEvidence ? 'jpeg_grid' : 'grid8', p);
    else if (p.lattice === 16) {
      if (jpegEvidence && !('Y' in p.channels)) addLattice('chroma_grid', p);
      else addLattice(jpegEvidence ? 'jpeg_grid' : 'grid8', p);
    } else free.push(p);
  }

  const freeA = free.filter(p => p.view === 'A').sort((a, b) => b.score - a.score);
  const freeB = free.filter(p => p.view === 'B').sort((a, b) => b.score - a.score);
  const bases = [], members = [];
  for (const p of freeA) {
    const i = derivedFrom(p, bases, w, h, latticeStep);
    if (i === null) { bases.push(p); members.push([p]); } else members[i].push(p);
  }
  const bBases = [], bMembers = [];
  for (const p of freeB) {
    const i = derivedFrom(p, bases, w, h, latticeStep);
    if (i !== null) { members[i].push(p); continue; }
    const j = derivedFrom(p, bBases, w, h, latticeStep);
    if (j === null) { bBases.push(p); bMembers.push([p]); } else bMembers[j].push(p);
  }
  for (let i = 0; i < bases.length; i++) {
    patterns.push({ category: '?', peaks: members[i], strength: bases[i].score, channel: bestChannel(bases[i]), view: 'A', measure: null, notes: [], weighted: false, rep: bases[i] });
  }
  for (let i = 0; i < bBases.length; i++) {
    patterns.push({ category: 'resampling', peaks: bMembers[i], strength: bBases[i].score, channel: bestChannel(bBases[i]), view: 'B', measure: null, notes: [], weighted: false, rep: bBases[i] });
  }
  for (const pat of patterns) {
    if (pat.isLatticeGroup) {
      let rep = pat.peaks[0];
      for (const q of pat.peaks) {
        const a = (q.view === 'B') ? 1 : 0, b = (rep.view === 'B') ? 1 : 0;
        if (a > b || (a === b && q.score > rep.score)) rep = q;
      }
      pat.rep = rep;
      pat.strength = Math.max(...pat.peaks.map(q => q.score));
      pat.channel = bestChannel(rep);
      pat.view = rep.view;
    }
  }
  patterns.sort((a, b) => b.strength - a.strength);

  const fin = [];
  const droppedInfo = [];
  let dropped = 0;
  for (let nMeasured = 0; nMeasured < patterns.length; nMeasured++) {
    if (fin.length >= 12 || nMeasured >= 24) break;
    const pat = patterns[nMeasured];
    progress(`Measuring patterns (${nMeasured + 1})`, 0.7 + 0.15 * Math.min(1, nMeasured / 12));
    const m = pat.rep;
    const ms = cache.get(m) || measurePeak(m);
    pat.measure = ms;
    pat.weighted = !!m.weighted;
    const onAxis = Math.abs(wrap(m.u)) * w <= 1.5 || Math.abs(wrap(m.v)) * h <= 1.5;
    if (pat.category === '?') {
      const hasBTwin = pat.peaks.some(q => q.view === 'B' && near(q.u - m.u, q.v - m.v, w, h));
      if (ms.is_global) {
        ms.amplitude = ms.amp_global;
        if (hasBTwin && ms.amplitude < 1.0) pat.category = 'resampling';
        else pat.category = ms.amplitude < 8.0 ? 'global_faint' : 'global_visible';
      } else {
        ms.amplitude = ms.amp_local;
        if (ms.amplitude < CFG.MIN_REGIONAL_AMPLITUDE) { dropped++; droppedInfo.push({ period: peakPeriod(m), angle: peakAngle(m), src: m.src, score: m.score, amp: ms.amplitude, zones: [ms.zones_present, ms.zones_absent], global: ms.is_global, cat: pat.category }); continue; }
        pat.category = 'regional';
      }
      if (pat.category === 'global_faint' && ms.amplitude < CFG.TRACE_AMPLITUDE) pat.category = 'trace';
      if (pat.category === 'global_faint' && onAxis) {
        pat.notes.push('It runs exactly horizontally or vertically, which favours a processing cause (a sensor or compression lattice that was later resized, scan lines, banding) over a deliberately placed pattern.');
      }
    } else {
      ms.amplitude = m.view === 'A' ? ms.amp_global : null;
      if ((pat.category === 'pixel_lattice' || pat.category === 'grid8') && ms.amplitude !== null
          && pat.peaks.every(q => q.view === 'A') && ms.amplitude < CFG.TRACE_AMPLITUDE) pat.category = 'trace';
    }
    if ((pat.category === 'pixel_lattice' || pat.category === 'grid8') && !ms.is_global && ms.zones_absent >= 3) { dropped++; droppedInfo.push({ period: peakPeriod(m), angle: peakAngle(m), src: m.src, score: m.score, amp: ms.amplitude, zones: [ms.zones_present, ms.zones_absent], global: ms.is_global, cat: pat.category }); continue; }
    if (pat.category === 'resampling' && pat.view === 'B' && !ms.is_global) { dropped++; droppedInfo.push({ period: peakPeriod(m), angle: peakAngle(m), src: m.src, score: m.score, amp: ms.amplitude, zones: [ms.zones_present, ms.zones_absent], global: ms.is_global, cat: pat.category }); continue; }
    if (pat.category === 'resampling') {
      const f = peakFreq(m);
      if (onAxis && f > 0.02 && f < 0.5) {
        pat.notes.push(`The rhythm repeats every ${(1 / f).toFixed(2)} px. Enlarging by about ${(1 / (1 - f)).toFixed(2)}x or ${(1 / f).toFixed(2)}x would do this; so would an 8- or 16-px compression grid shrunk to ${(100 / (8 * f)).toFixed(0)}% or ${(100 / (16 * f)).toFixed(0)}% of its size.`);
      } else if (!onAxis) {
        pat.notes.push('It is not aligned with the pixel rows or columns, which points to rotation or skew rather than a plain resize.');
      }
    }
    fin.push(pat);
  }
  let thrMax = 0;
  for (const k of Object.keys(thresholds)) thrMax = Math.max(thrMax, thresholds[k]);
  return { info, rgb, w, h, chans, sources, weights, spectra, patterns: fin, jpeg,
           thr: thrMax, n_scans: total, n_peaks: peaks.length, dropped, droppedInfo };
}

// ---------------------------------------------------------------- stacking several images
const RES_BOX = 5;                     // fine residual   = image - smooth3(image, 5)
const RES_BOX_WIDE = 31;               // wide residual   = image - smooth3(image, 31)

/* How much of a wave survives a residual's high-pass (1 = all of it). */
function residualGain(u, v, box) {
  const d = (f) => {
    const a = Math.PI * f;
    return Math.abs(Math.sin(a)) < 1e-9 ? 1 : Math.sin(a * box) / (box * Math.sin(a));
  };
  return 1 - Math.pow(d(u), 3) * Math.pow(d(v), 3);
}

/* Flat-image averaging and noise-residual stacking.
   mean      = plain average of the images: each picture's own content shrinks, anything
               identical in every image stays at full strength;
   residual  = each image minus a smoothed copy of itself, averaged with more weight where
               that image is smooth (inverse local variance, as for sensor-noise
               fingerprints). Two versions: a fine one for narrow stripes and a wide one
               that keeps slow ripples too. */
function stackBegin(w, h) {
  const n = w * h, z = () => new Float64Array(n), trio = () => ({ Y: z(), Cb: z(), Cr: z() });
  return { w, h, n: 0, colour: false, rgb: new Float64Array(n * 3), sum: trio(),
           num: trio(), den: trio(), numw: trio(), denw: trio() };
}

function stackAdd(st, rgba, W, H) {
  if (W !== st.w || H !== st.h) throw new Error('Stacked images must all be the same size.');
  const L = loadPixels(rgba, W, H, { notes: [] }, Math.max(W, H));
  const n = st.w * st.h;
  for (let i = 0; i < n * 3; i++) st.rgb[i] += L.rgb[i];
  if (L.chans.Cb) st.colour = true;
  for (const name of ['Y', 'Cb', 'Cr']) {
    const x = L.chans[name];
    if (!x) { for (let i = 0; i < n; i++) st.sum[name][i] += 128; continue; }
    const sum = st.sum[name];
    for (let i = 0; i < n; i++) sum[i] += x[i];
    for (const [box, scale, num, den] of [[RES_BOX, 3, st.num[name], st.den[name]],
                                          [RES_BOX_WIDE, 15, st.numw[name], st.denw[name]]]) {
      const sm = smooth3(x, st.w, st.h, box), wt = flatnessWeight(x, st.w, st.h, scale);
      for (let i = 0; i < n; i++) { num[i] += wt[i] * (x[i] - sm[i]); den[i] += wt[i]; }
    }
  }
  st.n++;
  return L;
}

function stackFinish(st) {
  const n = st.w * st.h, names = st.colour ? ['Y', 'Cb', 'Cr'] : ['Y'];
  const chans = {}, res = { res: {}, resw: {} };
  for (const name of names) {
    const m = new Float64Array(n), r = new Float64Array(n), rw = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      m[i] = st.sum[name][i] / st.n;
      r[i] = st.den[name][i] > 0 ? st.num[name][i] / st.den[name][i] : 0;
      rw[i] = st.denw[name][i] > 0 ? st.numw[name][i] / st.denw[name][i] : 0;
    }
    chans[name] = m; res.res[name] = r; res.resw[name] = rw;
  }
  const rgb = new Float64Array(n * 3);
  for (let i = 0; i < n * 3; i++) rgb[i] = st.rgb[i] / st.n;
  return { L: { chans, rgb, w: st.w, h: st.h }, res };
}

/* One image's amplitude and phase for one wave, over the whole frame. */
function projectGlobal(src, w, h, u, v, weight) {
  const box = boxFor(1 / Math.max(hyp(u, v), 1e-9), h, w);
  const lp = boxFilter(src, w, h, box);
  const cosx = new Float64Array(w), sinx = new Float64Array(w);
  for (let x = 0; x < w; x++) { cosx[x] = Math.cos(2 * Math.PI * u * x); sinx[x] = Math.sin(2 * Math.PI * u * x); }
  // Tapered to zero at the borders: a hard frame edge leaks slow content into every
  // frequency with the SAME phase in every image, which would fake agreement.
  const hx = hanning(w), hy = hanning(h);
  let re = 0, im = 0, sw = 0;
  for (let y = 0; y < h; y++) {
    const cy = Math.cos(2 * Math.PI * v * y), sy = Math.sin(2 * Math.PI * v * y), o = y * w;
    let rr = 0, ri = 0;
    for (let x = 0; x < w; x++) {
      const wt = (weight ? weight[o + x] : 1) * hx[x] * hy[y], val = (src[o + x] - lp[o + x]) * wt;
      rr += val * cosx[x]; ri -= val * sinx[x]; sw += wt;
    }
    re += rr * cy + ri * sy; im += ri * cy - rr * sy;      // (rr + i ri) * (cy - i sy)
  }
  return [re / sw, im / sw];
}

/* Second pass: is each pattern there, with the same alignment, in every single image? */
function coherenceAdd(patterns, L, acc) {
  const cache = {};
  const get = (kind, c) => {
    const k = kind + ':' + c;
    if (!cache[k]) {
      const x = L.chans[c];
      cache[k] = kind === 'lap' ? roughnessLap(x, L.w, L.h) : kind === 'step' ? roughnessStep(x, L.w, L.h)
        : kind === 'wA' ? flatnessWeight(x, L.w, L.h, 15) : kind === 'wB' ? flatnessWeight(x, L.w, L.h, 3) : x;
    }
    return cache[k];
  };
  patterns.forEach((pat, i) => {
    const m = pat.rep, c = L.chans[pat.channel] ? pat.channel : 'Y';
    const kind = (m.src === 'lap' || m.src === 'step') ? m.src : 'val';
    const a = projectGlobal(get(kind, c), L.w, L.h, m.u, m.v, get(m.view === 'B' ? 'wB' : 'wA', c));
    (acc[i] = acc[i] || []).push(a);
  });
}

function coherenceFinish(patterns, acc) {
  patterns.forEach((pat, i) => {
    const list = acc[i] || [], N = list.length;
    let ur = 0, ui = 0;
    const mags = [];
    for (const [re, im] of list) { const m = hyp(re, im) || 1e-12; ur += re / m; ui += im / m; mags.push(m); }
    const C = N ? hyp(ur, ui) / N : 0, phi = Math.atan2(ui, ur);
    let aligned = 0;
    for (const [re, im] of list) {
      const m = hyp(re, im) || 1e-12;
      if ((re * Math.cos(phi) + im * Math.sin(phi)) / m > 0.5) aligned++;   // within 60 degrees
    }
    const judged = N >= 5;
    pat.across = { n: N, coherence: C, chance: N ? 1 / Math.sqrt(N) : 1, aligned,
                   judged, shared: judged && N * C * C > 4.6 && C >= 0.5 && aligned >= 0.6 * N,
                   typical_amp: N ? 2 * median(mags) : 0 };
    // a sub-level ripple that every image carries is a finding, not a stray residue
    if (pat.category === 'trace' && pat.across.shared) pat.category = 'global_faint';
  });
}

const PatternCore = {
  analyzeChannels, stackBegin, stackAdd, stackFinish, coherenceAdd, coherenceFinish, residualGain, projectGlobal, CFG, CATEGORIES, EXPLAIN, LN2, analyzePixels, parseFileHeader, reconstruct,
  peakPeriod, peakAngle, peakFreq, percentile, median, boxFilter, smooth3, whitenedSpectrum,
  findPeaks, measure, blockBoundaryTest, latticeOf, flatnessWeight, roughnessLap, roughnessStep };
if (typeof module !== 'undefined' && module.exports) module.exports = PatternCore;
if (typeof self !== 'undefined') self.PatternCore = PatternCore;
