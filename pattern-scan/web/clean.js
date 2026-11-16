/* Pattern removal: takes the carrier waves the scan found (or ones typed in) and subtracts
   them from every image at full resolution.

   The patterns this targets are whole-frame ripples far below one brightness level. One
   image on its own cannot pin down a ripple that faint pixel by pixel, so removal is a
   measurement first and a subtraction second:

   1. measure  Each image is cut into tiles and every carrier is fitted in every tile
               (Hann-weighted least squares, clipped pixels ignored). Busy tiles get less
               weight. Only these small tables are kept, not the pixels.
   2. solve    A frequency read from a crop or a stack is close but not exact, and over a
               4000-pixel row a tiny error becomes a large phase drift. The exact frequency
               is found by searching for the one at which all tiles of all images line up
               best. Then each image's amplitude and phase are the weighted average of its
               tiles. When the images agree on the alignment (the usual case for a
               watermark) one shared estimate from all of them is used, which is far less
               noisy than any single image's.
   3. apply    The fitted ripple is subtracted (in brightness by changing R, G, B equally,
               or in a colour channel through the JPEG colour transform). Removing 0.05 of a
               level and rounding to 8 bits would change nothing, so pixels are quantised
               with dither, floor(x + U[0,1)), which is unbiased: on average every pixel
               lands on the corrected value and the ripple's statistics are cancelled.
               Clipped pixels (0 or 255) are left exactly as they were.

   API (all plain data, so it runs in a worker):
     measure(rgba, w, h, carriers)            -> m        one per image
     solve(ms, carriers, opts)                -> plan     once
     apply(rgba, w, h, plan, index, opts)     -> { rgba, changed }
     carriers: [{ u, v, channel: 'Y'|'Cb'|'Cr', levels }]   u, v in cycles per pixel
*/
(function () {
  'use strict';
  const TAU = 2 * Math.PI;

  // ITU-R BT.601 full range, as JPEG uses
  const PLANE = {
    Y:  [0.299, 0.587, 0.114],
    Cb: [-0.168736, -0.331264, 0.5],
    Cr: [0.5, -0.418688, -0.081312],
  };
  // what a +1 change in each plane does to R, G, B
  const BACK = { Y: [1, 1, 1], Cb: [0, -0.344136, 1.772], Cr: [1.402, -0.714136, 0] };

  function xorshift(seed) {
    let s = (seed >>> 0) || 0x9e3779b9;
    return function () {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  function validMask(rgba, n) {
    const valid = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
      valid[i] = (rgba[i * 4 + 3] === 255 && r > 0 && r < 255 && g > 0 && g < 255 && b > 0 && b < 255) ? 1 : 0;
    }
    return valid;
  }

  function tileFor(c, w, h) {
    const period = 1 / Math.max(Math.hypot(c.u, c.v), 1e-6);
    // near the 2-pixel limit a wave and its mirror differ only by a slow beat; the tile
    // must hold a few beats or the cos/sin fit cannot tell them apart and gets noisy
    const beat = Math.hypot(2 * c.u - Math.round(2 * c.u), 2 * c.v - Math.round(2 * c.v));
    const need = Math.max(64, 8 * period, beat > 0 ? 4 / beat : 256);
    const T = Math.ceil(need / 16) * 16;
    return Math.max(16, Math.min(T, 256, w, h));
  }

  /* One carrier, every tile: complex coefficient (pattern = Re(c e^{i phase})) and weight. */
  function fitTiles(P, valid, w, h, c, T) {
    const nx = Math.ceil(w / T), ny = Math.ceil(h / T), nt = nx * ny;
    const re = new Float64Array(nt), im = new Float64Array(nt), wt = new Float64Array(nt);
    const xc = new Float64Array(nt), yc = new Float64Array(nt);
    const win = new Float64Array(T);
    for (let i = 0; i < T; i++) win[i] = Math.sin(Math.PI * (i + 0.5) / T) ** 2;
    const cr = Math.cos(TAU * c.u), sr = Math.sin(TAU * c.u);
    for (let ty = 0; ty < ny; ty++) {
      const y0 = Math.min(ty * T, h - T);
      for (let tx = 0; tx < nx; tx++) {
        const x0 = Math.min(tx * T, w - T), t = ty * nx + tx;
        xc[t] = x0 + T / 2 - 0.5; yc[t] = y0 + T / 2 - 0.5;
        let sw = 0, s1 = 0, s2 = 0;
        for (let dy = 0; dy < T; dy++) {
          const row = (y0 + dy) * w + x0, wy = win[dy];
          for (let dx = 0; dx < T; dx++) {
            if (!valid[row + dx]) continue;
            const ww = wy * win[dx], val = P[row + dx];
            sw += ww; s1 += ww * val; s2 += ww * val * val;
          }
        }
        if (sw < 0.0625 * T * T) continue;                       // mostly clipped
        const mean = s1 / sw, vr = Math.max(0, s2 / sw - mean * mean);
        let A = 0, B = 0, C = 0, X = 0, Z = 0;
        for (let dy = 0; dy < T; dy++) {
          const y = y0 + dy, row = y * w + x0, wy = win[dy];
          const ph = TAU * (c.u * x0 + c.v * y);
          let co = Math.cos(ph), si = Math.sin(ph);
          for (let dx = 0; dx < T; dx++) {
            if (valid[row + dx]) {
              const ww = wy * win[dx], val = (P[row + dx] - mean) * ww;
              A += ww * co * co; B += ww * co * si; C += ww * si * si;
              X += val * co; Z += val * si;
            }
            const c2 = co * cr - si * sr; si = si * cr + co * sr; co = c2;
          }
        }
        const det = A * C - B * B;
        if (det <= 1e-9 * (A * C + 1e-12)) continue;
        const a = (C * X - B * Z) / det, b = (A * Z - B * X) / det;
        re[t] = a; im[t] = -b;                                    // a cos + b sin = Re((a - ib) e^{i ph})
        wt[t] = sw / (vr + 0.05);                                 // quiet tiles count more
      }
    }
    return { re, im, wt, xc, yc, T, nx, ny };
  }

  /* Power of the tiles summed coherently in blocks of B x B tiles after shifting the
     frequency by (du, dv). Small blocks give a broad response (for the coarse search),
     the whole frame a sharp one (for the final digits). */
  function blockPower(f, du, dv, B) {
    const bx = Math.ceil(f.nx / B), by = Math.ceil(f.ny / B);
    const sr = new Float64Array(bx * by), si = new Float64Array(bx * by);
    for (let t = 0; t < f.wt.length; t++) {
      const g = f.wt[t];
      if (!g) continue;
      const ph = -TAU * (du * f.xc[t] + dv * f.yc[t]);
      const c = Math.cos(ph), s = Math.sin(ph);
      const b = Math.floor((t / f.nx | 0) / B) * bx + Math.floor((t % f.nx) / B);
      sr[b] += g * (f.re[t] * c - f.im[t] * s);
      si[b] += g * (f.re[t] * s + f.im[t] * c);
    }
    let p = 0;
    for (let b = 0; b < sr.length; b++) p += sr[b] * sr[b] + si[b] * si[b];
    return p;
  }

  function plane(rgba, n, ch) {
    const k = PLANE[ch], P = new Float32Array(n);
    for (let i = 0; i < n; i++) P[i] = k[0] * rgba[i * 4] + k[1] * rgba[i * 4 + 1] + k[2] * rgba[i * 4 + 2];
    return P;
  }

  function measure(rgba, w, h, carriers, progress) {
    const n = w * h, valid = validMask(rgba, n), planes = {}, fields = [];
    carriers.forEach((c, k) => {
      if (progress) progress(k / carriers.length);
      const ch = c.channel || 'Y';
      if (!planes[ch]) planes[ch] = plane(rgba, n, ch);
      fields.push(fitTiles(planes[ch], valid, w, h, c, tileFor(c, w, h)));
    });
    return { w, h, fields };
  }

  /* Coherent sum of one image's tiles after shifting the frequency by (du, dv). */
  function coherent(f, du, dv) {
    let sr = 0, si = 0, sw = 0;
    for (let t = 0; t < f.wt.length; t++) {
      const g = f.wt[t];
      if (!g) continue;
      const ph = -TAU * (du * f.xc[t] + dv * f.yc[t]);
      const c = Math.cos(ph), s = Math.sin(ph);
      sr += g * (f.re[t] * c - f.im[t] * s);
      si += g * (f.re[t] * s + f.im[t] * c);
      sw += g;
    }
    return [sr, si, sw];
  }

  /* Tiles whose fit is far above the typical tile are real texture near the frequency
     (a fence, a fabric), not the overlay: they would steer the estimate, so they are left out. */
  function dropTexture(f, cap) {
    const mags = [];
    for (let t = 0; t < f.wt.length; t++) if (f.wt[t]) mags.push(Math.hypot(f.re[t], f.im[t]));
    if (!mags.length) return 0;
    mags.sort((a, b) => a - b);
    const lim = Math.max(cap, 5 * mags[mags.length >> 1]);
    let dropped = 0;
    for (let t = 0; t < f.wt.length; t++) {
      if (f.wt[t] && Math.hypot(f.re[t], f.im[t]) > lim) { f.wt[t] = 0; dropped++; }
    }
    return dropped / mags.length;
  }

  /* Detection statistic at a frequency shift: sum over images of |coherent sum|^2 divided
     by its noise expectation. Pure noise gives about one per image. */
  function stat(ms, k, du, dv) {
    let q = 0;
    for (const m of ms) {
      const f = m.fields[k], r = coherent(f, du, dv);
      let v = 0;
      for (let t = 0; t < f.wt.length; t++) if (f.wt[t]) v += f.wt[t] * f.wt[t] * (f.re[t] * f.re[t] + f.im[t] * f.im[t]);
      if (v > 0) q += (r[0] * r[0] + r[1] * r[1]) / v;
    }
    return q;
  }

  function solve(ms, carriers, opts) {
    opts = opts || {};
    const plan = { carriers: [], amps: [], notes: [] };
    carriers.forEach((c, k) => {
      const T = ms[0].fields[k].T;
      const cap = Math.max(0.35, 4 * (c.levels || 0));
      let textureShare = 0;
      for (const m of ms) textureShare += dropTexture(m.fields[k], cap) / ms.length;
      // exact frequency: the shift at which every image's tiles add up most strongly
      // (each image may have its own phase, so images are summed in power)
      const J = (du, dv, B) => {
        let s = 0;
        for (const m of ms) s += blockPower(m.fields[k], du, dv, B);
        return s;
      };
      const big = Math.max(...ms.map(m => Math.max(m.fields[k].nx, m.fields[k].ny)));
      let bu = 0, bv = 0, span = opts.refine === false ? 0 : 0.5 / T;
      while (span > 0) {
        // block size whose response is about as wide as two grid steps
        const B = Math.max(1, Math.min(big, Math.round(3 / (span * T))));
        let best = -1, nu = bu, nv = bv;
        for (let i = -6; i <= 6; i++) for (let j = -6; j <= 6; j++) {
          const du = bu + span * i / 6, dv = bv + span * j / 6, v = J(du, dv, B);
          if (v > best) { best = v; nu = du; nv = dv; }
        }
        bu = nu; bv = nv; span /= 3;
        if (B >= big && span * T * big < 0.02) break;
      }
      // keep the corrected frequency only if it beats what noise alone reaches somewhere
      // in the searched area; otherwise trust the frequency given
      // (the search covers about big x big independent frequencies)
      const N = ms.length, lnM = Math.log(Math.max(2, big * big));
      const qBest = stat(ms, k, bu, bv), q0 = stat(ms, k, 0, 0);
      const limit = N + 3 * Math.sqrt(N * lnM) + 2 * lnM;
      let moved = true;
      if (!(qBest > limit && qBest > q0)) { bu = 0; bv = 0; moved = false; }
      // each image's amplitude and phase at that frequency, with its noise
      const per = ms.map(m => {
        const f = m.fields[k], r = coherent(f, bu, bv);
        if (!r[2]) return { z: [0, 0], w: 0, s2: 0 };
        const z = [r[0] / r[2], r[1] / r[2]];
        let s2 = 0;
        for (let t = 0; t < f.wt.length; t++) {
          if (!f.wt[t]) continue;
          const ph = -TAU * (bu * f.xc[t] + bv * f.yc[t]), co = Math.cos(ph), si = Math.sin(ph);
          const dr = f.re[t] * co - f.im[t] * si - z[0], di = f.re[t] * si + f.im[t] * co - z[1];
          s2 += f.wt[t] * f.wt[t] * (dr * dr + di * di);
        }
        return { z, w: r[2], s2: s2 / (r[2] * r[2]) };           // variance of z
      });
      let ar = 0, ai = 0, aw = 0, am = 0;
      for (const p of per) { ar += p.w * p.z[0]; ai += p.w * p.z[1]; aw += p.w; am += p.w * Math.hypot(p.z[0], p.z[1]); }
      const agreement = am ? Math.hypot(ar, ai) / am : 0;
      const shared = N >= 3 && agreement >= 0.5 && opts.perImage !== true;
      // shrink toward zero by how much of the estimate is noise, then cap
      const settle = (z, s2) => {
        const snr2 = s2 > 0 ? (z[0] * z[0] + z[1] * z[1]) / s2 : 1e9;
        const g = Math.max(0, 1 - 1 / snr2);
        let a = [z[0] * g, z[1] * g];
        const mm = Math.hypot(a[0], a[1]);
        if (mm > cap) a = [a[0] * cap / mm, a[1] * cap / mm];
        return { a, snr: Math.sqrt(snr2) };
      };
      let sharedVar = 0;
      for (const p of per) sharedVar += p.w * p.w * p.s2;
      sharedVar /= Math.max(1e-12, aw * aw);
      const sh = aw ? settle([ar / aw, ai / aw], sharedVar) : { a: [0, 0], snr: 0 };
      const own = per.map(p => settle(p.z, p.s2));
      const amps = shared ? per.map(() => sh.a) : own.map(o => o.a);
      const snr = shared ? sh.snr : own.reduce((s, o) => s + o.snr, 0) / Math.max(1, own.length);
      const uu = c.u + bu, vv = c.v + bv;
      plan.carriers.push({ u: uu, v: vv, channel: c.channel || 'Y', period: 1 / Math.hypot(uu, vv),
        angle: ((Math.atan2(vv, uu) * 180 / Math.PI) % 180 + 180) % 180,
        shift: Math.hypot(bu, bv), moved, shared, agreement, snr, textureShare,
        weak: snr < 3,
        levels: amps.reduce((s, z) => s + Math.hypot(z[0], z[1]), 0) / Math.max(1, amps.length) });
      amps.forEach((z, i) => { (plan.amps[i] = plan.amps[i] || [])[k] = z; });
    });
    return plan;
  }

  function apply(rgba, w, h, plan, index, opts) {
    opts = opts || {};
    const n = w * h, valid = validMask(rgba, n), out = new Uint8ClampedArray(rgba);
    const corr = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    plan.carriers.forEach((c, k) => {
      const z = (plan.amps[index] || [])[k];
      if (!z || (!z[0] && !z[1])) return;
      const back = BACK[c.channel], cr = Math.cos(TAU * c.u), sr = Math.sin(TAU * c.u);
      for (let y = 0; y < h; y++) {
        let co = 0, si = 0;
        for (let x = 0; x < w; x++) {
          if ((x & 127) === 0) { const ph = TAU * (c.u * x + c.v * y); co = Math.cos(ph); si = Math.sin(ph); }
          const d = z[0] * co - z[1] * si, i = y * w + x;
          corr[0][i] -= back[0] * d; corr[1][i] -= back[1] * d; corr[2][i] -= back[2] * d;
          const c2 = co * cr - si * sr; si = si * cr + co * sr; co = c2;
        }
      }
    });
    const rnd = xorshift(opts.seed || (index + 1) * 2654435761);
    let changed = 0;
    for (let i = 0; i < n; i++) {
      if (!valid[i]) continue;
      for (let c = 0; c < 3; c++) {
        const v = Math.floor(rgba[i * 4 + c] + corr[c][i] + rnd());
        const q = v < 0 ? 0 : v > 255 ? 255 : v;
        if (q !== rgba[i * 4 + c]) changed++;
        out[i * 4 + c] = q;
      }
    }
    return { rgba: out, changed: changed / (3 * n) };
  }

  /* A typed-in pattern: stripe width (px) and the scan's angle (degrees, direction of change). */
  function carrierFromPeriodAngle(period, angleDeg, channel, levels) {
    const f = 1 / period, a = angleDeg * Math.PI / 180;
    return { u: f * Math.cos(a), v: f * Math.sin(a), channel: channel || 'Y', levels: levels || 0 };
  }

  /* Carriers from the scan's patterns: every wave of every watermark-like pattern, in the
     channel it was found in. A wave and its mirror are the same carrier. */
  const REMOVABLE = ['global_faint', 'global_visible', 'trace'];
  function carriersFromPatterns(patterns, pick) {
    const out = [];
    patterns.forEach((p, i) => {
      if (!REMOVABLE.includes(p.category)) return;
      if (pick && !pick(p, i)) return;
      const levels = (p.measure && p.measure.amplitude) || 0;
      const waves = p.peaks.filter(q => q.view === 'A');
      if (!waves.length && p.rep) waves.push(p.rep);
      for (const q of waves) {
        const ch = p.channel || 'Y';
        const dup = out.some(c => c.channel === ch &&
          [1, -1].some(s => Math.abs(c.u - s * q.u) < 1e-3 && Math.abs(c.v - s * q.v) < 1e-3));
        if (!dup) out.push({ u: q.u, v: q.v, channel: ch, levels });
      }
    });
    return out;
  }

  const PatternClean = { measure, solve, apply, carrierFromPeriodAngle, carriersFromPatterns, REMOVABLE, fitTiles };
  if (typeof module !== 'undefined' && module.exports) module.exports = PatternClean;
  if (typeof self !== 'undefined') self.PatternClean = PatternClean;
})();
