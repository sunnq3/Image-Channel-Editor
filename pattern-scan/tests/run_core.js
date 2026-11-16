// Node harness: run the browser analysis core on raw pixels.
//   node tests/run_core.js manifest.json  -> prints JSON results
const fs = require('fs');
const path = require('path');
const core = require(path.join(__dirname, '..', 'web', 'core.js'));
const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const c of manifest) {
  const rgba = new Uint8ClampedArray(fs.readFileSync(c.rgba));
  const bytes = new Uint8Array(fs.readFileSync(c.file));
  const info = core.parseFileHeader(bytes, c.name);
  const t0 = Date.now();
  const res = core.analyzePixels(rgba, c.width, c.height, info, { maxSide: c.maxSide || 2048 });
  out.push({
    name: c.name, ms: Date.now() - t0, w: res.w, h: res.h, jpeg: res.jpeg,
    info: { format: info.format, is_jpeg: info.is_jpeg, q: info.jpeg_quality, ss: info.jpeg_subsampling },
    n_peaks: res.n_peaks, dropped: res.dropped,
    patterns: res.patterns.map(p => ({
      category: p.category, strength: p.strength,
      period: core.peakPeriod(p.rep), angle: core.peakAngle(p.rep),
      amplitude: p.measure.amplitude, is_global: p.measure.is_global,
      zones: [p.measure.zones_present, p.measure.zones_absent],
      peaks: p.peaks.map(q => [q.u, q.v, q.view, q.score]),
    })),
  });
}
process.stdout.write(JSON.stringify(out));
