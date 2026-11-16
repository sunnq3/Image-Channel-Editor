// Node harness for the multi-image (stacking) mode.
//   node tests/run_stack.js manifest.json  -> prints JSON results
const fs = require('fs');
const path = require('path');
const core = require(path.join(__dirname, '..', 'web', 'core.js'));
const sets = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const set of sets) {
  const t0 = Date.now();
  const st = core.stackBegin(set.width, set.height);
  for (const f of set.rgba) core.stackAdd(st, new Uint8ClampedArray(fs.readFileSync(f)), set.width, set.height);
  const fin = core.stackFinish(st);
  const info = core.parseFileHeader(new Uint8Array(fs.readFileSync(set.file)), set.name);
  info.width = set.width; info.height = set.height; info.notes = [];
  if (!set.jpeg) info.is_jpeg = false;
  const res = core.analyzeChannels(fin.L, info, { res: fin.res });
  const acc = [];
  for (const f of set.rgba) {
    const L = core.stackAdd(core.stackBegin(set.width, set.height), new Uint8ClampedArray(fs.readFileSync(f)), set.width, set.height);
    core.coherenceAdd(res.patterns, L, acc);
  }
  core.coherenceFinish(res.patterns, acc);
  out.push({
    name: set.name, ms: Date.now() - t0, w: res.w, h: res.h, droppedInfo: res.droppedInfo,
    patterns: res.patterns.map(p => ({
      category: p.category, period: core.peakPeriod(p.rep), angle: core.peakAngle(p.rep), src: p.rep.src,
      amplitude: p.measure.amplitude, is_global: p.measure.is_global, strength: p.strength,
      across: p.across, peaks: p.peaks.map(q => [q.u, q.v, q.view, q.score]),
    })),
  });
}
process.stdout.write(JSON.stringify(out));
