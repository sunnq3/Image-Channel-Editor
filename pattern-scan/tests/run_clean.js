// Node harness for pattern removal: scan a stack, remove what it found, scan again.
//   node tests/run_clean.js manifest.json  -> prints JSON results
// Each set: { name, width, height, file, rgba: [...], outDir, manual?: [[period, angle], ...] }
const fs = require('fs');
const path = require('path');
const core = require(path.join(__dirname, '..', 'web', 'core.js'));
const clean = require(path.join(__dirname, '..', 'web', 'clean.js'));

function scan(set, files) {
  const st = core.stackBegin(set.width, set.height);
  for (const f of files) core.stackAdd(st, new Uint8ClampedArray(fs.readFileSync(f)), set.width, set.height);
  const fin = core.stackFinish(st);
  const info = core.parseFileHeader(new Uint8Array(fs.readFileSync(set.file)), set.name);
  info.width = set.width; info.height = set.height; info.notes = []; info.is_jpeg = false;
  const res = core.analyzeChannels(fin.L, info, { res: fin.res });
  const acc = [], list = res.patterns.concat(res.latticeCands);
  for (const f of files) {
    const L = core.stackAdd(core.stackBegin(set.width, set.height), new Uint8ClampedArray(fs.readFileSync(f)), set.width, set.height);
    core.coherenceAdd(list, L, acc);
  }
  core.coherenceFinish(list, acc);
  core.promoteSharedLattice(res);
  return res;
}
const brief = res => res.patterns.map(p => ({
  category: p.category, period: core.peakPeriod(p.rep), angle: core.peakAngle(p.rep), channel: p.channel,
  amplitude: p.measure.amplitude, strength: p.strength / core.LN2, shared: p.across ? p.across.shared : null,
}));

const sets = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const set of sets) {
  const before = scan(set, set.rgba);
  const carriers = set.manual
    ? set.manual.map(m => clean.carrierFromPeriodAngle(m[0], m[1], 'Y', m[2] || 0))
    : clean.carriersFromPatterns(before.patterns);
  const t0 = Date.now(), cleaned = [], reports = [];
  const read = f => new Uint8ClampedArray(fs.readFileSync(f));
  const files = set.only ? set.rgba.slice(0, set.only) : set.rgba;
  const measured = carriers.length ? files.map(f => clean.measure(read(f), set.width, set.height, carriers)) : [];
  const plan = carriers.length ? clean.solve(measured, carriers, set.opts || {}) : null;
  let deep = null;
  if (set.deep && files.length >= 2) {
    const D = clean.deepBegin(set.width, set.height, set.deep);
    files.forEach((f, i) => clean.deepAdd(D, read(f), clean.carrierCorrection(set.width, set.height, plan, i), i));
    deep = clean.deepSolve(D, set.deepOpts || {});
  }
  fs.mkdirSync(set.outDir, { recursive: true });
  files.forEach((f, i) => {
    const r = clean.finish(read(f), set.width, set.height, plan, deep, i);
    const p = path.join(set.outDir, path.basename(f) + '.clean');
    fs.writeFileSync(p, Buffer.from(r.rgba.buffer));
    cleaned.push(p); reports.push({ changed: r.changed });
  });
  const ms = (Date.now() - t0) / files.length;
  const after = files.length >= 2 ? scan(set, cleaned) : null;
  out.push({ name: set.name, carriers, plan: plan ? plan.carriers : [], deep: deep ? deep.stats : null, msPerImage: ms, cleaned, reports,
             before: brief(before), after: after ? brief(after) : [] });
}
process.stdout.write(JSON.stringify(out));
