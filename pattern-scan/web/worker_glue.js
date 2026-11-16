/* Worker entry: pixels in, finished report out.
   One image: a single message. Several images: stack-begin, stack-add (xN),
   stack-analyse, coh-add (xN), stack-finish. */
var stack = null;
function progress(msg, frac) { self.postMessage({ type: 'progress', msg: msg, frac: frac }); }
function fail(err) {
  var msg = String((err && err.message) || err);
  if (/memory|allocation|Array buffer/i.test(msg)) msg = 'This device ran out of memory for that size. Choose 1024 px, or fewer images, and try again.';
  self.postMessage({ type: 'error', message: msg });
  stack = null;
}
function finishStack() {
  self.PatternCore.coherenceFinish(stack.list, stack.acc);
  self.PatternCore.promoteSharedLattice(stack.res);
  for (var i = 0; i < stack.res.patterns.length; i++) {
    var pat = stack.res.patterns[i], a = pat.across;
    if (a && a.judged && !a.shared) pat.notes.push('The images do not agree on this one, so it is not a mark they all carry. One or a few images contain it strongly enough to show through the average.');
  }
  var out = self.PatternRender.buildReport(stack.res, function (m, f) { progress(m, 0.9 + 0.1 * f); });
  stack = null;
  self.postMessage({ type: 'done', report: out.report }, out.transfer);
}
self.onmessage = function (e) {
  var d = e.data, core = self.PatternCore;
  try {
    if (d.type === 'stack-begin') {
      stack = { st: core.stackBegin(d.width, d.height), orig: d.orig, skipped: d.skipped || 0,
                allJpeg: true, first: null, acc: [] };
      self.postMessage({ type: 'ack' });
    } else if (d.type === 'stack-add') {
      var hdr = core.parseFileHeader(new Uint8Array(d.head), d.name);
      if (!hdr.is_jpeg) stack.allJpeg = false;
      if (!stack.first) stack.first = hdr;
      core.stackAdd(stack.st, new Uint8ClampedArray(d.rgba), stack.st.w, stack.st.h);
      self.postMessage({ type: 'ack' });
    } else if (d.type === 'stack-analyse') {
      var n = stack.st.n, fin = core.stackFinish(stack.st);
      stack.st = null;
      var info = { path: n + ' images stacked', format: stack.allJpeg ? 'JPEG' : (stack.first.format === 'JPEG' ? 'mixed formats' : stack.first.format),
                   is_jpeg: stack.allJpeg, jpeg_quality: stack.allJpeg ? stack.first.jpeg_quality : null,
                   jpeg_subsampling: stack.allJpeg ? stack.first.jpeg_subsampling : null,
                   width: stack.orig.width, height: stack.orig.height, notes: [] };
      var res = core.analyzeChannels(fin.L, info, { res: fin.res }, function (m, f) { progress(m, 0.4 + 0.35 * f); });
      if (stack.orig.width !== res.w || stack.orig.height !== res.h) {
        info.notes.unshift('Images are large; stacked the central ' + res.w + 'x' + res.h + ' pixels of each at full resolution (starting at ' + stack.orig.x0 + ',' + stack.orig.y0 + ').');
      }
      if (stack.skipped) info.notes.push(stack.skipped + ' image(s) were left out because their size differs from the rest. A pattern tied to the pixel grid only lines up between images of the same size.');
      if (n < 10) info.notes.push('Only ' + n + ' images were stacked. Each picture fades by the square root of the count, so 10 to 30 images of different scenes work much better.');
      res.stack = { n: n, skipped: stack.skipped };
      res.stackRes = fin.res;
      stack.res = res;
      stack.list = res.patterns.concat(res.latticeCands);   // everything to look up in each image
      if (!stack.list.length) finishStack();
      else self.postMessage({ type: 'need-pass2' });
    } else if (d.type === 'coh-add') {
      var L = core.stackAdd(core.stackBegin(stack.res.w, stack.res.h), new Uint8ClampedArray(d.rgba), stack.res.w, stack.res.h);
      core.coherenceAdd(stack.list, L, stack.acc);
      self.postMessage({ type: 'ack' });
    } else if (d.type === 'stack-finish') {
      finishStack();
    } else {
      var info1 = core.parseFileHeader(new Uint8Array(d.head), d.name);
      var res1 = core.analyzePixels(new Uint8ClampedArray(d.rgba), d.width, d.height, info1,
        { maxSide: Math.max(d.width, d.height) }, progress);
      if (d.orig && (d.orig.width !== d.width || d.orig.height !== d.height)) {
        info1.width = d.orig.width; info1.height = d.orig.height;
        info1.notes.unshift('Image is large; analysed the central ' + d.width + 'x' + d.height +
          ' pixels at full resolution (starting at ' + d.orig.x0 + ',' + d.orig.y0 +
          '). Pictures and positions in this report refer to that region.');
      }
      var out = self.PatternRender.buildReport(res1, progress);
      self.postMessage({ type: 'done', report: out.report }, out.transfer);
    }
  } catch (err) { fail(err); }
};
