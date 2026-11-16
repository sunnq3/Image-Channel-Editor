/* Worker entry: pixels in, finished report out. */
self.onmessage = function (e) {
  var d = e.data;
  try {
    var progress = function (msg, frac) { self.postMessage({ type: 'progress', msg: msg, frac: frac }); };
    var info = self.PatternCore.parseFileHeader(new Uint8Array(d.head), d.name);
    var res = self.PatternCore.analyzePixels(new Uint8ClampedArray(d.rgba), d.width, d.height, info,
      { maxSide: Math.max(d.width, d.height) }, progress);
    if (d.orig && (d.orig.width !== d.width || d.orig.height !== d.height)) {
      info.width = d.orig.width; info.height = d.orig.height;
      info.notes.unshift('Image is large; analysed the central ' + d.width + 'x' + d.height +
        ' pixels at full resolution (starting at ' + d.orig.x0 + ',' + d.orig.y0 +
        '). Pictures and positions in this report refer to that region.');
    }
    var out = self.PatternRender.buildReport(res, progress);
    self.postMessage({ type: 'done', report: out.report }, out.transfer);
  } catch (err) {
    var msg = String((err && err.message) || err);
    if (/memory|allocation|Array buffer/i.test(msg)) msg = 'This device ran out of memory for that size. Choose 1024 px and try again.';
    self.postMessage({ type: 'error', message: msg });
  }
};
