"""Check that the browser core (web/core.js) and the Python tool agree.

    python tests/compare_js.py            # needs Node.js

Runs both on the same pixels for every benchmark case and scores the browser
core against the known answers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from cases import build_cases, synthetic_photo   # noqa: E402
from patternscan.analyze import analyze          # noqa: E402


class JsPeak:
    def __init__(self, u, v):
        self.u, self.v = u, v


def js_matches(pat, cats, uv, w, h):
    if pat["category"] not in cats:
        return False
    if uv is None:
        return True
    for (u, v, _view, _score) in pat["peaks"]:
        for s in (1, -1):
            if abs(u - s * uv[0]) * w <= 2.5 and abs(v - s * uv[1]) * h <= 2.5:
                return True
    return False


def main(seeds=(61, 62, 63, 64, 65, 66), compare_python=True):
    sizes = [(640, 512), (512, 512), (768, 576), (900, 600), (512, 768), (1024, 768)]
    tmp = tempfile.mkdtemp()
    manifest, expects, py = [], {}, {}
    for seed, (w, h) in zip(seeds, sizes):
        base = synthetic_photo(seed, w, h)
        for name, data, ext, expect in build_cases(base, f"s{seed}"):
            fpath = os.path.join(tmp, f"{len(manifest)}.{ext}")
            with open(fpath, "wb") as fh:
                fh.write(data)
            im = Image.open(fpath).convert("RGBA")
            rpath = fpath + ".rgba"
            with open(rpath, "wb") as fh:
                fh.write(np.asarray(im).tobytes())
            manifest.append({"name": name, "file": fpath, "rgba": rpath,
                             "width": im.width, "height": im.height})
            expects[name] = expect
            if compare_python:
                res = analyze(fpath)
                py[name] = sorted(p.category for p in res["patterns"])
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh)
    out = subprocess.run(["node", "--max-old-space-size=4096", os.path.join(HERE, "run_core.js"), mpath],
                         capture_output=True, text=True, check=True).stdout
    results = json.loads(out)

    stats = defaultdict(lambda: [0, 0, 0])
    same = 0
    for r in results:
        name, expect = r["name"], expects[r["name"]]
        w, h = r["w"], r["h"]
        used, hits = set(), []
        for cats, uv in expect["must"]:
            ok = False
            for i, p in enumerate(r["patterns"]):
                if js_matches(p, cats, uv, w, h):
                    used.add(i)
                    ok = True
            hits.append(ok)
        extra = []
        for i, p in enumerate(r["patterns"]):
            if i in used or p["category"] in expect["allowed"]:
                continue
            if any(uv is not None and js_matches(p, {p["category"]}, uv, w, h)
                   for _, uv in expect["must"]):
                continue
            extra.append(p)
        key = name.split("/", 1)[1]
        stats[key][0] += sum(hits)
        stats[key][1] += len(hits)
        stats[key][2] += len(extra)
        js_cats = sorted(p["category"] for p in r["patterns"])
        if compare_python:
            if js_cats == py[name]:
                same += 1
            else:
                print(f"  differs from Python: {name}: js={js_cats} py={py[name]}")
        if not all(hits) or extra:
            print(f"  ! {name}: found={hits} unexpected="
                  f"{[(p['category'], round(p['period'], 2)) for p in extra]}")
    print(f"\n{'case':46s} {'found':>9s} {'false alarms':>13s}")
    th = te = tf = 0
    for key, (a, b, c) in stats.items():
        print(f"{key:46s} {str(a) + '/' + str(b) if b else '-':>9s} {c:>13d}")
        th, te, tf = th + a, te + b, tf + c
    print(f"{'TOTAL (browser core)':46s} {str(th) + '/' + str(te):>9s} {tf:>13d}")
    if compare_python:
        print(f"Same list of findings as the Python tool: {same} of {len(results)} images")
    print("average time per image in Node: %.1f s" % (sum(r["ms"] for r in results) / len(results) / 1000))


if __name__ == "__main__":
    main()
