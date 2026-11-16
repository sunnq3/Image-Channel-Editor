"""Measure hit and false-alarm rates on images with known answers.

    python tests/benchmark.py            # prints a table (used for the README)
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from cases import build_cases, synthetic_photo   # noqa: E402
from patternscan.analyze import analyze          # noqa: E402


def matches(pat, cats, uv, w, h):
    if pat.category not in cats:
        return False
    if uv is None:
        return True
    u, v = uv
    for p in pat.peaks:
        for s in (1, -1):
            if abs(p.u - s * u) * w <= 2.5 and abs(p.v - s * v) * h <= 2.5:
                return True
    return False


def evaluate(path, expect):
    res = analyze(path)
    w, h = res["info"]["analysed_width"], res["info"]["analysed_height"]
    pats = res["patterns"]
    used = set()
    hits = []
    for cats, uv in expect["must"]:
        ok = False
        for i, p in enumerate(pats):
            if matches(p, cats, uv, w, h):
                used.add(i)
                ok = True
        hits.append(ok)
    # harmonics / wrong-category reports of a planted frequency are not false alarms,
    # but a wrong category is a miss
    extra = []
    for i, p in enumerate(pats):
        if i in used or p.category in expect["allowed"]:
            continue
        if any(uv is not None and matches(p, {p.category}, uv, w, h) for _, uv in expect["must"]):
            continue
        extra.append(p)
    return hits, extra, res


def main(seeds=tuple(range(61, 73)), verbose=True):
    # Seeds 1-6, 21-32 and 41-52 were looked at while tuning. The default set below was
    # generated after the last change and is what the README numbers come from.
    sizes = [(640, 512), (512, 512), (768, 576), (900, 600), (512, 768), (1024, 768)] * 2
    stats = defaultdict(lambda: [0, 0, 0])      # name -> [hits, expected, false alarms]
    with tempfile.TemporaryDirectory() as tmp:
        for seed, (w, h) in zip(seeds, sizes):
            base = synthetic_photo(seed, w, h)
            for name, data, ext, expect in build_cases(base, f"s{seed}"):
                path = os.path.join(tmp, f"case.{ext}")
                with open(path, "wb") as fh:
                    fh.write(data)
                hits, extra, res = evaluate(path, expect)
                key = name.split("/", 1)[1]
                stats[key][0] += sum(hits)
                stats[key][1] += len(hits)
                stats[key][2] += len(extra)
                if verbose and (not all(hits) or extra):
                    print(f"  ! {name}: found={hits} unexpected="
                          f"{[(p.category, round(p._rep.period, 2), round(p._rep.angle)) for p in extra]}")
    print(f"\n{'case':46s} {'found':>9s} {'false alarms':>13s}")
    th = te = tf = 0
    for key, (hh, ee, ff) in stats.items():
        print(f"{key:46s} {str(hh) + '/' + str(ee) if ee else '-':>9s} {ff:>13d}")
        th, te, tf = th + hh, te + ee, tf + ff
    print(f"{'TOTAL':46s} {str(th) + '/' + str(te):>9s} {tf:>13d}   ({len(seeds)} base pictures)")
    return stats


if __name__ == "__main__":
    main()
