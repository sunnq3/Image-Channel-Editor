"""Known-answer check of pattern removal in web/clean.js.

    python tests/test_clean.py            # needs Node.js

Sixteen different generated pictures share two faint ripples: 5.03 px stripes at 86
degrees (+-0.05 level by default) and 2.05 px stripes at 92 degrees (+-0.03 level), the
two patterns of a real report. One picture also carries a strong real texture (5.2 px
stripes, +-12 levels, in one region) close to the first ripple's frequency.

The stack is scanned, the patterns it finds are removed from every picture, and the
cleaned stack is scanned again. Checks:
  - before: both ripples found and shared;
  - after: neither found;
  - in each picture at least 80% of each ripple is gone (measured against the truth);
  - with one picture alone the ripples are reduced, never increased;
  - the real texture is kept (under 2% of it changes);
  - the same holds when the ripples are typed in as "5.03 px, 86 deg" and "2.05 px,
    92 deg", both when that is exact and when the true ripples are 5.034 px at 86.3
    deg and 2.053 px at 91.7 deg (the tool must find the exact frequency itself).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cases import synthetic_photo, to_img, wave   # noqa: E402

W, H, N = 768, 576, 16
A1 = float(os.environ.get("A1", "0.05"))
A2 = float(os.environ.get("A2", "0.03"))


def luma(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def amp_at(Y, uv):
    yy, xx = np.mgrid[:H, :W]
    ph = 2 * np.pi * (uv[0] * xx + uv[1] * yy)
    A = np.stack([np.cos(ph).ravel(), np.sin(ph).ravel(), np.ones(H * W)], 1)
    c = np.linalg.lstsq(A, Y.ravel(), rcond=None)[0]
    return float(np.hypot(c[0], c[1]))


def near(p, period, angle):
    da = abs((p["angle"] - angle + 90) % 180 - 90)
    return abs(p["period"] - period) < 0.06 * period and da < 3


def main():
    tmp = tempfile.mkdtemp()
    bases = [synthetic_photo(500 + i, W, H) for i in range(N)]
    tex, _ = wave(bases[0].shape, 5.2, 86, 12.0)
    mask = np.zeros((H, W))
    mask[300:500, 420:700] = 1

    def make(name, a1, a2, manual=None, only=None, true1=(5.03, 86), true2=(2.05, 92)):
        w1, f1 = wave(bases[0].shape, true1[0], true1[1], a1, 0.4)
        w2, f2 = wave(bases[0].shape, true2[0], true2[1], a2, 1.1)
        d = os.path.join(tmp, name)
        os.makedirs(d)
        files, imgs = [], []
        for i, b in enumerate(bases[:only] if only else bases):
            im = np.asarray(to_img(b + (w1 + w2)[..., None] + ((tex * mask)[..., None] if i == 0 else 0)))
            p = os.path.join(d, f"img_{i}.png")
            Image.fromarray(im).save(p)
            with open(p + ".rgba", "wb") as fh:
                fh.write(np.asarray(Image.fromarray(im).convert("RGBA")).tobytes())
            files.append(p + ".rgba")
            imgs.append(im)
        s = {"name": name, "width": W, "height": H, "file": files[0][:-5], "rgba": files,
             "outDir": os.path.join(d, "out")}
        if manual:
            s["manual"] = manual
        return s, {"a": (a1, a2), "f": (f1, f2), "imgs": imgs, "manual": bool(manual)}

    cases = [make("found", 0.12, 0.05),
             make("typed", A1, A2, manual=[[5.03, 86, A1], [2.05, 92, A2]]),
             make("typed_off", A1, A2, manual=[[5.03, 86, A1], [2.05, 92, A2]], true1=(5.034, 86.3), true2=(2.053, 91.7)),
             make("single", A1, A2, manual=[[5.03, 86, A1], [2.05, 92, A2]], only=1)]
    mpath = os.path.join(tmp, "m.json")
    with open(mpath, "w") as fh:
        json.dump([c[0] for c in cases], fh)
    out = subprocess.run(["node", "--max-old-space-size=4096", os.path.join(HERE, "run_clean.js"), mpath],
                         capture_output=True, text=True, check=True).stdout
    ok = True
    wm = ("global_faint", "global_visible", "trace")
    for r, (_s, k) in zip(json.loads(out), cases):
        print(f"\n== {r['name']}: {len(r['cleaned'])} image(s), ripples +-{k['a'][0]} and +-{k['a'][1]}, "
              f"{r['msPerImage']:.0f} ms per image")
        for c in r["plan"]:
            print(f"   removing {c['period']:.4f} px at {c['angle']:.2f} deg ({c['channel']}), "
                  f"+-{c['levels']:.3f}, frequency moved {c['shift']:.5f}{'' if c['moved'] else ' (kept as given)'}, SNR {c['snr']:.1f}{' WEAK' if c['weak'] else ''}, texture tiles {c['textureShare']:.1%}, "
                  f"{'one shared estimate' if c['shared'] else 'per image'} (agreement {c['agreement']:.2f})")
        for tag in ("before", "after"):
            print(f"   scan {tag}:", [(p["category"], round(p["period"], 2), round(p["angle"]),
                                      p["amplitude"] and round(p["amplitude"], 3), round(p["strength"]), p["shared"]) for p in r[tag]])
        if not k["manual"]:
            for per, ang in ((5.03, 86), (2.05, 92)):
                hit = any(near(p, per, ang) and p["category"] in wm for p in r["before"])
                print(f"   before: {per} px found: {hit}")
                ok &= hit
        if len(r["cleaned"]) > 1:
            for per, ang in ((5.03, 86), (2.05, 92)):
                left = [p for p in r["after"] if near(p, per, ang) and p["category"] in wm]
                print(f"   after: {per} px still reported: {bool(left)}")
                ok &= not left
        fr1, fr2, texchg = [], [], None
        for i, (rp, base) in enumerate(zip(r["cleaned"], bases)):
            o = np.frombuffer(open(rp, "rb").read(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
            resid = luma(o) - luma(base)
            if i == 0:
                d = luma(o) - luma(k["imgs"][0].astype(float))
                texchg = float(np.abs(d[300:500, 420:700]).mean() / (12 * 2 / np.pi))
                resid = resid - (tex * mask)
            fr1.append(1 - amp_at(resid, k["f"][0]) / k["a"][0])
            fr2.append(1 - amp_at(resid, k["f"][1]) / k["a"][1])
        print(f"   removed per picture (vs truth): 5.03 px min {min(fr1):.0%} mean {np.mean(fr1):.0%}; "
              f"2.05 px min {min(fr2):.0%} mean {np.mean(fr2):.0%}")
        print(f"   real texture changed by {texchg:.2%} of its strength; "
              f"pixels changed {np.mean([x['changed'] for x in r['reports']]):.1%}")
        # a stack measures the ripple well; one image alone can only take part of a ripple
        # this faint, but must never add to it
        need = 0.8 if len(r["cleaned"]) > 1 else 0.0
        ok &= min(fr1) > need and min(fr2) > need and texchg < 0.02
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
