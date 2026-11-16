"""Known-answer check of the multi-image (stacking) mode in web/core.js.

    python tests/test_stack.py            # needs Node.js

Twelve different generated pictures share two ripples that are too faint to find
in any one of them. Stacking must find both and say they are shared; a ripple that
only one picture has must be reported as not shared; clean pictures must give nothing.
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
from cases import jpeg_bytes, synthetic_photo, to_img, wave   # noqa: E402

W, H, N = 512, 384, 12


def near(p, uv):
    return any(abs(u - s * uv[0]) * W <= 2.5 and abs(v - s * uv[1]) * H <= 2.5
               for (u, v, _a, _b) in p["peaks"] for s in (1, -1))


def main():
    tmp = tempfile.mkdtemp()
    bases = [synthetic_photo(200 + i, W, H) for i in range(N)]
    slow, f_slow = wave(bases[0].shape, 30.0, 50, 0.8)       # wide stripes, +-0.8 level
    fine, f_fine = wave(bases[0].shape, 6.0, 20, 0.1)        # fine stripes, +-0.1 level
    lone, f_lone = wave(bases[0].shape, 9.0, 70, 20.0)       # strong, in ONE picture only
    shared = (slow + fine)[..., None]

    def write(name, imgs, as_jpeg=False):
        files, first = [], None
        for i, a in enumerate(imgs):
            im = to_img(a)
            path = os.path.join(tmp, f"{name}_{i}." + ("jpg" if as_jpeg else "png"))
            if as_jpeg:
                with open(path, "wb") as fh:
                    fh.write(jpeg_bytes(im, 85))
            else:
                im.save(path)
            rp = path + ".rgba"
            with open(rp, "wb") as fh:
                fh.write(np.asarray(Image.open(path).convert("RGBA")).tobytes())
            files.append(rp)
            first = first or path
        return {"name": name, "rgba": files, "file": first, "width": W, "height": H, "jpeg": as_jpeg}

    sets = [
        write("clean", bases),
        write("shared", [b + shared for b in bases]),
        write("single", [bases[1] + shared]),
        write("lone", [b + (lone[..., None] if i == 3 else 0) for i, b in enumerate(bases)]),
        write("shared_jpeg", [b + shared for b in bases], as_jpeg=True),
    ]
    mpath = os.path.join(tmp, "m.json")
    with open(mpath, "w") as fh:
        json.dump(sets, fh)
    out = subprocess.run(["node", "--max-old-space-size=4096", os.path.join(HERE, "run_stack.js"), mpath],
                         capture_output=True, text=True, check=True).stdout
    res = {r["name"]: r for r in json.loads(out)}
    for r in res.values():
        print(f"\n{r['name']}  ({r['ms'] / 1000:.1f} s)  set aside: {[(round(d['period'], 1), d['src'], d['cat'], d['amp'] and round(d['amp'], 2), d['zones']) for d in r['droppedInfo']]}")
        for p in r["patterns"]:
            a = p["across"]
            print(f"   {p['category']:14s} every {p['period']:6.2f} px  amp "
                  f"{'-' if p['amplitude'] is None else round(p['amplitude'], 2)}  "
                  f"aligned {a['aligned']}/{a['n']}  coherence {a['coherence']:.2f}  shared={a['shared']}")

    def find(name, uv):
        return [p for p in res[name]["patterns"] if near(p, uv)]

    checks = {
        "clean stack reports nothing": res["clean"]["patterns"] == [],
        "one image alone misses the wide ripple": not find("single", f_slow),
        "stack finds the wide ripple, shared": any(p["across"]["shared"] and p["category"] in ("global_faint", "trace") for p in find("shared", f_slow)),
        "stack finds the fine ripple, shared": any(p["across"]["shared"] and p["category"] in ("global_faint", "trace") for p in find("shared", f_fine)),
        "wide ripple strength within 0.25 of 0.8": any(abs(p["amplitude"] - 0.8) < 0.25 for p in find("shared", f_slow)),
        "fine ripple strength within 0.04 of 0.1": any(abs(p["amplitude"] - 0.1) < 0.04 for p in find("shared", f_fine)),
        "no other pattern in the shared stack": all(near(p, f_slow) or near(p, f_fine) for p in res["shared"]["patterns"]),
        "one-picture stripes are marked not shared": all(not p["across"]["shared"] for p in find("lone", f_lone)),
        "JPEG stack: both ripples shared": all(any(p["across"]["shared"] for p in find("shared_jpeg", f)) for f in (f_slow, f_fine)),
        "JPEG stack: block grid reported and shared": any(p["category"] == "jpeg_grid" and p["across"]["shared"] for p in res["shared_jpeg"]["patterns"]),
    }
    print()
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    return all(checks.values())


def test_stack_mode():
    assert main()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
