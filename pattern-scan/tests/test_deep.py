"""Known-answer check of the deep clean in web/clean.js (everything the images share).

    python tests/test_deep.py            # needs Node.js

Sixteen different generated pictures share five marks:
  - 5.03 px ripple at 86 deg, +-0.12 level      (the scan finds it)
  - 2.05 px ripple at 92 deg, +-0.05 level      (the scan finds it)
  - 3.7 px ripple at 20 deg, +-0.04 level       (too faint for the scan)
  - 4.4 px ripple at 140 deg in Cr colour, +-0.1
  - a fixed noise-like pattern, 0.4 level rms   (no ripple at all: spread-spectrum style)

Removal is measured against the truth, in each picture, after saving as PNG, as JPEG
quality 95 (4:4:4) and as JPEG quality 90 (4:2:0, as browsers save). "Carriers only" is the removal of the scan's named patterns alone;
"deep" adds the shared-pattern pass. A second set of eight pictures of the SAME scene
checks that shared scenery is not mistaken for a mark.

How much of the noise-like mark can be separated from picture detail depends on the
number of images (about a quarter with 16, half with 48); the check asks for 20%.
"""
from __future__ import annotations

import io
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
Y_W = np.array([0.299, 0.587, 0.114])
CR_W = np.array([0.5, -0.418688, -0.081312])


def plane(a, k):
    return a @ k


def main():
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(7)
    bases = [synthetic_photo(700 + i, W, H) for i in range(N)]
    m1, _ = wave((H, W), 5.03, 86, 0.12, 0.4)
    m2, _ = wave((H, W), 2.05, 92, 0.05, 1.1)
    m3, _ = wave((H, W), 3.7, 20, 0.04, 2.0)
    m4, _ = wave((H, W), 4.4, 140, 0.10, 0.3)                      # in Cr
    from scipy.ndimage import gaussian_filter
    ss = gaussian_filter(rng.normal(0, 1, (H, W)), 0.7)
    ss = ss / ss.std() * 0.4
    luma_marks = {"5.03 px": m1, "2.05 px": m2, "3.7 px (faint)": m3, "noise-like": ss}
    # put the luma marks on R=G=B, the Cr mark through the inverse transform
    add = (m1 + m2 + m3 + ss)[..., None] + m4[..., None] * np.array([1.402, -0.714136, 0])

    def write(name, imgs):
        d = os.path.join(tmp, name)
        os.makedirs(d)
        files = []
        for i, a in enumerate(imgs):
            im = to_img(a)
            p = os.path.join(d, f"img_{i:02d}.png")
            im.save(p)
            with open(p + ".rgba", "wb") as fh:
                fh.write(np.asarray(im.convert("RGBA")).tobytes())
            files.append(p + ".rgba")
        return files

    marked = write("marked", [b + add for b in bases])
    scene = synthetic_photo(999, W, H)
    same = write("same", [scene + rng.normal(0, 1.4, scene.shape) + add for _ in range(8)])
    sets = [
        {"name": "carriers only", "width": W, "height": H, "file": marked[0][:-5], "rgba": marked, "outDir": os.path.join(tmp, "o1")},
        {"name": "deep", "width": W, "height": H, "file": marked[0][:-5], "rgba": marked, "outDir": os.path.join(tmp, "o2"),
         "deep": ["Y", "Cb", "Cr"]},
        {"name": "same scene", "width": W, "height": H, "file": same[0][:-5], "rgba": same, "outDir": os.path.join(tmp, "o3"),
         "deep": ["Y", "Cb", "Cr"]},
    ]
    mp = os.path.join(tmp, "m.json")
    with open(mp, "w") as fh:
        json.dump(sets, fh)
    out = json.loads(subprocess.run(["node", "--max-old-space-size=6000", os.path.join(HERE, "run_clean.js"), mp],
                                    capture_output=True, text=True, check=True).stdout)
    ok = True
    for r in out:
        print(f"\n== {r['name']}: {len(r['cleaned'])} images, {r['msPerImage']:.0f} ms per image")
        for c in r["plan"]:
            print(f"   named carrier {c['period']:.3f} px at {c['angle']:.1f} deg ({c['channel']}): +-{c['levels']:.3f}")
        if r["deep"]:
            for ch, s in r["deep"].items():
                print(f"   deep {ch}: removed {s["rms"]:.3f} rms (found {s["foundRms"]:.3f}), {s['share']:.2%} of frequencies"
                      + (" -- SKIPPED: images share content" if s["skipped"] else ""))
        print("   scan after:", [(p["category"], round(p["period"], 2), round(p["angle"])) for p in r["after"]
                                  if p["category"] in ("global_faint", "global_visible", "trace")])
        if r["name"] == "same scene":
            sk = r["deep"]["Y"]["skipped"]
            print(f"   shared scenery recognised and left alone: {sk}")
            ok &= sk
            continue
        for fmt in ("PNG", "JPEG q95", "JPEG q90 4:2:0"):
            rem = {k: [] for k in list(luma_marks) + ["4.4 px (Cr)"]}
            err_in, err_out = [], []
            for rp, base, src in zip(r["cleaned"], bases, marked):
                o = np.frombuffer(open(rp, "rb").read(), np.uint8).reshape(H, W, 4)[..., :3]
                if fmt != "PNG":
                    buf = io.BytesIO()
                    if fmt == "JPEG q95":
                        Image.fromarray(o).save(buf, "JPEG", quality=95, subsampling=0)
                    else:
                        Image.fromarray(o).save(buf, "JPEG", quality=90, subsampling=2)
                    o = np.asarray(Image.open(buf).convert("RGB"))
                o = o.astype(float)
                ry = plane(o, Y_W) - plane(base, Y_W)
                for k, m in luma_marks.items():
                    rem[k].append(1 - float((ry * m).sum() / (m * m).sum()))
                rc = plane(o, CR_W) - plane(base, CR_W)
                rem["4.4 px (Cr)"].append(1 - float((rc * m4).sum() / (m4 * m4).sum()))
                i0 = np.frombuffer(open(src, "rb").read(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
                err_in.append(np.sqrt(((plane(i0, Y_W) - plane(base, Y_W)) ** 2).mean()))
                err_out.append(np.sqrt(((plane(o, Y_W) - plane(base, Y_W)) ** 2).mean()))
            print(f"   {fmt}: " + "; ".join(f"{k} {np.mean(v):.0%} (min {min(v):.0%})" for k, v in rem.items()))
            if r["name"] == "deep":
                need = {"5.03 px": .85, "2.05 px": .8, "3.7 px (faint)": .8, "noise-like": .2, "4.4 px (Cr)": .6}
                ok &= all(np.mean(rem[k]) >= v for k, v in need.items())
            if fmt == "PNG":
                print(f"   distance from the unmarked picture (brightness rms): before {np.mean(err_in):.3f}, after {np.mean(err_out):.3f}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
