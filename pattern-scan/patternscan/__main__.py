"""Command line:  python -m patternscan IMAGE [-o report.html] [--json report.json]"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .analyze import CATEGORIES, analyze
from .report import _period_text, _summary_lines, build_report


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="patternscan",
        description="Find repeating patterns in an image, highlight them, and write a report.")
    ap.add_argument("image", help="image file (PNG, JPEG, WebP, TIFF, ...)")
    ap.add_argument("-o", "--out", help="report file (default: IMAGE.patterns.html)")
    ap.add_argument("--json", help="also write the findings as JSON to this file")
    ap.add_argument("--full", action="store_true",
                    help="analyse up to 4096 px per side instead of the central 2048 "
                         "(several times slower)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.image):
        ap.error(f"file not found: {args.image}")
    out = args.out or os.path.splitext(args.image)[0] + ".patterns.html"
    t0 = time.time()
    try:
        res = analyze(args.image, full=args.full)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    build_report(res, out, args.json)

    for line in _summary_lines(res):
        print(line)
    for i, p in enumerate(res["patterns"], 1):
        print(f"  {i}. {CATEGORIES[p.category][0]:42s} every {_period_text(p)}")
    print(f"\nReport: {out}   ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
