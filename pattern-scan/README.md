# Pattern scan

Give it one image. It reads every pixel, tests every stripe width and direction for
repeating patterns, draws what it finds on top of the image, and writes a report
underneath that says what each pattern most likely is.

It separates patterns by cause, so that JPEG blocks and resize traces are not mistaken
for something that was added to the picture:

| Colour | Cause | What it is |
|---|---|---|
| orange | JPEG compression blocks | the 8x8-pixel grid JPEG leaves, confirmed by the small steps at block edges |
| brown | 16-pixel colour blocks | JPEG storing colour at half resolution |
| amber | 8-pixel lattice, not JPEG | repeats every 8 px but without JPEG's block edges |
| purple | 2/4-pixel lattice | sensor mosaic, whole-factor enlargement, AI-generator upsampling |
| blue | resize / rotate trace | the rhythm interpolation leaves in the image's roughness |
| pink | faint global overlay | a ripple added over the whole frame, same alignment everywhere, too faint to see (invisible watermarks look like this) |
| yellow | strong global pattern | visible stripes, scan lines, halftone, a photographed screen |
| green | repeating texture in part of the image | fabric, fences, bricks: real subjects |
| grey | sub-level trace | a ripple under a third of one brightness step, listed for completeness |

## Use it in a browser

The browser tool is built around one question: does the image carry a watermark-like
pattern, meaning a faint regular ripple laid over the whole frame? The report opens with
that verdict. For each such pattern it gives the strength, the direction, where it is,
its strength in each colour channel, a table of the carrier waves it is built from
(stripe width, angle, strength, phase, position in the frequency map) and a picture of
the pattern by itself with the image removed. JPEG blocks, pixel lattices and resize
traces are still identified, so they are not mistaken for an added pattern, but they
are folded away at the end and are not drawn on the picture.

Open `index.html`. It is one self-contained file, like the editor in the repo root: choose,
drop or paste an image and the report appears underneath. It works on a phone. The image
never leaves the device; all the analysis runs in the page.

The page analyses up to 1024 px per side by default (about 6 s on a laptop, longer on a
phone) or 2048 px on request. Larger images are read at full resolution from their
centre, never shrunk.

### Removing a pattern

Under the verdict is a **Remove patterns** panel. Every watermark-like pattern the scan
found is listed and ticked; a pattern can also be typed in by its repeat and angle, as
this page's table reports them (rounded values are fine). Remove takes the ticked ones
out of every image that was scanned, at full resolution (not just the analysed centre),
and gives back PNGs: one by one, or all in a .zip. **Scan the cleaned images** runs the
same scan on the result to confirm.

How it works (`web/clean.js`):

1. **Measure.** Each image is cut into tiles and each carrier wave is fitted in every
   tile. Busy tiles count less. Tiles far above the rest are real texture near the same
   frequency (a fence, a fabric) and are left out, so they are neither used nor touched.
2. **Find the exact frequency.** A frequency read from a crop, a stack or a typed-in
   value is close but not exact, and over a 4000-pixel row a tiny error becomes a large
   phase drift. The tool searches for the frequency at which the tiles of all images line
   up best, and keeps it only if it beats what noise alone could produce.
3. **Estimate once, from all images.** When the images agree on the ripple's alignment
   (the usual case for a watermark) one shared amplitude and phase is used for all of
   them, which is much less noisy than any single image's. Otherwise each image gets its
   own. Estimates that are mostly noise are shrunk toward zero rather than subtracted.
4. **Subtract with dither.** The ripple is subtracted (in brightness by changing R, G and
   B equally, in a colour channel through the JPEG colour transform). A correction of
   0.05 of a level would vanish when rounded to 8 bits, so pixels are quantised with
   dither, `floor(x + U[0,1))`, which is unbiased: on average every pixel lands on the
   corrected value. Clipped pixels (0 or 255) are left exactly as they were.

`python tests/test_clean.py` checks it on known answers: 16 generated pictures sharing
a 5.03 px ripple at 86° and a 2.05 px ripple at 92°, one picture also carrying strong
real 5.2 px stripes in one region.

| Case | 5.03 px removed | 2.05 px removed | Real texture changed |
|---|---|---|---|
| found by the scan (±0.12 and ±0.05) | 97–98% | 90–92% | 1.0% |
| typed in, exact (±0.05 and ±0.03) | 93–94% | 82–85% | 0.5% |
| typed in, true ripple at 5.034 px / 86.3° and 2.053 px / 91.7° | 93–94% | 95–97% | 0.5% |
| one picture alone, typed in (±0.05 and ±0.03) | 61% | 57% | 0.5% |

After cleaning, a scan of the cleaned stack reports neither ripple. One picture alone
cannot measure a ripple this faint precisely, so only part of it is taken out (the page
says so); 10 or more pictures from the same source measure it well.

Notes: output is PNG, because a JPEG re-save adds its own noise. Pixel values are kept as
stored; an embedded colour profile is not carried over.

### Several images: stacking

Choose several images of the same pixel size and the page stacks them instead of
analysing one. Each picture's own content is different, so it fades as images are added;
a pattern that sits in the same place in every image keeps its full strength. Two stacks
are made and scanned with the same tests as a single image:

- **Flat-image averaging**: the plain average of the images. Best when the images are
  already plain (white, grey, a blank frame), and it keeps wide stripes.
- **Noise-residual stacking**: each image minus a smoothed copy of itself, averaged with
  more weight where that image is smooth. Best for ordinary photos of different scenes.

Every pattern found is then looked up in each image separately. If the images agree on
its alignment far more than chance allows, the report says every image carries it; if
not, it says the pattern comes from individual images. The report also shows both stacks
as pictures, amplified.

`python tests/test_stack.py` checks this on known answers: twelve different generated
pictures sharing a +-0.8-level ripple with 30 px stripes and a +-0.1-level ripple with
6 px stripes. One of those pictures alone shows neither; the stack finds both, measures
their strength, and marks them as shared. Stripes present in only one of twelve pictures
are marked as not shared, and twelve clean pictures give nothing. Ten to thirty images
is a good number; fewer than five cannot be judged for sharing. Stacking is in the
browser tool only, not in the Python command line.

A mark placed exactly on a 2-, 4-, 8- or 16-pixel frequency looks like compression in a
single image. A stack separates them: compression blocks differ from picture to picture,
so their added brightness averages away, while a placed mark keeps its alignment. Such a
component is reported as an overlay when every image carries it and it is stronger than
0.25 of a brightness level. Below that it stays with the compression traces, because
decoding a JPEG leaves a fixed ripple of about 0.1 level on those frequencies by itself.
The test covers both: a +-0.8-level ripple on the 8-pixel lattice in JPEG files is
reported as a shared overlay next to the JPEG grid, and clean JPEG stacks at quality 50,
85 and 95 report compression only.

The browser version is the same analysis as the Python tool, ported line for line
(`web/core.js`). `python tests/compare_js.py` runs both on the same pixels: on 76 test
images they produced the same list of findings every time. To rebuild `index.html` after
changing anything in `web/`, run `python web/build.py`.

## Use it from the command line

```
pip install -r requirements.txt
python -m patternscan photo.png                 # writes photo.patterns.html
python -m patternscan photo.jpg -o report.html --json report.json
python -m patternscan big_photo.jpg --full      # analyse more than the central 2048 px
```

Open the HTML file in any browser. It is one self-contained file: the highlighted image,
a plain-language summary, one card per pattern (where it is, a zoomed patch with the
pattern isolated and amplified), the compression history, the frequency maps and a table.

`examples/demo.patterns.html` is a finished report for `examples/demo.jpg`, a generated
picture with a faint diagonal ripple over the whole frame, visible stripes in one region,
and JPEG compression on top. All three are found and named separately.

Use the original file. A screenshot or a re-saved copy has different pixels.

## How it works

1. **Read the pixels** exactly as stored (no resizing), split into brightness and two
   colour channels. The file's own JPEG tables are read when present.
2. **Scan all frequencies**, four ways per channel (12 scans for a colour image):
   - the pixel values, plain: finds strong patterns anywhere;
   - the pixel values, weighted toward smooth areas: finds faint patterns, which are only
     measurable where the picture is quiet;
   - two roughness maps (how much each pixel differs from its neighbours): finds patterns
     in the image's *statistics*, which is where JPEG blocks and resizing live.
3. **Divide out the picture.** Each spectrum is normalised by its local background, so
   ordinary detail gives a flat map and only a regular pattern stands out. A peak must be
   isolated and far above that background to count.
4. **Measure each pattern back in the image**: its strength in brightness levels, where it
   occurs, and whether it has the same strength and alignment in every zone that is quiet
   enough to judge. Overlays and processing pass that test; real subjects do not.
5. **Name the cause** from position and behaviour:
   - on the 1/8 or 1/16 frequency lattice, with block-edge steps found in the pixels: JPEG;
   - on the 1/2 or 1/4 lattice: pixel lattice;
   - only in the roughness maps, whole frame: resize or rotate;
   - in the pixel values, whole frame: overlay (faint or visible by its measured strength);
   - in part of the frame: texture.
   Harmonics, mixes of two patterns, and the echoes that block compression makes of an
   overlay are folded into the pattern they come from instead of being listed separately.
6. **JPEG history test**, independent of the spectra: looks for the 8-pixel grid of block
   edges directly. It also reports where the grid starts; a start other than (0, 0) means
   the picture was cropped after it was compressed.

## Measured accuracy

`python tests/benchmark.py` builds pictures with known answers and scores the tool. The
base pictures are generated (clouds, soft-edged shapes, sensor noise), so they have no
camera or compression history of their own and anything reported on a clean one is a
false alarm. Thresholds were tuned on other pictures; this is the result on 12 pictures
(228 test images) generated afterwards:

| Case | Found | Unexpected extra reports |
|---|---|---|
| clean picture | nothing to find | 0 |
| faint ripple, 5 px, +-2 levels | 12/12 | 0 |
| faint ripple, 12.5 px, +-4 levels | 12/12 | 0 |
| visible stripes, 9 px, +-16 levels | 12/12 | 1 |
| faint 2-D grid (two ripples) | 24/24 | 2 |
| stripes in one region only | 12/12 | 0 |
| stripes in one region, exactly 8 px wide | 12/12 | 0 |
| JPEG quality 50 / 75 / 90 / 95 | 48/48 | 0 |
| JPEG re-saved as PNG | 12/12 | 0 |
| JPEG, cropped, saved as PNG | 12/12 | 0 |
| enlarged 1.3x / 1.5x (bicubic) | 24/24 | 0 |
| enlarged 2x (pixel doubling) | 10/12 | 1 |
| enlarged exactly 2x (bicubic) | 12/12 | 0 |
| rotated 7 degrees | 12/12 | 0 |
| faint ripple, then JPEG quality 85 | 24/24 | 2 |
| **Total** | **238/240** | **6** |

"Found" means found *and* given the right cause. In the last row both the ripple and the
JPEG grid must be reported, each under its own name. A second check on two 2048x1536
pictures (clean, two faint ripples, regional stripes, JPEG, enlargement, ripple plus
JPEG) gave 14/14 found and no unexpected reports.

Faintest whole-frame ripple it finds (brightness levels out of 255, generated pictures):

| Stripe width | 640x512 image | 1600x1200 image |
|---|---|---|
| 3 to 5 px | 0.5 | 0.5 |
| 12.5 px | 0.5 to 0.75 | 0.5 |
| 30 px | 3 | 0.5 |
| 60 px | 8 to 12 | 3 |

On four small real photographs (300 to 640 px, the sample images shipped with
scikit-image) the faintest ripple found was +-1 level at 5 px and +-1 to 2 levels at
12.5 px.

Speed on two CPU cores: about 3 s for 1024x768, 10 to 25 s for 2048x1536. Larger images
are analysed in their central 2048x2048 pixels, at full resolution and never resized;
whole-frame patterns are all there. `--full` analyses up to 4096 px per side instead
(a 12-megapixel photo then takes about two minutes).

## Limits

- **One image is weaker evidence than many.** A slow ripple (wide stripes) or one buried
  under busy detail can be missed; the tables above show how faint is too faint. Real
  photographs are busier than the generated test pictures, so they need a ripple about
  twice as strong.
- **A cause is read from geometry.** "Faint global overlay" means a whole-frame ripple
  that JPEG, pixel lattices and resizing do not explain. It does not say who or what put
  it there.
- **Anything sitting exactly on the 2/4/8/16-pixel lattice is attributed to the lattice**
  unless it is strong and clearly limited to part of the frame. A pattern deliberately
  placed on those frequencies would be labelled as a lattice.
- **Tested on generated pictures with planted patterns**, plus spot checks on a handful of
  photographs. The photographs showed traces of their own history (earlier JPEG grids,
  earlier resizing), which is what the tool is meant to find but means there is no "clean"
  real photo to count false alarms on.
- Only straight, evenly spaced patterns are found. Patterns that bend, or change spacing
  with perspective, spread over many frequencies and may be missed or reported as several
  weak textures.

## Files

```
index.html                the browser tool (built from web/)
web/core.js               the analysis, ported to JavaScript
web/render.js             report text and pictures for the page
web/clean.js              pattern removal (measure, exact frequency, subtract with dither)
web/template.html         page layout and controls
web/build.py              assembles index.html
tests/compare_js.py       checks the browser core against the Python tool (needs Node.js)
tests/test_stack.py       known-answer check of multi-image stacking (needs Node.js)
tests/test_clean.py       known-answer check of pattern removal (needs Node.js)
tests/run_clean.js        Node harness for it: scan, remove, scan again
patternscan/analyze.py    the analysis (spectra, peak finding, measurement, classification)
patternscan/report.py     pictures and the HTML report
patternscan/__main__.py   command line
tests/cases.py            generator for test images with known answers
tests/benchmark.py        hit / false-alarm table above
tests/test_patternscan.py fast checks:  python -m pytest tests -q
examples/                 demo image, its report, and the script that builds the image
```

## Background reading

The methods are standard image forensics, combined: noise-residual and spectral-peak
analysis for periodic fingerprints (Corvi et al., "Intriguing properties of synthetic
images", 2023), JPEG blocking-artifact grid detection (Li et al. 2009; Nikoukhah et al.
2019-2020), resampling detection through periodic interpolation statistics (Popescu and
Farid 2005; Gallagher 2005; Kirchner 2008), and spread-spectrum watermark detection by
correlation (Cox et al. 1997).
