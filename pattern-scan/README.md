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

## Use

```
pip install -r requirements.txt
python -m patternscan photo.png                 # writes photo.patterns.html
python -m patternscan photo.jpg -o report.html --json report.json
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
| rotated 7 degrees | 12/12 | 2 |
| faint ripple, then JPEG quality 85 | 24/24 | 2 |
| **Total** | **238/240** | **8** |

"Found" means found *and* given the right cause. In the last row both the ripple and the
JPEG grid must be reported, each under its own name.

Faintest whole-frame ripple it finds (brightness levels out of 255, generated pictures):

| Stripe width | 640x512 image | 1600x1200 image |
|---|---|---|
| 3 to 5 px | 0.5 | 0.5 |
| 12.5 px | 0.5 to 0.75 | 0.5 |
| 30 px | 3 | 0.5 |
| 60 px | 8 to 12 | 3 |

Speed on two CPU cores: about 3 s for 1024x768, about 25 s for 2048x1536. Images larger
than 4096 px on a side are analysed in their central 4096 px, at full resolution.

## Limits

- **One image is weaker evidence than many.** A slow ripple (wide stripes) or one buried
  under busy detail can be missed; the table above shows how faint is too faint. Real
  photographs are busier than the generated test pictures, so expect less sensitivity on
  them for stripes wider than about 10 px.
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
