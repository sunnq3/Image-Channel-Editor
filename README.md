# Channel Lab — Image Channel Editor

A single-file browser tool for editing an image at the pixel level and getting a JPG out that looks like one capture, not a composite.

**Open it:** `index.html` runs anywhere, no build, no server. Everything is in one file.

## What it does

- **RGB channel shift** — move each of R, G, B on X / Y (pixels) and Z (scale about the centre), with clamp / wrap / black edges, bilinear or nearest sampling.
- **Luminance (Y)** — gain, lift, gamma, and **Invert**: full negative, brightness only (keeps colours), or colour only.
- **Chroma (Cb / Cr)** — Cb only, Cb + Cr together, or separately: offset, gain, hue rotation. Live vectorscope and luma histogram.
- **Edges** — takes the harshness out of hard transitions without blurring the image: removes halos and ringing (the rims an invert makes loud), softens razor edges and stair-steps over a set radius, softens colour edges, and can snap colour fringes onto brightness edges. Strong content such as thin text strokes is protected. Edge map view shows where it acts.
- **Bake** — reads the photo's fingerprint from the parts you didn't touch and gives the edited parts the same one: softness, colour resolution, black level and highlight saturation, a frequency-band seam blend (when you drop in the original), a half-pixel resample that erases any previous JPEG grid, then grain last. One slider; every value is measured.
- **Regrain** — measures the native grain (σ per luminance, grain size, chroma level) and puts back only the variance that is missing where editing smoothed things over, so it reads as part of the image rather than a layer. Manual mode for images with no grain to copy.
- **Survives compression** — edits are applied in the JPEG's own YCbCr space and written straight into the file by a built-in baseline encoder (4:4:4 or 4:2:0, quality 1–100). The tool then decodes that exact file and reports per-plane error, how much of each edit was kept, PSNR, gamut clipping, and an **edit-visibility** score from a normalised error-level map (the **Seam map** view). A stress test re-encodes at 4:2:0 q75 to show what an upload will do.
- **Export** — the JPG you save is byte-for-byte the file that was measured.

## Views

Original · Edited · Compressed (decoded from the actual JPEG) · Difference (×1–×16) · Grain map · Seam map · Edge map. Fit or 100% with panning.

## Notes

- Images above 12 MP are scaled down on load (4 MP on low-memory devices).
- Full-resolution processing runs in a Web Worker; the preview stays live while you drag.
- The exported file carries no metadata, just the image.
