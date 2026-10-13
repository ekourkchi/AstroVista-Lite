# Changelog

Every release of AstroVista Lite (PyPI: `AstroVista`). Releases are tagged on GitHub; the
newest is at the top.

## Unreleased

- **Pairs of galaxies.** `pair N SKY= ... A: ... B: ...` and Tools ▸ Pair fit: two
  overlapping galaxies fitted in turn with ELLIPROF, each with the other's model taken
  out; frozen sky and cores; a model is kept only when its fit passes every check; a
  status that never mistakes a rejected update for convergence; optional adaptive R1;
  partial bootstrap starts that are never part of a result. The models, an unmasked
  residual, both sets of ellipses, the profiles and the change per cycle, in a worker
  thread with Stop. `pairexport` writes the inputs, settings, calls and products.
- **The sky by segmentation.** `sky N SEGMENT`: every source down to 0.5 sigma left out;
  the level, cell medians, a fitted gradient, sampling and spatial uncertainties, and a
  verdict on the support; measurements turn stale when the image or masks change.
- Mask roles (`mask role NAME fit|sky|candidate`); sky layers are never used in a fit.
- `elliprof`: `MODE=` (or `COS3X=`, `COS4X=`), `FIXCTR=`, `RLAW=`, `GAIN=`, `TIE=`,
  `MINUS=` and `RAW`.
- `window`: a cut-out of a buffer that keeps its WCS.
- A command can go on over several lines, each but the last ending in `\`.
- A synthetic pair example (File ▸ Open the pair example) and a lesson on the website.
- A fuller README: supported systems, Python versions, installing with conda,
  troubleshooting.
- A mask saved with the *Legacy bitmap* type always gets the `.dmask` ending.
- `CONTRIBUTING.md`, this changelog, and Zenodo metadata.
- `mask sources` and `mask segments` before any fit keep the largest source (the galaxy)
  unmasked, instead of masking it with the rest; the reply says which source was kept.
- Drafts of a JOSS paper and an arXiv manuscript in `paper/`, with the script that makes
  their figures and numbers from the example.

## 1.0.1 (2026-10-12)

- `filt` on an image with NaN pixels: a NumPy median filter, 25 times faster, with the
  same results.
- Variable names work in any case (`set XC=567`, then `X0=XC` or `X0=xc`).
- The terminal's syntax and a reference of every command, in Help ▸ Documentation and on
  the website.
- Credits to John Tonry for ELLIPROF and the MONSTA command language.

## 1.0.0 (2026-10-12)

The first release: ELLIPROF isophote fitting, the source finder (SEP), mask layers and
`.dmask` files, image arithmetic, MONSTA-style image commands, the terminal, the display
and contours, the Inspector, built-in documentation and the UGC 12517 example.

- Data commands run in the window's thread, and PySide6 6.12.0 is excluded: together
  they avoid a crash of PySide6 with Python 3.10 and 3.11 on Linux.
- Two images opened at once go into two buffers.
