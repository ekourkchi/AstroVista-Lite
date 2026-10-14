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
- **The sky factor.** The sky subtracted is `SKYFACTOR` × `SKY`. After a pair fit, the
  residual on the sky pixels (its median, sigma and histogram) suggests the next factor,
  factor + median / base; `SKYITER=n` refits up to n times and keeps the fit with the
  smallest residual median. The Pair fit panel shows the sky pixels' histogram, keeps
  every segmentation setting open (with `FWHM=`, a Gaussian detection filter, and a
  threshold down to 0.01 sigma), shows the sky mask on request, and lets the user take the
  measured median whatever the verdict (which is kept with the fit). Results ▸ Pair ▸ Sky
  lists the fits.
- **Two-galaxy iterative fit** (the Pair fit tool's default): A on D, B on D − M_A, then
  A on D − M_B and B on D − M_A each cycle, every input made afresh from the original image;
  no masking between the galaxies (`CORES=0`); each galaxy's R1 grows by its own schedule
  (`START=` with `FACTOR=` or `STEP=`, or `RADII=`); fixed centres by default; stops after
  `CONSECUTIVE=` converged cycles at the final radii (`CONVERGE=`); `ONFAIL=hold|retry|stop`.
  A live iteration table; `pairstep N K` shows any call's model and input; the export writes
  `iterations.json` (every call's input, companion, radii, mask, arguments, profile and
  model checksum). A pair folder's centres are moved to the light peak.
- **Block and grow.** `B: BLOCK=r` leaves a region around B out of A's first fit, so the
  larger galaxy is fitted alone first; `START=r` and `STEPS=n` start each galaxy at a small
  R1 and grow both, step by step, to their R1s (a galaxy whose growth fails stays at its
  last good radius). A pair folder opens with B blocked and three growth steps.
- **The pair report.** Copy report (and `pairreport N [file]`): the settings, every call
  and why it failed, growth and cycles, the sky, the residual in rings around each galaxy
  (facing the other galaxy and away from it) and hints, as plain text to paste anywhere.
- File ▸ Open a pair folder (and `astrovista --pair FOLDER`): a FITS image, its `.dmask`
  masks and a DS9 region file with the centres of A and B, opened together, with starting
  radii to check.
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
