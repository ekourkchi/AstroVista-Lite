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
- **Automatic pair settings** (Pair fit ▸ Automatic settings, on by default): choose the
  sky, the masks and the two centres; the rest comes from 150 simulated pairs (Sérsic
  galaxies with ellipticity and angle changing with radius, boxy/disky shapes, outer
  isophotes off the inner centre, lopsided, tidal and m=1/m=3 distortions, heavy overlaps,
  companions 0.15–1 times A's size, stars on and off the galaxies, white and correlated
  noise, sky gradients), each run end to end (source finder, objects mask, sky, pair fit).
  `R1=AUTO` (where the light away from the companion falls below the noise; measured again
  on the image minus the companion's model of a first pass), `START=AUTO` (a third of R1,
  at most 0.35 of the separation, at least 9 px), `NR=AUTO` (isophotes about 10 % apart).
  Against the earlier defaults the pairs fitted rose from about 65 % to 91 %; capping the
  start by the separation fixed most heavy overlaps; measuring R1 again lowered the profile
  error of small companions from 5.6 % to 4.0 % (median).
- **Sky with correlated noise.** `sky N SEGMENT` scales its detection noise by how much
  more noise the filtered image has than white noise would give (1.7 on a drizzled HST
  frame). Before, noise correlated over a few pixels was taken for sources, most of the
  frame was masked and the level came out low (1.5 sigma on a simulated frame).
- **Mask the other objects** (Pair fit ▸ Fitting masks): the mask layer `objects`, role
  fit, with every detected source but A and B (finding the sources first if needed).
  `mask sources` and `mask segments` take several centres to keep (`X0=xa,xb Y0=ya,yb`),
  and also keep the segment under each centre. `mask segments` makes the layer exactly the segments:
  everything else is unmasked (`ADD` adds them to the layer instead); Masks ▸ Mask the
  found sources' segments writes its own layer, `segments`. The Layers panel has a **Pair fit** row,
  and unticking **Isophotes** hides the pair's fitted isophotes too.
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
- **Damping and the search for the best fit.** `DAMP=` subtracts a blend of the
  companion's new and previous models; undamped final cycles (`POLISH=`) make the result
  ELLIPROF's own models. `TRYDAMP=`, `TRYFIX=` and `TRYFACTOR=` fit every combination and
  keep the stable one with the smallest residual outside the cores, preferring fixed
  centres unless free centres lower that residual by more than 0.01 of the noise (on 24
  simulated pairs this improved both the residual and the profiles of A and B over simply
  keeping the smallest residual). The search is on by default in the Pair fit panel; every
  variant is listed with the reason for the one kept. The export's
  replay command repeats the fit kept with every setting explicit.
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
