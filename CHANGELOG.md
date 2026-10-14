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
- **Automatic pair settings** (Pair fit ▸ Automatic settings): choose the
  sky, the masks and the two centres; the rest comes from 150 simulated pairs (Sérsic
  galaxies with ellipticity and angle changing with radius, boxy/disky shapes, outer
  isophotes off the inner centre, lopsided, tidal and m=1/m=3 distortions, heavy overlaps,
  companions 0.15–1 times A's size, stars on and off the galaxies, white and correlated
  noise, sky gradients), each run end to end (source finder, objects mask, sky, pair fit).
  `R1=AUTO` (where the light away from the companion falls below the noise; measured again
  on the image minus the companion's model of a first pass), `START=AUTO` (a third of R1,
  at most 0.35 of the separation, at least 9 px), `NR=AUTO` (isophotes about 10 % apart;
  30 % since the SBF suite).
  Against the earlier defaults the pairs fitted rose from about 65 % to 91 %; capping the
  start by the separation fixed most heavy overlaps; measuring R1 again lowered the profile
  error of small companions from 5.6 % to 4.0 % (median).
- **Pair fits for SBF.** A second simulation suite, closer to SBF data: pairs of
  PSF-convolved Sérsic ellipticals (n 2.5–5, with ellipticity gradients, twists, boxy or
  disky isophotes and tidal stretching) with surface brightness fluctuations, globular
  clusters (a cored system, the turnover at about one pixel of the galaxy at re), foreground
  stars, photon and read noise and sometimes a tilted sky; each run end to end. Judged
  against the truth on 32 scenes not used to choose the settings: the model of A + B within
  1.5 % of the truth (median, where the galaxies are brighter than the sky; 0.5 % to 2.5 %),
  dark or bright patches in the smoothed residual over less than 1 % of the galaxies in 27
  scenes in 32, the sky within 0.04 sigma (median), and the SBF amplitude measured on the
  residual 0.98 times that measured with the true model (within 10 % for 4 galaxies in 5).
  What it took, in the automatic settings and as the starting values of the manual ones:
  - `TAIL=sersic` (new): elliprof extrapolates its model beyond the last isophote with ln I
    linear in r^¼; for a shallower profile, or noisy last isophotes, that tail is far too
    bright, and in a pair it is subtracted from the companion's image: the dark regions in
    pair residuals. The tail is now a Sérsic law fitted to the outer profile.
  - `RMSTAR`: globular clusters on a galaxy belong to its detection segment and stay
    unmasked; rejecting bright samples along each isophote brought the SBF amplitude
    within 4 % of the truth for 9 galaxies in 10 (from within 28 %).
  - `TIE=-3`: the shapes smoothed over 3 isophotes (the isophote-to-isophote jitter showed
    as four-leaved patterns).
  - `NR=AUTO` isophotes about 30 % apart instead of 10 %: closer isophotes take up part of
    the fluctuations (20 % apart: 0.95 of the SBF amplitude; 30 %: 0.98).
  - `SKYITER=3` with a measured sky: the sky error fell from up to 0.9 sigma to 0.04 sigma
    (median). Each step
    is now at most 2 sigma, and the iterations stop, keeping the sky asked for, when the
    residual does not follow the sky (a halo that fills the frame takes up any change: on
    N6768 the sky ran from 2911 to 2311 without it).
  On the earlier suite of strongly distorted pairs the residual improved (model error 0.39 to
  0.21 sigma) but each galaxy's profile less so (A + B 2.7 % to 3.5 %, median), and a run
  takes about twice as long.
- **Pair fits on real frames** (seven HST and JWST pairs, run end to end with the
  automatic settings). On frames a halo fills, elliprof's and the Sérsic tails left the
  outskirts dark (NGC 6768: the model 405 against 66 in the far corners, -4 sigma on the
  sky pixels):
  - `TAIL=measured` (new, in the automatic settings): beyond the last isophote the model is
    the light measured in rings of its shape on the image minus the sky and the companion
    (medians, made non-increasing), a Sérsic law only beyond the last ring. On NGC 6768 the
    smoothed residual beyond 1 sigma fell from 57 % of the frame to 22 %; on the simulated
    pairs, bright patches (90th percentile) from 9 % to 1.6 % of the galaxies and B's
    model error from 3.4 % to 2.4 %, the SBF amplitude unchanged (0.98).
  - `SKYPLANE=1` (new, in the automatic settings): a plane fitted to the residual of the
    fit kept; when significant, the search again on the image minus its tilt, kept if the
    residual is lower.
  - No sky iterations when an automatic R1 reaches the frame's edge (the halo and the sky
    cannot be told apart); the report says so.
  - `mask segments ... WIDEN=f`: each segment grown by GROW + f x its equivalent radius;
    Mask the other objects uses 0.5, so bright stars' halos are covered.
  Still open on NGC 6768: wings of bright stars beyond their masks, a glow along the top
  edge, and weak spokes around the nucleus (about 1 % of its light).
- **Pair fit panel**: Settings is Automatic (the default), Automatic, adjustable (every
  setting shown at the automatic values, the radii still automatic) or Manual; new controls
  for the smoothing with radius, RMSTAR, the model beyond R1, the sky plane and the polish
  cycles, and auto boxes for NR and the initial R1.
- **The window stays responsive while a command runs** (the source finder, a sky, a sky
  plane, ELLIPROF, a script): those run in a worker thread, as the pair fit does, so the
  layers can be ticked and any buffer shown meanwhile. `ASTROVISTA_MAIN_THREAD=1` runs them
  in the window's thread as before.
- **A sky plane buffer is shown linear from its minimum to its maximum**, not as one grey,
  and the Sky plane panel no longer calls its fit stale when another buffer made from the
  image is shown.
- **The sky as a plane.** `skyplane N` and Tools ▸ Sky plane fit the sky as a level and a
  gradient, each with its error: the sources are found on the image minus the plane's tilt
  (so a bright side of the frame is not taken for a source), the plane is fitted to the
  medians of 32-pixel boxes of what is left, and the two are repeated until it settles. The
  report says whether the gradient is significant and whether the boxes scatter more than
  the noise allows (light that is not sky). `TOFLAT=` makes the image with the tilt taken
  out and the level kept, with its sky measured, for ELLIPROF and the pair fit; `TOPLANE=`
  and `TOSUB=` the plane and the image minus it. On a frame with a gradient of 2 sigma per
  1000 px, one level left 27 % of the frame as sky and came out 0.6 sigma low; the plane
  keeps 91 % and recovers the level and both slopes within their errors.
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
