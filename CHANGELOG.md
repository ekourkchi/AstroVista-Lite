# Changelog

Every release of AstroVista Lite (PyPI: `AstroVista`). Releases are tagged on GitHub; the
newest is at the top.

## 1.1.0rc1 (not released yet)

- **The buffers in tabs.** Above the buffer list: *All*, *Images* (the project's images,
  masks and auxiliary files), *Model* (the fit kept: its model, residual and sky), *Mean*
  (the averaged fit's mean model, residual, spread and sky), *Fit 1*, *Fit 2*, ... (each
  averaged fit's own model, residual and sky), the models picked in the models table, and
  the same for the pair fit (*Pair model*, *Pair mean*, *Pair fit 1*, ...). Showing a
  buffer brings its tab forward.
- **A picture of a region.** View ▸ Picture of a region… (Shift+S, the camera in the tool
  bar): drag a box over the image and it is saved as a PNG as shown (the scale, colours,
  isophotes, masks and marks), and copied to the clipboard; a click takes the whole view.
- **Harmonics: none by default, the common ones searched.** COS3X and COS4X are 0 by
  default in ELLIPROF (every mode) and the pair fit. The *Harmonic search* tries the ones
  most used, COS3X=-1 and COS4X=1, against none (`HARMONICS=COMMON`, the default); *Every*
  (`HARMONICS=AUTO`) tries all 18 combinations, *None* none.
- **Projects.** A project is one field: the galaxy image, its masks and auxiliary images,
  all of one size, with the sources, fits and models made from them. Opening another file
  adds it to the project (a mask, a weight map, another HDU); a file of another size is
  another field, and AstroVista Lite offers to start a new project for it. File ▸ New
  project and File ▸ Close project close everything (images, buffers, mask layers, models,
  source runs, fits; the session is saved first); the window's title names the project.
  Terminal: `project`, `project new [NAME]`, `project close`, `project name NAME`.
- **Several HDUs at once.** Opening a file with several images lists them; choose several
  (or *Open all*) and each goes into a buffer of its own.
- **FITS by content, whatever the name.** Images and masks are read when the file is FITS,
  with any name or none, gzip-compressed or not (a gzipped legacy bitmap mask too).
- **The companion's core, seen and chosen.** Pair fit ▸ 3. Fitting masks has *Companion's
  core* in every mode: *Off* (nothing left out around the other galaxy's centre, the
  automatic settings' choice), *Auto size, capped at* (CORECAP, or JOECOREMAX in Joe's
  mode) or *Fixed radius* (the new CORERADIUS=). The cores the settings shown would leave
  out are drawn on the image as you change them (dashed and filled, magenta; Layers: *Pair
  fit: cores left out*), with the isophotes, and a note gives their size and what set it.
  After a fit, the cores drawn are those the fit used (in Joe's mode the cores drawn were the
  other method's).
- **The pair fit's mean and spread: only the fits averaged.** A repeat (PARALLEL, RUNS) is
  left out of the mean, the spread and the per-fit buffers when it kept a model after a
  failed fit (and the fit kept did not), a galaxy was fitted short of 0.8 x its R1 in the fit kept, or its residual is
  over 1.05 x the kept's, as for one galaxy; the report and the Search table say why.
- **Add the sky image** is ticked by default (ELLIPROF and Pair fit).
- **Close all closes the mask layers too**, and what belonged to the images (sources, fits,
  skies, the fit tables); the session is saved first.
- **The Layers fold.** The ▾ beside Layers folds them to their title and gives the buffer
  list the room (remembered).
- **A in cyan, B in orange, your choice.** The pair fit's A and B each have a row in Layers;
  click its colour box to change the colour (the card, the drawing and the plots follow,
  and the colour is remembered, as are the other overlays' colours).
- **Pick a centre by clicking inside the source.** For A, B and ELLIPROF, a click anywhere
  inside a found source (its segment, or its Kron ellipse) takes that source's centre
  (*Snap to sources*; Alt-click: the point clicked).
- **Source centres as small crosses** (Layers: Source centres, and the Source finder's
  *Show their centres*), as in AstroVista.
- **Your work is kept for next time.** The buffers (with their names, tags and model
  parameters) and the mask layers are saved to a session folder (`.temp` beside the
  settings, or one chosen in File ▸ Session folder…) when buffers are closed, when the
  window closes, and every 2 minutes when something changed. At the start, AstroVista Lite
  offers to continue; File ▸ Restore the last session and the `session` command
  (`session save`, `session restore [FOLDER]`) do it at any time. The save before the last
  is kept as `.temp.previous`.
- **Every model buffer has its parameters.** Each buffer an iterative ELLIPROF or pair fit
  makes (the fit kept, the mean, each fit averaged, the spread, a pair call's input and
  model) has Model parameters… on its right-click menu.
- **The sky as a buffer.** *Add the sky image* (ELLIPROF and Pair fit; `TOSKY=M`, and
  `models N SHOW=… TOSKY=1`) puts the sky subtracted, the level plus the plane when a sky
  tilt was taken out, in a buffer, for the fit kept, the mean and each fit averaged.
- **Fix: the averaged single fits and a sky tilt.** When the fit with the sky tilt taken out
  was kept, the other fits averaged were made on the image with the tilt left in; they are
  now made on the same image as the fit kept.
- **The segmentation's inverse as a mask layer.** With the segmentation image (TOSEG), the
  Source finder makes the mask layer `segK` (K: the run): every source's pixels masked
  (`SEGMASK=0`, or the panel's box unticked: not made).
- **A measured sky is used at once.** Measure sky (ELLIPROF and Pair fit) makes its median the
  base level (in ELLIPROF, subtracted) without Use this sky, which takes it back after a typed
  value.
- **ELLIPROF and Pair fit in the same order.** Image and Fit at the top, then 1. Sky, 2. The
  galaxy (the two galaxies), 3. Fitting masks, 4. Fitting (fits to average, harmonics,
  residual sources, then the rest), 5. Run and 6. Residual sky. ELLIPROF's Fits to average
  and Search every harmonic are now shown in Automatic too, and Automatic follows them.
- **Close all buffers.** `close ALL`, and Close all in a buffer's right-click menu (it asks
  first).
- **The pair fit's options for one galaxy.** ELLIPROF's Fit: One elliprof call (as before),
  Automatic (the pair fit's recipe for one galaxy) or Iterative (each setting shown). With
  no companion there is no R1 growth: the galaxy is fitted out to R1 at once, and to a
  smaller R1 when the isophotes fail the pair fit's checks (START, FACTOR and TRYFACTOR are
  refused); R1 and NR automatic; the sky factor refined from the residual sky and a sky
  tilt taken out; fixed and free centres searched; the
  harmonics searched; the outermost isophotes checked; N fits averaged with the mean
  residual and each fit's own; residual sources masked. Measure (sky by segmentation) and
  Cancel in the panel. The plain `elliprof` command is unchanged unless one of these
  keywords is given.
- **ELLIPROF's fitting progress.** While an iterative fit runs, a table under Fit isophotes
  lists every elliprof call (its step: search, harmonics, sky tilt, residual sources or
  averaging; R0 to R1, NR, passed or why it failed) and every finished fit in bold with its
  score, the best so far in green; the image shows the isophotes of the latest fit.
- **ELLIPROF's models.** Every model of an iterative fit is listed (report, the Models table,
  `models N`) with its parameters (centre, FIXCTR, COS3X, COS4X, TIE, R0, R1 asked and
  reached, NR, sky, tilt), its residual and its core's. The residual is measured between
  1.15 × R0 (the core is often disturbed) and 1.1 × R1, R0 and R1 as asked and the same for
  every model, in rings 20 % apart: each ring's RMS over its own noise (from the image, the
  galaxy's photon noise included), every ring counted the same, so a bad middle is not
  hidden by the faint outskirts. The fit kept is the lowest residual; a model whose core is
  over twice the cleanest is never kept nor averaged. A tab per step in the panel; any model
  put in buffers with its residual, the selected ones averaged (`models N SHOW=`,
  `AVERAGE=`); right-click a model, or a buffer made from one, for its parameters and the
  elliprof command that makes it.
- **ELLIPROF tries harder (SEARCH=FULL, the default).** The search also fits with TIE off
  and with a larger R0 (out of a masked core: isophotes fitted on masked pixels turned the
  fits wrong), then moves a fixed centre by 1/2, 1/4 and 1/8 px to the lowest residual,
  and after the harmonics tries fewer and more isophotes. The averaged fits never put R0
  into a masked core, and a fit with a residual over 1.05 × the kept's or that stopped
  short of 0.8 × its R1 is left out of the mean. On UGC 12517 (R0=9 inside a masked core of
  10 px, TIE=-3) the kept fit's residual fell from 3.37 to 1.51 and its core from 5.95 to
  1.11 (free centres, TIE off, R0 13.5, NR 28), in 6 s. `SEARCH=BASIC` searches only the
  centres (TRYFIX).
- **ELLIPROF: the sky analyser and the harmonic analysis.** The pair fit's sky section
  (segmentation settings, Measure, the verdict, histogram and cell map, Use this sky, the
  factor and its refinement, and after a fit the residual on the sky pixels with the next
  factor) is ELLIPROF's too. Automatic fits every COS3X and COS4X on the fit kept and
  alternates with the search over centres and growth until the harmonics stay (3 rounds
  at most); the report lists every combination's residual and R1, and says how close the
  ellipses come to crossing (none may reach more than 2 % out of the next). Profile plots
  of the cos 3θ amplitude and of both angles.
- **Sources in the residual.** ELLIPROF and the Pair fit can find the faint sources a fit's
  residual shows (globular clusters, stars and small galaxies on the galaxies, which the
  first source finding cannot separate from their light), mask them and fit again:
  Residual sources in both panels (`RESIDMASK=n`, `RESIDTHRESH=8`, `RESIDGROW=2`,
  `RESIDCORE=10`). Each round's catalogue is a source run, and the pixels the layer
  `residsources` (role fit). On 18 simulated pairs the first source finding masked none of
  the clusters and faint stars on the galaxies; at 8 sigma every source found in the
  residual was real, at 3 sigma over half were peaks of the SBF and masking them made the
  models worse.
- **Each tool's image.** The Source finder, ELLIPROF and the Pair fit each have an Image box:
  the image shown (the default) or any buffer, used whatever is shown.
- **A clock.** The status bar shows how long the command running has taken, then how long it
  took; the terminal says it too.
- **Cancel.** The Pair fit tool's Cancel stops a fit as soon as the elliprof calls running
  then end and keeps nothing of it: no models, residuals, means or views; the buffers and
  the last pair result stay as they were. (Stop still keeps the last cycle that passed.)
- **R1 grows by 1.2.** The growth factor of R1 is 1.2 by default (it was 1.5): START=AUTO,
  each galaxy's "× factor", and Joe's mode (it was 1.3); the automatic search tries 1.2
  and 1.35 (it tried 1.5 and 1.25). On 18 simulated pairs, the automatic settings' SBF
  amplitude came within 1.5 % of the truth for half the galaxies (1.7 % before; 17 better,
  10 worse, not significant) and 9 fits ended stable instead of 6; each fit takes a few
  more elliprof calls.
- **Joe's mode.** Settings ▸ Joe's mode (`METHOD=JOE`) fits a pair the way Joe Jensen
  does it by hand: each galaxy's centre left out while the other is fitted, the smaller
  galaxy first and not far out, then the two in turn, each R1 grown only as far as its
  outer ellipses (fitted with free centres) do not move towards the other. No search and
  no sky refinement: seconds instead of minutes. On 18 simulated pairs its errors were
  close to the automatic settings' with worse worst cases; on N6768, N4825 and C328014 it
  left 1.3 to 3 times more large-scale structure in the residual. A quick first look.
  Its settings, in its own box of the panel: the galaxy fitted first (`FIRST=`), the R1
  tried each cycle (`JOEGROW=`), the centre left out (`JOECORE=`, `JOECOREMIN=`,
  `JOECOREMAX=`), what counts as a pull (`PULLPX=`, `PULLFRAC=`, `PULLRUN=`, `PULLR0=`, also
  used by the pull check), the centres of the model kept, the cycles, the smoothing and the
  model beyond R1. Each galaxy's card sets its R0, R1, first R1, NR and harmonics.
- **Joe's mode: outer ellipses.** At the end of the fit, a galaxy whose 3 outermost ellipses
  leave the trend inside them (`SHAPETOL=0.12`, ellipticity and angle together, compared
  with both the median and the log-radius trend of the 4 inside), or whose model rises or
  is brighter than the light beyond R1, is refitted with coarser isophotes or an R1 just
  inside the bad ellipse; the largest clean fit is kept and both galaxies are fitted once
  more. On 18 simulated pairs the SBF amplitude in the residual came within 1.6 % of the
  truth for half the galaxies (2.8 % without; the automatic settings 1.7 %), and 6 fits
  ended stable instead of 1. A guard applied while the galaxies grow, or to every
  isophote, cut real changes of shape short (C328014's A to 36 px) and was dropped.
- **The pull check.** Every complete pair fit ends with Joe's test: each galaxy fitted once
  more with free centres on the image minus the other's model, and the report says whether
  its outer ellipses move towards the other galaxy and from which radius (`PULLCHECK=0`:
  off). On C328014 it shows that B's ellipses move towards A from Rmaj 68 of its 83.
- **Repeat runs, averaged.** `RUNS=n` (now set by Fits to average, PARALLEL=) fits the fit kept
  again n − 1 times, each with every galaxy's R0 and R1 changed at random by 10–15 %
  (`JITTER=0.15`, `SEED=`), and averages the complete runs: buffers get the mean A, B and
  A + B models, the mean residual (the original image minus the mean sky minus the mean
  A + B, with any sky tilt a fit removed counted in its sky; the mean sky is a buffer
  then), the spread of the models and every run's own model and residual; the report gives
  the mean sky and every run. A later fit of the image without a mean clears these
  buffers.
- **Every harmonic searched.** `HARMONICS=AUTO` (Search every harmonic) tries every
  `COS3X` and `COS4X` for A, then for B, on the fit kept, and keeps the stable fit with
  the smallest residual.
- **Star wings and spikes.** `WINGS=1` masks the light of bright sources beyond their
  masks (a star's wings, its diffraction spikes) that the residual of a first pass shows,
  joined to their masks and never near a galaxy's centre.
- **Pairs that failed at their first fit.** A fitting mask that covers a galaxy's own
  centre (NGC 4825: a mask made to fit A alone hid B) is lifted there, with a note; when
  the innermost isophotes (within 10 px) fail, the fit starts from a larger R0 (NGC 5090:
  its isophotes at 3 and 4.5 px crossed); a close pair whose first radius (~10 px) is too
  small for ELLIPROF is started further out. All four such failures (two real, two
  simulated) now fit.
- **Fits to average.** `PARALLEL=N` (Pair fit ▸ Fits to average, 5 by default on a
  computer with 6 cores or more) makes N fits and averages all of them: the fit kept and
  N − 1 repeats with R0 and R1 varied, N at a time, each refining its own sky. (It first
  set only how many variants of the search ran at once, with a mean of those within 5 % of
  the best score, often 2 of 8; that mean is gone.) `RUNS=` sets another number of fits.
- **Harmonics in ELLIPROF's terms.** Each galaxy's `COS3X=` and `COS4X=` (2 each isophote's
  own, the default; 1 the median; 0 none; COS3X −2, −1, −3 the 6th order) in the panel
  and the command, and `TRYCOS3X=`/`TRYCOS4X=` to try other values on the fit kept: A's
  with B as it is, then B's with A's best, kept when the residual falls by more than 0.01
  of the noise. On 36 hard simulated pairs (boxy and disky galaxies with |a4| 0.01–0.04,
  overlaps, halos filling the frame, bright stars, small companions in a giant's halo,
  galaxies cut by the edge) ELLIPROF's default already followed the boxy and disky
  shapes; the search changed 7 scenes, mostly next to a bright star, and is not part of
  the automatic settings.
- **The pair's isophotes while it runs.** Each galaxy's latest isophotes are drawn as
  every ELLIPROF call ends; galaxy A is magenta (it was cyan, the colour of the source
  finder's ellipses, which hid it).
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
