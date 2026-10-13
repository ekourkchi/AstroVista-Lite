---
title: 'AstroVista Lite: isophote fitting, source finding and masks for astronomical images in one window'
tags:
  - Python
  - astronomy
  - galaxies
  - surface photometry
  - isophotes
  - image analysis
  - education
authors:
  - name: Ehsan Kourkchi
    orcid: 0000-0002-5514-3354
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: Department of Electrical and Computer Engineering, University of California, Davis, CA 95616, USA
    index: 1
  - name: Department of Physics and Astronomy, Utah Valley University, Orem, UT 84058, USA
    index: 2
date: 13 October 2026
bibliography: paper.bib
---

# Summary

AstroVista Lite is a desktop program for the surface photometry of galaxies. In one window
it displays FITS images, fits elliptical isophotes to a galaxy with ELLIPROF, finds and
measures the stars and galaxies around it, builds masks of what is in the way, and does
arithmetic on whole images. Every action that changes data is also a command, echoed in an
embedded terminal whose language follows MONSTA, John Tonry's image analysis system; a
session can therefore be read, edited, saved and run again as a plain text file. The
program is written in Python on Qt for Python [@pyside6] and PyQtGraph [@pyqtgraph]. It
installs with `pip install AstroVista` on Linux and macOS, includes an example (a Hubble
Space Telescope image of the elliptical galaxy UGC 12517 with its mask), and documents
itself: a first lesson, a terminal guide, a reference of every command whose examples are
run by the test suite, and an explanation behind each setting.

# Statement of need

Isophote fitting [@Jedrzejewski1987] remains a basic measurement of galaxy structure and
an essential step of surface brightness fluctuation (SBF) distances [@Tonry1988], where a
smooth model of the galaxy is subtracted before the fluctuations are measured. ELLIPROF,
originally developed by John Tonry as part of MONSTA, has played that role from the I-band
SBF Survey [@Tonry1997] to studies with HST [@Jordan2004; @Mei2005; @Jensen2021]. It is
now available as a Python package [@elliprof] that compiles the original code unchanged.

What surrounds the fit is often harder than the fit itself: the centre and radii must be
chosen, the sky measured, and every star and background galaxy masked. These steps are
usually spread over a display program, a source extractor, scripts that write masks in the
legacy `.dmask` format, and the fitting program. Students learning surface photometry, and
researchers preparing many galaxies, both lose time moving between them.

AstroVista Lite brings these steps together. The centre can be picked on the image (a click
snaps to a detected source); the source finder, built on SEP [@Barbary2016], which
implements SExtractor's algorithms [@Bertin1996], masks every source but the galaxy in one
step, or those matching a condition on the catalogue; mask shapes can be moved, resized and
rotated before they are applied, and masks are read and written as FITS or `.dmask`. The
profile is plotted against $R$ or $R^{1/4}$ with de Vaucouleurs' law [@deVaucouleurs1948]
fitted, and the model and residual images are compared by blinking. Because every step is
also a command, a session doubles as a reproducible record and as an exercise that a
teacher can hand out.

# State of the field

General image viewers such as SAOImage DS9 [@Joye2003] display images and draw regions but
do not fit isophotes or write masks for ELLIPROF. Isophote fitting is available in IRAF
[@Tody1986] and in Photutils [@Bradley_photutils], as libraries driven by scripts;
two-dimensional model fitting programs such as GALFIT [@Peng2002] address a different
problem. SEP and SExtractor detect and measure sources but leave masking and fitting to the
user. AstroVista Lite does not replace these tools: it uses ELLIPROF and SEP unchanged and
contributes the interactive layer around them (display, masks, a reproducible command
language, documentation) aimed at everyday measurements and at teaching.

# Design and validation

A session holds numbered image buffers, mask layers, source-finder runs and fits. The
window and the terminal act on it only through commands, so the graphical interface and the
command language cannot disagree. Arithmetic is evaluated pixel by pixel from a parsed
expression (never run as Python), with division by zero giving NaN; image commands
(`shift`, `smooth`, `zap`, `clip` and others) are new code written with NumPy and SciPy
[@Harris2020; @Virtanen2020]; FITS input and output and world coordinates use Astropy
[@Astropy2013; @Astropy2018; @Astropy2022], and exported figures use Matplotlib
[@Hunter2007].

The fit of UGC 12517 made through AstroVista Lite's command agrees with elliprof's own
published example command in every profile column, to the precision elliprof writes
(relative differences below $6\times10^{-6}$). The test suite (65 tests, run on Linux and
macOS with Python 3.10 to 3.13) covers the commands, masks, arithmetic, the window and
its documentation; one test runs every example on the website, in order, on the example
image.

# AI usage disclosure

TODO (required by JOSS; to be completed by the author before submission): the generative
AI tools and models used, with their versions; where they were used (code, tests,
documentation, the website, this paper); the nature of the assistance (code generation,
refactoring, drafting); and the statement that the human author reviewed, edited and
validated all AI-assisted outputs.

# Acknowledgements

ELLIPROF and the MONSTA command language are the work of John Tonry; MONSTA joined Vista
(originally by Richard Stover and Tod Lauer, with image display by John Tonry) and Mongo
(by John Tonry). The source finder uses SEP by Kyle Barbary and contributors. The example
image of UGC 12517 comes from the elliprof package's example.

# References
