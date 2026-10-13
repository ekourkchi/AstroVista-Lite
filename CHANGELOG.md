# Changelog

Every release of AstroVista Lite (PyPI: `AstroVista`). Releases are tagged on GitHub; the
newest is at the top.

## Unreleased

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
