# AstroVista Lite

[![Tests](https://github.com/ekourkchi/AstroVista-Lite/actions/workflows/tests.yml/badge.svg)](https://github.com/ekourkchi/AstroVista-Lite/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/AstroVista.svg)](https://pypi.org/project/AstroVista/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ekourkchi/AstroVista-Lite/blob/main/LICENSE)

**Website, tutorial and screenshots: https://ekourkchi.github.io/AstroVista-Lite/**

![AstroVista Lite with the UGC 12517 example: the isophotes fitted, the found sources masked](https://raw.githubusercontent.com/ekourkchi/AstroVista-Lite/main/docs/images/hero.png)

The lite edition of **AstroVista**, an astronomical image analysis studio in
development: simple enough for students and classrooms, useful for everyday work. In one
window:

- **ELLIPROF isophote fitting** of galaxies, through the
  [`elliprof`](https://github.com/ekourkchi/elliprof) package: the profile table and plots,
  the isophotes drawn on the image, and the model and residual images.
- **The Source finder**: the stars and galaxies in the image, measured (position, shape,
  flux, magnitude, size) and drawn on the image, with a sortable catalogue, plots,
  numbered runs and the segmentation image. The model, residual and segmentation images
  go into the buffers chosen (only free buffers are offered unless asked otherwise).
- **For teaching**: a welcome screen; built-in documentation, Help ▸ Documentation (F1),
  with a *First lesson* (about thirty minutes on the example galaxy), recipes, image
  arithmetic in detail, a FAQ, troubleshooting and every command; a **?** next to each
  tool and each setting, all explained in Help ▸ Documentation ▸ *Every setting
  explained*; Help ▸ Tips and Help ▸ About.
- **Panels** are resized by dragging their edges, folded to a narrow strip with the
  button in their title bar (or a double-click on it, or the buttons in the status bar),
  and popped out as windows of their own (⧉) for another monitor, then put back (⤓); each
  panel docks only in its own place. View ▸ Reset the layout restores the standard
  layout.
- **The galaxy's centre from the sources**: pick it on the image (a click inside a source
  snaps to its centre), or double-click a source in the table.
- **Layers**: isophotes, sources, the ELLIPROF centre (a small cross) with R0 and R1,
  contours, a crosshair, and any number of mask layers: each shown or hidden and in its
  own colour (click its colour box); masks with their own opacity. The compass and scale
  bar are always shown (for images with a WCS).
- **The crosshair** follows the mouse while its tool (or its layer) is on, and stays where
  it is when another tool is chosen; Alt with the arrow keys nudges it. The Inspector shows the
  pixels around it.
- **Masks**: drawn with circle, ellipse, box, polygon and brush tools, and an eraser (Alt or Shift
  unmasks; an ellipse, box or polygon can be moved, resized and rotated before it is applied; Ctrl-click, ⌘-click on macOS, removes the masked patch under the cursor);
  made from the found sources (all, the selected one, or those matching a condition
  such as `MAG_AUTO < 20`), from their segmentation (grown), from non-finite
  pixels or a range of values, from a buffer, or loaded from FITS and legacy `.dmask`
  files; grown, shrunk, inverted, combined (union, intersection), copied, renamed, saved
  and deleted, with undo and redo (right-click the Layers panel). The ticked layers, and any image buffer used as a mask
  (where it is 0 or NaN), are masked in the fit.
- **Arithmetic on images**, as in the full AstroVista: `[3] = [1] - [2]`,
  `[4] = median([1], [2], [3])`, `[5] = [1] - median([1])`, `[6] = div([1], [2], 0)`,
  `[7] = [1] * mymask` (a mask layer: NaN where it masks), comparisons, `where`, `clip` and
  the usual functions (`help calc`).
- **A terminal**: every action that changes data is a command, echoed as it runs,
  which can be typed, repeated, saved and run again from a file; `ls` lists in colour,
  with folders and FITS files one click away. Programs on the PATH run there too (`ls -l`, `ll`, `grep ...`; `!` forces the shell; Ctrl+C stops one). The
  *Terminal* button in the status bar folds or unfolds it. Tab completes commands,
  keywords, mask names, buffers and file names.
- **Files**: FITS images are recognised by their contents, whatever they are called
  (gzip-compressed too); one image of a multi-extension file opens as `name.fits[sci]`,
  `[1]` or `[sci,2]` (`fitsinfo` lists them); File ▸ Open recent; legacy BITPIX = 1
  `.dmask` bitmaps open as mask layers.
- **Profiles**: the ELLIPROF profile plotted against R or R¼, with the de Vaucouleurs law
  fitted and over-plotted (Re, Ie, and the scatter about the law).
- **Contours**, their own tool with DS9's settings: the number of levels, the scale and
  limits they are spread over (or the display's), the smoothing (Gaussian or block), the
  editable list of values, and the line colour, width and dashes.
- **The Inspector** (the first tab of the right-hand panel, beside Tools): the pixel under the cursor, its value and sky
  position, a magnified view in the display's colours, and the statistics around it.
- **The display**: limits from percentiles, ZScale, ZMax, min-max or typed, or dragged
  on the histogram; linear, log, power, square-root, squared, asinh, sinh and
  histogram-equalised stretches; twelve colour maps, inverted if wanted; contrast and
  bias (drag with the right mouse button); a colour bar; zoom, 1:1, North up, rotation
  and mirroring; *Go to* a pixel or RA/Dec; the histogram and blinking (← and → between
  the buffers chosen, with the same scale if wanted) switched on in the display bar above
  the image, which folds away for more room; the FITS header.

AstroVista Lite is published on PyPI as **AstroVista** (`pip install AstroVista`;
`pip install astrovista` is the same, since PyPI ignores capitals). The full AstroVista will
follow there when it is ready.

## Installing

```sh
pip install AstroVista
astrovista --example        # the UGC 12517 example, shipped with the package
astrovista image.fits       # or your own images
astrovista -i galaxy.fits other.fits -m galaxy.dmask   # images into buffers 1, 2; a mask layer
```

`-i` loads images into the buffers in the order given (`name.fits[sci]` picks one image
of a file) and `-m` loads masks (FITS or `.dmask`, 0 = masked) as mask layers, ticked for
the fit; both can be repeated. `astrovista --help` lists every option.

AstroVista Lite is on PyPI for **Linux and macOS** (Python 3.10 or later; on Linux,
glibc on x86_64 or aarch64). Everything needed comes with `pip install`; there is nothing
else to install. The commands `astrovista`, `astro-vista` and `astrovista-lite` are the
same program, and so is `python -m astro_vista`.

A virtual environment keeps it apart from other Python programs:

```sh
python3 -m venv ~/astrovista-env
source ~/astrovista-env/bin/activate
pip install AstroVista
```

**On Linux**, the window is drawn by Qt, which needs a few libraries every desktop already
has. On a minimal system (a fresh server, WSL or a container) where Qt reports that it
*could not load the Qt platform plugin "xcb"*, install them:

```sh
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1 libfontconfig1 libdbus-1-3   # Debian, Ubuntu
sudo dnf install xcb-util-cursor libxkbcommon-x11 mesa-libEGL mesa-libGL fontconfig dbus-libs  # Fedora, RHEL
```


## The example

[`examples/u12517`](https://github.com/ekourkchi/AstroVista-Lite/tree/main/examples/u12517) holds the example image, its `.dmask` mask and
`session.txt`, the commands that reproduce elliprof's published fit:

```sh
cd examples/u12517
astrovista --run session.txt
```

The same image and mask ship inside the package, for `astrovista --example`.

## Commands

```
rd 1 galaxy.fits[sci]              read an image (an HDU number or name) into buffer 1
sources 1 DETECT_THRESH=2.5        find the sources (stars and galaxies)
centre 1 SOURCE=12                 ELLIPROF's centre from a found source
mask new mymask 1                  a mask layer the size of buffer 1
mask sources mymask SCALE=1.5      mask the sources, except the galaxy at the centre
mask circle mymask 812 440 12      mask a circle (ERASE unmasks)
mask sources mymask WHERE='FWHM_IMAGE < 3'
mask grow mymask 2                 grow it by 2 pixels; undo takes it back
sky 1 MASK=mymask                  a clipped median of the pixels left
elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 MASK=mymask TOMODEL=3 TORESID=4
[4] = [1] - [2]                    arithmetic on buffers
wd 3 residual.fits                 write a buffer
run session.txt                    run a file of commands
```

Image commands in the style of VISTA and MONSTA, written for AstroVista Lite with NumPy and
SciPy: `ac sc mc dc`, `ai si mi di`, `log ln sqrt inverse round`, `mn max min abx print`,
`shift rotate flip bin mash`, `smooth filt zap clip skysub`, `fft noise
gauss`, `ch chead`, and variables (`set xc=567`, then `X0=xc` in any command).

`help` lists them all, and `help mask` the mask commands. The website explains
[how the terminal works](https://ekourkchi.github.io/AstroVista-Lite/terminal.html) (how a
line is written, buffers, coordinates, variables, programs, the keyboard, command files)
and gives the
[full command reference](https://ekourkchi.github.io/AstroVista-Lite/reference.html): every
command with its keywords and an example. Both are also in Help ▸ Documentation.

## Using it

1. **File ▸ Open example** loads an HST image of UGC 12517 and its mask with the
   parameters of elliprof's published example. Press **Fit isophotes**. X0, Y0, R0 and R1
   can be picked on the image.
2. The **Source finder** tab finds the sources. Click a source on the image, or a row in the
   **Sources** table, to select it.
3. In the **ELLIPROF** tab, pick the centre on the image (it snaps to a found source), and
   **Mask the found sources** makes a mask layer of every other source. The mask
   tools (circle, box) add to the mask chosen in the Layers list; Shift unmasks.
4. **File ▸ Export** writes the profile (CSV), the isophotes (DS9 regions), the sources
   (CSV, comma separated with every column aligned; also *Save CSV…* in Results ▸ Sources)
   and the profile and source plots as PNG figures made with matplotlib (also *Save PNG…*
   beside each plot); **File ▸ Save the commands** writes the session as a command file.

Coordinates: X0, Y0 and the mask commands use ELLIPROF's coordinates, in which the first
pixel spans 0 to 1 (its centre is at 0.5). The status bar shows these, the pixel's column
and row (from 1), its value and its sky coordinates.

## Tests

```sh
pip install -e '.[test]'
pytest -q
```

## Licence

MIT (see [`LICENSE`](https://github.com/ekourkchi/AstroVista-Lite/blob/main/LICENSE)). The libraries it installs keep their own licences, listed
under *Credits*.

## Citing

If AstroVista Lite helps your work or your teaching, please cite its GitHub repository:

> Kourkchi, E., *AstroVista Lite*, https://github.com/ekourkchi/AstroVista-Lite

(GitHub's "Cite this repository" button gives the same from [`CITATION.cff`](https://github.com/ekourkchi/AstroVista-Lite/blob/main/CITATION.cff).)
For the measurements, please also cite elliprof (isophotes), and SEP and SExtractor (the
Source finder), listed below.

## Credits

AstroVista Lite is by Ehsan Kourkchi.

**John Tonry.** ELLIPROF, the isophote fitting at the heart of AstroVista Lite, was
originally developed by John Tonry as part of MONSTA; it is used here through the elliprof
package, which compiles his original code unchanged. The terminal follows the command
language of MONSTA, John Tonry's image analysis and plotting system, which joined Vista
(originally by Richard Stover and Tod Lauer, with image display by John Tonry) and Mongo (by
John Tonry). The commands of AstroVista Lite are new code written in that style.

It is built on:

- **elliprof**: John Tonry's ELLIPROF as a Python package
  ([github.com/ekourkchi/elliprof](https://github.com/ekourkchi/elliprof); the package is
  MIT, the original ELLIPROF sources keep their own terms).
- **SEP** (Source Extraction and Photometry), by Kyle Barbary and contributors
  ([github.com/sep-developers/sep](https://github.com/sep-developers/sep), LGPL-3.0):
  K. Barbary 2016, *Journal of Open Source Software* 1(6), 58. The Source finder uses it.
- **SExtractor**, whose algorithms SEP implements: E. Bertin & S. Arnouts 1996,
  *Astronomy & Astrophysics Supplement Series* 117, 393. Please cite SEP and SExtractor
  when you publish measurements from the Source finder.
- **Qt for Python** (PySide6, LGPL), **pyqtgraph** (MIT), **matplotlib** (the PNG
  figures; matplotlib licence, BSD-style), **NumPy**, **SciPy** and **Astropy** (BSD).
- The example image of UGC 12517 (Hubble Space Telescope, WFC3/IR, F110W) and its mask
  come from the elliprof package's example.
- The command language follows MONSTA (Vista and Mongo): the names and style are those of
  John Tonry, Richard Stover and Tod Lauer; the commands are new code, written for
  AstroVista Lite.

The same list is in Help ▸ Documentation ▸ Credits.
