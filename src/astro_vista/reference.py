"""The reference of the terminal: how a command line is written, and every command with
its syntax, its keywords and an example.  One source for Help ▸ Documentation and for the
website (docs/terminal.html and docs/reference.html, made by docs/build.py), so that the
two never disagree; the tests check that every command the terminal knows is here and
that every example runs.

The names of many commands are those of MONSTA, John Tonry's image analysis and plotting
system (the union of Vista and Mongo).  The commands of AstroVista Lite are new code: the
names and the style are MONSTA's, the options are AstroVista Lite's own, as written here.
"""

from __future__ import annotations

import html
from typing import List, NamedTuple, Tuple


class Command(NamedTuple):
    names: str  # the command, or several that share one entry ("ac sc mc dc")
    syntax: str  # how it is written; [ ] = optional, | = one of
    what: str  # what it does
    keywords: Tuple[Tuple[str, str], ...] = ()  # (KEYWORD or word, meaning and default)
    example: str = ""  # lines that run as written (the tests run them, in order)
    monsta: bool = False  # MONSTA has a command of this name


# the commands of MONSTA's command loop that AstroVista Lite also has (by name)
MONSTA_NAMES = set(
    (
        "rd wd tv buf close ac sc mc dc ai si mi di log ln sqrt inverse round mn max min abx "
        "print shift rotate mash smooth filt zap clip skysub fft noise gauss ch set typ mask "
        "sky elliprof help"
    ).split()
)


def _c(names, syntax, what, keywords=(), example=""):
    first = names.split()[0]
    return Command(names, syntax, what, tuple(keywords), example, first in MONSTA_NAMES)


GROUPS: List[Tuple[str, str, List[Command]]] = [
    (
        "Images and buffers",
        "An image lives in a numbered buffer: 1, 2, 3, … The list on the left of the window "
        "shows them, each with the command that made it.",
        [
            _c(
                "rd",
                "rd N file[ext]",
                "Read a FITS image into buffer N. The file may be gzip-compressed and may have "
                "any name: FITS is recognised by its contents. Without [ext], the first image "
                "in the file is read.",
                (
                    ("[n]", "the HDU with that number, 0 being the primary: rd 1 frame.fits[1]"),
                    ("[NAME]", "the image extension called NAME (its EXTNAME): [sci]"),
                    ("[NAME,v]", "NAME with version v (EXTVER): [sci,2]"),
                ),
                "rd 1 galaxy.fits\nrd 2 frame.fits[sci]\nrd 3 frame.fits[sci,2]",
            ),
            _c(
                "fitsinfo",
                "fitsinfo file",
                "List what a FITS file holds: each HDU by number and name, with the size of "
                "its image or the rows and columns of its table.",
                (),
                "fitsinfo frame.fits",
            ),
            _c(
                "wd",
                "wd N file",
                "Write buffer N as a FITS file (32-bit floating point), with its header. The "
                "commands that made the buffer are kept as HISTORY cards.",
                (),
                "wd 1 copy_of_galaxy.fits",
            ),
            _c("tv", "tv N", "Show buffer N in the image window.", (), "tv 1"),
            _c(
                "copy",
                "copy N M",
                "Copy buffer N, pixels and header, into buffer M.",
                (),
                "copy 1 5",
            ),
            _c(
                "close",
                "close N | ALL",
                "Remove buffer N, or every buffer with ALL.",
                (),
                "close 5",
            ),
            _c(
                "buffers buf",
                "buffers   (or buf)",
                "List the buffers: number, name, size, and the line that made each.",
                (),
                "buffers",
            ),
            _c(
                "tag",
                'tag N "name"',
                "Give buffer N a name of your own, shown in the buffer list. Any command or "
                'formula that makes a buffer also takes tag="name" at the end of its line.',
                (),
                'tag 1 "UGC 12517, F110W"\n[6] = [1] * 2 tag="twice"',
            ),
        ],
    ),
    (
        "Arithmetic",
        "Formulas over whole images, worked out pixel by pixel. Help ▸ Documentation ▸ "
        "Image arithmetic explains them in detail.",
        [
            _c(
                "[M] = BM =",
                "[M] = expression     (or BM = expression)",
                "Put the result of a formula into buffer M: [3] = [1] - [2] subtracts buffer 2 "
                "from buffer 1, pixel by pixel. A formula holds buffers ([n] or Bn), numbers, "
                "variables, mask layers by name (1 where good, NaN where masked), + - * / % "
                "and ^ or ** for powers, comparisons (1 or 0), and functions. Dividing by 0 "
                "gives NaN, never infinity.",
                (
                    (
                        "mean median sum min max std",
                        "of several values: pixel by pixel; of "
                        "one buffer: one number over the whole image",
                    ),
                    ("div(a, b, fill)", "a / b, with fill where b is 0"),
                    ("where(c, a, b)", "a where c is true (not 0), b elsewhere"),
                    ("clip(a, lo, hi)", "the values limited to lo … hi"),
                    ("good(m), masked(m)", "1 where the mask layer m is good / masks"),
                    (
                        "abs sqrt exp log log10 …",
                        "functions of each pixel (help calc lists " "them all)",
                    ),
                ),
                "[2] = [1] - median([1])\n[3] = median([1], [2], [6])\n"
                "[4] = div([1], [2], 0)\n[7] = where([1] > 1000, [1], nan)",
            ),
            _c(
                "calc",
                "calc M EXPR='expression'",
                "The same as [M] = expression, written as a command with a keyword.",
                (),
                "calc 8 EXPR='[1] * 2.5'",
            ),
            _c(
                "ac sc mc dc",
                "ac N value   (sc, mc, dc)",
                "Add (ac), subtract (sc), multiply by (mc) or divide by (dc) a number, in "
                "place. The value may be a variable or an expression of variables.",
                (),
                "ac 8 100\nsc 8 100\nmc 8 2\ndc 8 2",
            ),
            _c(
                "ai si mi di",
                "ai N M   (si, mi, di)",
                "Add (ai), subtract (si), multiply (mi) or divide (di) buffer N by buffer M, "
                "pixel by pixel, in place in N. The buffers must have the same size; dividing "
                "by 0 gives NaN.",
                (),
                "ai 8 1\nsi 8 1",
            ),
            _c(
                "log ln sqrt inverse round",
                "log N   (ln, sqrt, inverse, round)",
                "Replace each pixel of buffer N by its base-10 logarithm (log), natural "
                "logarithm (ln), square root (sqrt), 1/x (inverse) or nearest whole number "
                "(round). Where the function has no value (log of 0 or less, square root of a "
                "negative number, 1/0) the pixel becomes NaN.",
                (),
                "copy 1 9\nlog 9\nround 9",
            ),
        ],
    ),
    (
        "Statistics",
        "Numbers about a buffer. Columns and rows are counted from 0. mn, max and min also "
        "keep their results as variables, for the commands that follow.",
        [
            _c(
                "stats",
                "stats N [MASK=a,b]",
                "The mean, median, standard deviation, minimum, maximum and number of pixels "
                "of buffer N, leaving out NaN pixels and those of the mask layers named.",
                (("MASK=", "mask layers to leave out, separated by commas"),),
                "stats 1\nstats 1 MASK=mymask",
            ),
            _c(
                "sky",
                "sky N [MASK=a,b]   ·   sky N SEGMENT [THRESH=0.5] [MINAREA=12] [GROW=2] "
                "[CELLS=8] [BACK_SIZE=0] [FWHM=0] [PAD=v] [MASK=a,b] [TOMASK=skymask] "
                "[TOMAP=M] [TOPLANE=M]",
                "The sky level of buffer N. Alone: the median of the pixels the masks leave, "
                "clipped at 3 standard deviations until it settles. With SEGMENT: every source "
                "found down to THRESH standard deviations is left out (grown by GROW pixels), "
                "and the sky is measured on the rest: its median, a clipped mean and sigma, the "
                "medians of CELLS × CELLS boxes, a plane fitted to them (the gradient, as an "
                "angle on the image from +x), an approximate sampling error and the spatial "
                "spread, kept apart. The verdict: usable_support, candidate_inspect_support "
                "(with its reasons) or no_estimate. The verdict is advice: the median is kept "
                "whenever there are sky pixels, for you to judge. Enough sky is needed, not "
                "proof: a halo too faint to be detected can remain. The pixels left out become "
                "the mask layer TOMASK, of role sky, which is never used in a fit.",
                (
                    ("MASK=", "mask layers to leave out as well"),
                    ("SEGMENT", "find and leave out every source first"),
                    (
                        "THRESH=, MINAREA=",
                        "the detection: sigma above the level (0.5), pixels (12)",
                    ),
                    ("GROW=", "pixels added around every source (2; 0: none)"),
                    ("CELLS=", "the boxes per side for the cells and the plane (8)"),
                    (
                        "BACK_SIZE=",
                        "0: one level, iterated (default); > 0: a background mesh of that "
                        "size, as SExtractor's BACK_SIZE (it takes part of a large halo for sky)",
                    ),
                    ("FWHM=", "a Gaussian detection filter of that FWHM, pixels (0: 3 × 3)"),
                    ("PAD=", "a value that marks pixels without data (0 around a mosaic)"),
                    ("TOMASK=", "the name of the sky layer (skymask)"),
                    ("TOMAP=, TOPLANE=", "buffers for the cell medians and the fitted plane"),
                ),
                "sky 1 MASK=mymask\nsky 1 SEGMENT\nsky 1 SEGMENT THRESH=1 GROW=3 TOMAP=19",
            ),
            _c(
                "skyplane",
                "skyplane N [THRESH=0.5] [MINAREA=12] [GROW=2] [BOX=32] [PASSES=10] [FWHM=0] "
                "[PAD=v] [MASK=a,b] [TOMASK=skymask] [TOPLANE=M] [TOFLAT=M] [TOSUB=M]",
                "The sky of buffer N as a plane: a level at the centre of the frame and a "
                "gradient along x and y, each with its error. A gradient makes one side of the "
                "frame brighter than the sky's level, and a detection against one level takes "
                "that side for a source; so the sources are found on the image minus the "
                "plane's tilt, the plane is fitted to the medians of BOX × BOX boxes of what is "
                "left (weighted by their pixels, clipped at 3 MAD), and the two are repeated "
                "until the tilt changes by less than 0.02 sigma. The report says whether the "
                "gradient is significant (3 times its error) and how much the boxes scatter "
                "about the plane: much more than the noise allows means light that is not sky "
                "(a halo). TOFLAT= makes the image to fit: the tilt removed, the level kept, "
                "and its sky measured as sky N SEGMENT would, so that elliprof and pair use it "
                "with a constant sky.",
                (
                    ("THRESH=, MINAREA=, GROW=, FWHM=, PAD=, MASK=", "the detection, as for sky"),
                    ("BOX=", "the size of the boxes, pixels (32)"),
                    ("PASSES=", "at most this many detections and fits (10)"),
                    ("TOMASK=", "the name of the sky layer (skymask)"),
                    ("TOPLANE=", "a buffer for the plane"),
                    ("TOFLAT=", "a buffer for the image minus the tilt (the level kept)"),
                    ("TOSUB=", "a buffer for the image minus the whole plane"),
                ),
                "skyplane 1\nskyplane 1 TOFLAT=20 TOPLANE=21",
            ),
            _c(
                "mn",
                "mn N",
                "The mean, median, standard deviation (sigma) and number of the finite pixels "
                "of buffer N. Sets the variables mean, median and sigma.",
                (),
                "mn 1",
            ),
            _c(
                "max min",
                "max N   (min N)",
                "The largest (smallest) pixel and its column and row. Sets the variables max, "
                "maxc, maxr (min, minc, minr).",
                (),
                "max 1\nmin 1",
            ),
            _c(
                "abx",
                "abx N",
                "The sum, mean and number of the finite pixels, and the maximum and minimum "
                "with their columns and rows.",
                (),
                "abx 1",
            ),
            _c(
                "print",
                "print N C R [SIZE=5]",
                "The pixel values in a SIZE × SIZE square around column C, row R, as a table "
                "with the rows from top to bottom.",
                (("SIZE=", "the side of the square, in pixels (default 5)"),),
                "print 1 567 562 SIZE=3",
            ),
        ],
    ),
    (
        "Changing the geometry",
        "Results go into the buffer itself unless OUT=M asks for another one.",
        [
            _c(
                "shift",
                "shift N DX= DY= [ORDER=3] [OUT=M]",
                "Move the image by DX columns and DY rows, fractions of a pixel too. Pixels "
                "that come in from outside are NaN.",
                (
                    ("DX=, DY=", "the shift in pixels (default 0)"),
                    ("ORDER=", "the order of the spline interpolation, 0 to 5 (default 3)"),
                    ("OUT=", "the buffer for the result"),
                ),
                "shift 9 DX=1.5 DY=-0.25",
            ),
            _c(
                "rotate",
                "rotate N ANGLE= [ORDER=3] [OUT=M]",
                "Rotate the image about its centre by ANGLE degrees, counter-clockwise. "
                "Multiples of 90° are exact (no interpolation); other angles keep the image's "
                "size, with NaN in the corners.",
                (
                    ("ANGLE=", "degrees, counter-clockwise (rotate N 90 works too)"),
                    ("ORDER=", "the spline order (default 3)"),
                    ("OUT=", "the buffer for the result"),
                ),
                "rotate 9 ANGLE=90\nrotate 9 ANGLE=12.5 OUT=10",
            ),
            _c(
                "flip",
                "flip N X|Y [OUT=M]",
                "Mirror the image left to right (X) or top to bottom (Y).",
                (),
                "flip 9 X",
            ),
            _c(
                "bin",
                "bin N k [MEAN] [OUT=M]",
                "Add up the pixels in blocks of k × k, making an image k times smaller; with "
                "MEAN, average them instead. NaN pixels are left out.",
                (("MEAN", "average the blocks instead of adding them"),),
                "bin 9 2 MEAN OUT=10",
            ),
            _c(
                "mash",
                "mash N ROWS|COLS [OUT=M]",
                "Add up all the rows into one row (ROWS), or all the columns into one column "
                "(COLS). The result goes into a free buffer unless OUT= names one.",
                (),
                "mash 1 ROWS OUT=10",
            ),
        ],
    ),
    (
        "Smoothing and cleaning",
        "Results go into the buffer itself unless OUT=M asks for another one.",
        [
            _c(
                "smooth",
                "smooth N FWHM=   or   smooth N BOXCAR=k   [OUT=M]",
                "Smooth with a Gaussian of the full width at half maximum given, or with a "
                "k × k boxcar. NaN pixels neither spread nor count.",
                (
                    ("FWHM=", "pixels (default 2)"),
                    ("BOXCAR=", "the side of the box, in pixels"),
                ),
                "smooth 9 FWHM=3 OUT=10\nsmooth 9 BOXCAR=5 OUT=10",
            ),
            _c(
                "filt",
                "filt N MEDIAN=k [OUT=M]",
                "Replace each pixel by the median of the k × k pixels around it.",
                (("MEDIAN=", "the side of the box (default 5)"),),
                "bin 1 4 MEAN OUT=10\nfilt 10 MEDIAN=3",
            ),
            _c(
                "zap",
                "zap N [SIG=5] [SIZE=5] [OUT=M]",
                "Replace the pixels more than SIG standard deviations above the median of "
                "the SIZE × SIZE around them by that median: cosmic rays and hot pixels. The "
                "standard deviation is measured locally, from the median absolute deviation.",
                (
                    ("SIG=", "how many standard deviations above the local median (default 5)"),
                    ("SIZE=", "the side of the box (default 5)"),
                ),
                "zap 1 SIG=5 OUT=10",
            ),
            _c(
                "clip",
                "clip N [MIN=] [MAX=] [VALUE=] [OUT=M]",
                "Pixels below MIN become MIN and pixels above MAX become MAX, or VALUE when it "
                "is given (VALUE=nan makes them NaN).",
                (
                    ("MIN=, MAX=", "the limits"),
                    ("VALUE=", "what clipped pixels become instead of the limit"),
                ),
                "clip 10 MIN=0 MAX=60000\nclip 10 MAX=60000 VALUE=nan",
            ),
            _c(
                "skysub",
                "skysub N [VALUE=]",
                "Subtract the sky from buffer N: the value given, or the median of the image "
                "clipped at 3 standard deviations.",
                (("VALUE=", "the sky to subtract"),),
                "skysub 10\nskysub 10 VALUE=3246",
            ),
        ],
    ),
    (
        "Making images",
        "For practice and tests: add a star or noise to an image, or look at its power "
        "spectrum.",
        [
            _c(
                "gauss",
                "gauss N X= Y= FWHM= PEAK=",
                "Add a Gaussian star at X, Y (ELLIPROF's coordinates: the first pixel spans "
                "0 to 1) with the full width at half maximum and the peak given.",
                (),
                "gauss 10 X=200 Y=200 FWHM=2.5 PEAK=5000",
            ),
            _c(
                "noise",
                "noise N SIGMA= [SEED=]",
                "Add Gaussian noise of standard deviation SIGMA. SEED= makes it the same every "
                "time.",
                (),
                "noise 10 SIGMA=10 SEED=1",
            ),
            _c(
                "fft",
                "fft N OUT=M",
                "The power spectrum of buffer N, with the zero frequency at the centre, into "
                "buffer M. The mean is subtracted first and NaN pixels count as 0.",
                (),
                "fft 1 OUT=11",
            ),
        ],
    ),
    (
        "The FITS header",
        "",
        [
            _c(
                "ch",
                "ch N KEY=value ...",
                "Set FITS header keywords of buffer N. Numbers are kept as numbers; quote "
                "text with spaces.",
                (),
                "ch 1 OBJECT='UGC 12517' GAIN=2.4",
            ),
            _c(
                "chead",
                "chead N [WORD]",
                "Print the header of buffer N, or only the cards containing WORD.",
                (),
                "chead 1 NAXIS",
            ),
        ],
    ),
    (
        "Variables",
        "Numbers kept by name. Variable names are not case-sensitive (set keeps them in "
        "lower case). A variable, or an expression of variables, can be used as any "
        "keyword's value (X0=xc, R1=2*re), as a number in a command (ac 1 sky, print 1 "
        "maxc maxr) and in formulas ([2] = [1] - sky).",
        [
            _c(
                "set",
                "set NAME=expression ...",
                "Set variables. An expression may use numbers, other variables, + - * / ** "
                "and sqrt log10 log exp abs sin cos tan, and pi.",
                (),
                "set xc=567 yc=562 re=126.5\nset r1=2.75*re",
            ),
            _c(
                "typ",
                "typ [NAME ...]",
                "Print variables: those named, or all of them.",
                (),
                "typ\ntyp xc yc",
            ),
        ],
    ),
    (
        "Isophotes and sources",
        "The two measurements of AstroVista Lite. The tools on the right of the window type "
        "these commands for you.",
        [
            _c(
                "elliprof",
                "elliprof N X0= Y0= R0= R1= NR= [NITER=10] [SKY=] [RMSTAR] [MASK=a,b] "
                "[MASKBUF=M,...] [MASKFILE=file] [TOMODEL=M] [TORESID=M]",
                "Fit elliptical isophotes to buffer N with ELLIPROF, John Tonry's isophote "
                "fitting from MONSTA (through the elliprof package): NR ellipses from R0 to "
                "R1 around X0, Y0. The profile goes to Results ▸ Profile and the ellipses onto "
                "the image.",
                (
                    ("X0=, Y0=", "the centre, in ELLIPROF's coordinates (required)"),
                    ("R0=, R1=", "the first and last semi-major axis, in pixels (required)"),
                    ("NR=", "the number of ellipses (required)"),
                    ("NITER=", "iterations of the fit (default 10)"),
                    ("SKY=", "a constant sky subtracted before the fit"),
                    ("RMSTAR", "reject star-like bright samples along each ellipse"),
                    ("MASK=", "mask layers whose pixels are left out"),
                    ("MASKBUF=", "buffers used as masks: masked where 0 or NaN"),
                    ("MASKFILE=", "a mask file (FITS or .dmask, 0 = masked)"),
                    ("TOMODEL=, TORESID=", "the buffers for the model and the residual"),
                    ("RAW", "the residual on every pixel (masked ones too), not blanked"),
                    (
                        "R1=AUTO, NR=AUTO",
                        "an iterative fit, as the pair fit's for one galaxy: R1 where the light "
                        "fades into the noise, isophotes about 30 % apart; the galaxy is fitted "
                        "out to R1 at once (one galaxy has no growth), and to a smaller R1 when "
                        "the isophotes fail the checks",
                    ),
                    (
                        "SKYITER=, SKYPLANE=1, SKYFACTOR=",
                        "the sky factor refined from the residual sky (needs sky N SEGMENT); "
                        "a sky tilt taken out when it lowers the residual",
                    ),
                    (
                        "TRYFIX=, HARMONICS=AUTO, TRYCOS3X=, TRYCOS4X=",
                        "the search: fixed and free centres fitted, the lowest residual kept; "
                        "then the harmonics",
                    ),
                    (
                        "SEARCH=FULL",
                        "the default: TIE off and a larger R0 (1.5 × R0, or out of a masked "
                        "core) tried in the search too, the fixed centre moved to 1/8 px and "
                        "the number of isophotes changed (NR=AUTO), the lowest residual kept; "
                        "SEARCH=BASIC: only TRYFIX",
                    ),
                    (
                        "PARALLEL=, RUNS=, JITTER=, SEED=, TAIL=, SHAPETOL=",
                        "N fits averaged (mean model, mean residual = image − mean sky − "
                        "mean model, each fit's own); the model beyond R1; the check of the "
                        "outermost isophotes",
                    ),
                    (
                        "RESIDMASK=, RESIDTHRESH=, RESIDGROW=, RESIDCORE=",
                        "n rounds: the faint sources the residual shows (above 8 × its local "
                        "RMS, not within 10 px of the centre) masked, grown by 2 px, and the "
                        "galaxy fitted again; the layer residsources and a source run per "
                        "round are made",
                    ),
                    ("MINUS=", "buffers subtracted from the image before the fit"),
                    (
                        "MODE=",
                        "harmonics by name: none, 3, 4, 3,4, 4,6, 3m, 4m, 6m,4m (m: their "
                        "median over all ellipses; 6 replaces 3); or COS3X=, COS4X=",
                    ),
                    ("FIXCTR=", "0 every ellipse finds its centre, 1 fixed at X0, Y0, 2 median"),
                    ("RLAW=, GAIN=, TIE=", "ELLIPROF's spacing of radii, gain and smoothing"),
                ),
                "elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 SKY=3246 RMSTAR "
                "MASK=u12517j TOMODEL=3 TORESID=4\n"
                "elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 SKY=3246 MODE=4 FIXCTR=0 "
                "MASK=u12517j TORESID=4 RAW",
            ),
            _c(
                "models",
                "models N [STEP=step] [SHOW=k,...] [AVERAGE=k,...]",
                "The models of buffer N's latest iterative ELLIPROF fit (Automatic or "
                "Iterative): every model made, numbered, with its parameters (centre, FIXCTR, "
                "COS3X, COS4X, TIE, R0, R1 asked and reached, NR, sky, sky tilt), its residual "
                "and its core's. The residual: between 1.15 × R0 and 1.1 × R1 (as asked, the "
                "same for every model), in rings 20 % apart, each ring's RMS over its own "
                "noise (measured once on the image), every ring counted the same, so a "
                "residual in the bright middle is not hidden by the faint outskirts; the core: "
                "the rings out to twice 1.15 × R0. The lowest residual of the search, the "
                "centre, the harmonics, the isophotes and the sky tilt is the fit kept; a "
                "model whose core is over twice the cleanest (and over 3) is high-contrast, "
                "never kept nor averaged; the averaged fits and the refits after it are "
                "listed only. ELLIPROF's Models table shows the same, a tab per step, with "
                "To buffers, Average selected and each model's parameters on a right click; "
                "the buffers made carry them too (right-click the buffer).",
                (
                    (
                        "STEP=",
                        "one step only: search, centre, harmonics, isophotes, tilt, residual "
                        "or average",
                    ),
                    ("SHOW=", "these models and their residuals (image − sky − model) in buffers"),
                    (
                        "AVERAGE=",
                        "the mean of these models in buffers: the mean model and the residual "
                        "image − mean sky − mean model, with its residual on the same rings "
                        "and each model's alone",
                    ),
                ),
                "elliprof 1 X0=567 Y0=562 R0=9 R1=AUTO NR=AUTO SKY=3246 HARMONICS=AUTO "
                "MASK=u12517j\nmodels 1 STEP=harmonics\nmodels 1 SHOW=2\n"
                "models 1 AVERAGE=1,2,3",
            ),
            _c(
                "sources",
                "sources N [KEYWORD=value ...] [MASK=a,b] [TOSEG=M [SEGMASK=0]]",
                "Find the sources (stars and galaxies) in buffer N and measure them, with SEP "
                "(the algorithms of SExtractor). Each run is numbered; the catalogue is in "
                "Results ▸ Sources. Keywords not given take the values of the Source finder "
                "panel's defaults.",
                (
                    (
                        "THRESH_TYPE=",
                        "RELATIVE (in standard deviations of the background, "
                        "default) or ABSOLUTE",
                    ),
                    ("DETECT_THRESH=", "the detection threshold (default 3)"),
                    ("DETECT_MINAREA=", "connected pixels above it (default 5)"),
                    (
                        "FILTER=, FILTER_FWHM=, FILTER_TYPE=",
                        "smoothing before detection: Y/N, "
                        "FWHM in pixels (0: a 3 × 3 kernel), matched or conv",
                    ),
                    ("DEBLEND_NTHRESH=, DEBLEND_MINCONT=", "deblending (32, 0.005)"),
                    ("CLEAN=, CLEAN_PARAM=", "cleaning of spurious detections (Y, 1)"),
                    (
                        "BACK_TYPE=, BACK_VALUE=, BACK_SIZE=, BACK_FILTERSIZE=, BACK_FILTTHRESH=",
                        "the background: AUTO (a map in cells of BACK_SIZE = 64 pixels) or MANUAL "
                        "(the constant BACK_VALUE)",
                    ),
                    (
                        "PHOT_APERTURES=, PHOT_AUTOPARAMS=, PHOT_FLUXFRAC=",
                        "aperture diameters "
                        "(5), MAG_AUTO's Kron factor and smallest radius (2.5,3.5), FLUX_RADIUS's "
                        "fraction (0.5)",
                    ),
                    ("MAG_ZEROPOINT=, GAIN=", "the magnitude of one count (0), e-/ADU (0)"),
                    ("MASK=", "mask layers whose pixels are left out"),
                    (
                        "TOSEG=",
                        "the buffer for the segmentation image; the mask layer segK (K the "
                        "run) masks every source's pixels, its inverse",
                    ),
                    ("SEGMASK=0", "no segK mask layer with TOSEG"),
                ),
                "sources 1\nsources 1 DETECT_THRESH=2.5 DETECT_MINAREA=8 TOSEG=12",
            ),
            _c(
                "runs",
                "runs",
                "List the source-finder runs: number, image, how many sources, and the "
                "settings that differ from the defaults.",
                (),
                "runs",
            ),
            _c(
                "centre",
                "centre N SOURCE=number [RUN=k]",
                "Take ELLIPROF's centre from a found source (by its NUMBER), of the last run "
                "on buffer N or of run k.",
                (),
                "centre 1 SOURCE=1",
            ),
        ],
    ),
    (
        "Pairs of galaxies",
        "Two overlapping galaxies are fitted in turn, each with the other's model taken out. "
        "Help ▸ Documentation ▸ Fitting a pair of galaxies explains the method.",
        [
            _c(
                "pair",
                "pair N SKY=s [shared keywords] A: X0= Y0= R0= R1= [A's keywords] "
                "B: X0= Y0= R0= R1= [B's keywords]",
                "Fit a pair: A, then B on the image minus A (each with the other's core left "
                "out), then CYCLES times A on the image minus B and B on the image minus A. A "
                "model is kept only when its fit passes every check; the result is the last "
                "cycle in which both passed. It makes four buffers: the models of A and B, "
                "A + B, and the residual (the image minus the sky and both models, on every "
                "pixel). The status says how it ended: stable_valid_pair, "
                "valid_pair_not_stable, retained_after_failed_update, "
                "incomplete_initialization or cancelled. help pair lists every keyword.",
                (
                    ("SKY=", "the sky, the same for every call (required)"),
                    ("SKYFACTOR=", "the sky used is SKYFACTOR × SKY (1; 0.5 to 1.5)"),
                    (
                        "SKYITER=",
                        "refit up to this many times, each with the factor + m / SKY, m the "
                        "clipped median of image − sky − A − B on the sky pixels of the image's "
                        "sky measurement; the fit with the smallest |m| is kept (0: off). "
                        "Each step moves the sky by at most 2 sigma; when the residual follows "
                        "less than half a step (the galaxies' light fills the frame and takes "
                        "it up), it stops and keeps the sky asked for",
                    ),
                    (
                        "MASK=, MASKFILE=, MASKBUF=",
                        "what to leave out of every fit (sky layers are refused)",
                    ),
                    (
                        "CYCLES=, NITER=",
                        "A then B this many times (4); iterations in each call (10)",
                    ),
                    ("CORECAP=", "the largest core left out around the other galaxy (50 px)"),
                    ("TOMODELA=, TOMODELB=, TOMODEL=, TORESID=", "the result buffers (free ones)"),
                    ("VIEWS", "also the image − sky − B and image − sky − A"),
                    ("STOP, BOOTSTRAP=0", "stop once stable; no partial first fits"),
                    ("CORES=0", "mask nothing around the companion (its model is subtracted)"),
                    (
                        "CONVERGE=, CONSECUTIVE=",
                        "stop after CONSECUTIVE cycles at the final radii with every change "
                        "below CONVERGE (0.02; 0: run every cycle)",
                    ),
                    (
                        "ONFAIL=",
                        "reduce (default: smaller steps), hold, retry or stop when a fit fails",
                    ),
                    ("STEPS=", "geometric growth steps for a galaxy with START alone"),
                    (
                        "DAMP=, POLISH=",
                        "damping of the companion model subtracted (1: none); undamped final "
                        "cycles (2) so that the result is ELLIPROF's own models",
                    ),
                    (
                        "TRYDAMP=, TRYFIX=, TRYFACTOR=",
                        "fit every combination and keep the stable one with the smallest "
                        "residual, with fixed centres unless free centres lower it by more "
                        "than 0.01 of the noise (Results > Pair > Search lists them)",
                    ),
                    (
                        "TIE=, RMSTAR",
                        "passed to every call: smoothing with radius (−3: over 3 isophotes) "
                        "and the rejection of bright samples along each isophote",
                    ),
                    (
                        "TAIL=",
                        "the model beyond each galaxy's last isophote, which is subtracted "
                        "from the other's image: elliprof (its r^¼ extrapolation, default), "
                        "measured (the light measured in rings of the last isophote's shape), "
                        "sersic (a Sérsic law fitted to the outer profile), exp or cut",
                    ),
                    (
                        "SKYPLANE=1",
                        "a plane fitted to the residual of the fit kept; when significant, the "
                        "search again on the image minus its tilt, kept if the residual is "
                        "lower",
                    ),
                    (
                        "PARALLEL=",
                        "N fits, all averaged (1 to 16): the fit kept and N − 1 repeats with "
                        "R0 and R1 changed at random by 10–15 % (JITTER=0.15, SEED=), run N at "
                        "a time, each refining its own sky; buffers get the mean A, B and "
                        "A + B models, the mean residual (image − mean sky − mean A + B), the "
                        "models' spread and every fit's own model and residual",
                    ),
                    (
                        "RUNS=",
                        "the number of fits averaged when it is not PARALLEL's (RUNS=10 "
                        "PARALLEL=5: ten fits, five at a time)",
                    ),
                    (
                        "HARMONICS=AUTO",
                        "try every COS3X and COS4X for each galaxy on the fit kept, and keep "
                        "the stable fit with the smallest residual",
                    ),
                    (
                        "WINGS=1",
                        "mask the wings and spikes of bright sources that a first "
                        "pass's residual shows beyond their masks",
                    ),
                    (
                        "METHOD=JOE",
                        "Joe's mode: each galaxy's centre left out while the other is fitted, "
                        "the smaller galaxy first and not far out, then the two in turn, each "
                        "R1 grown only as far as its ellipses (fitted with free centres) are "
                        "not pulled towards the other; no search, about a tenth of the calls",
                    ),
                    (
                        "FIRST=, JOEGROW=, JOECORE=, JOECOREMIN=, JOECOREMAX=",
                        "Joe's mode: the galaxy fitted first (SMALLER, A or B), the R1 each "
                        "cycle tries (× the last one kept, 1.2), each centre left out while "
                        "the other is fitted (0.2 of the separation, 6 to 30 px; 0: none)",
                    ),
                    (
                        "PULLPX=, PULLFRAC=, PULLRUN=, PULLR0=",
                        "what counts as a pull, in Joe's mode and the pull check: a move "
                        "towards the other of more than PULLPX px (0.5) or PULLFRAC of the "
                        "radius (0.02), PULLRUN isophotes in a row (2), in a free-centre fit "
                        "starting at R0 PULLR0 at least (8 px)",
                    ),
                    (
                        "SHAPETOL=",
                        "Joe's mode, at the end: when one of a galaxy's 3 outermost ellipses "
                        "is off the trend inside it by more than this (0.12; ellipticity and "
                        "angle together) or its model beyond R1 is anomalous, coarser "
                        "isophotes and a smaller R1 are tried and the largest clean fit kept "
                        "(0: off)",
                    ),
                    (
                        "RESIDMASK=, RESIDTHRESH=, RESIDGROW=, RESIDCORE=",
                        "n rounds: the faint sources in the residual of the fit kept (above "
                        "8 × its local RMS, not within 10 px of a centre) masked, grown by "
                        "2 px, and the fit kept fitted again; the layer residsources and a "
                        "source run per round are made",
                    ),
                    (
                        "PULLCHECK=0",
                        "no pull check; by default each galaxy of a complete fit is fitted "
                        "once more with free centres, and the report says whether its "
                        "ellipses move towards the other and from which radius",
                    ),
                    (
                        "A: … B: …",
                        "each galaxy: X0= Y0= R0= R1= NR=25 NAME= FIXCTR= CORE= ADAPTIVE "
                        "COS3X= COS4X= (ELLIPROF's harmonics in the model: 2 each isophote's, "
                        "the default; 1 the median; 0 none; COS3X −2, −1, −3: the 6th order) "
                        "or MODE=; TRYCOS3X= TRYCOS4X= (values tried on the fit kept, A's then "
                        "B's, kept when the residual falls by more than 0.01 of the noise); "
                        "START= (first R1) with STEP= (+ pixels) or FACTOR= (×), or RADII= "
                        "(a list); BLOCK= (left out around it during the other's first fit); "
                        "R1=AUTO, START=AUTO and NR=AUTO (about 30 % apart): the radii and isophotes chosen on "
                        "simulated pairs (R1 where the light away from the other galaxy falls "
                        "below the noise, measured again on the image minus the other's "
                        "model)",
                    ),
                ),
                "pair 1 SKY=3246 MASK=u12517j CYCLES=2 A: NAME=UGC12517 X0=567 Y0=562 R0=9 "
                "R1=200 NR=20 B: X0=362.1 Y0=498.4 R0=2 R1=25 NR=10",
            ),
            _c(
                "pairstep",
                "pairstep N K",
                "Put the model made by call K of the last pair fit of buffer N's image, and "
                "the image that call was fitted on (D for A's first fit, D − M_B for A, D − "
                "M_A for B), into two buffers. The iteration table of the Pair fit tool does "
                "this when a row is double-clicked.",
                (),
                "pairstep 1 3",
            ),
            _c(
                "pairreport",
                "pairreport N [file]",
                "A plain-text report of the last pair fit of buffer N's image: the settings, "
                "the cores and blocks, every elliprof call with why it failed, the growth "
                "steps and cycles, the sky, the residual in rings around each galaxy (on the "
                "side facing the other galaxy and away from it) and hints about what may "
                "have gone wrong. Copy report in the Pair fit tool puts it on the clipboard.",
                (),
                "pairreport 1\npairreport 1 n6768_report.txt",
            ),
            _c(
                "pairexport",
                "pairexport N folder",
                "Write the last pair fit of buffer N's image to a folder: the inputs it used, "
                "the requested and effective settings, every elliprof call, the profiles, the "
                "ellipses as DS9 regions, the models, the residual (float64) and a manifest. "
                "transaction.json says complete once everything is written.",
                (),
                "pairexport 1 pair_result",
            ),
            _c(
                "window",
                "window N BOX=x0,y0,x1,y1 [OUT=M] [MASKS=a,b|all]",
                "Cut out pixels x0 ≤ x < x1, y0 ≤ y < y1 (whole-pixel edges; data[y0:y1, "
                "x0:x1]) into buffer OUT, the next free one by default. The WCS follows, and "
                "the offset is kept in LTV1, LTV2. A box partly outside is clipped.",
                (("MASKS=", "mask layers to cut too (new layers NAME_w); all: every one"),),
                "window 1 BOX=300,300,800,800 OUT=20",
            ),
        ],
    ),
    (
        "Masks",
        "A mask layer marks the pixels to leave out of a measurement. In files, 0 means "
        "masked and 1 good. Coordinates are ELLIPROF's: the first pixel spans 0 to 1. The "
        "drawing actions take ERASE to unmask instead. help mask lists them all.",
        [
            _c(
                "mask new",
                "mask new NAME [N]",
                "An empty mask layer the size of buffer N (without N: of the lowest-numbered buffer).",
                (),
                "mask new edges 1",
            ),
            _c(
                "mask load",
                "mask load NAME file [N]",
                "Read a mask from a FITS file or a legacy .dmask bitmap (0 = masked). The mask "
                "also appears as a buffer (1 good, 0 masked) kept up to date with the layer.",
                (),
                "mask load published u12517j.dmask 1",
            ),
            _c(
                "mask save",
                "mask save NAME file",
                "Write a mask layer: a .dmask name writes a legacy bitmap, any other name a "
                "FITS image (1 good, 0 masked).",
                (),
                "mask save edges edges_mask.fits",
            ),
            _c(
                "mask frombuf tobuf",
                "mask frombuf NAME N   ·   mask tobuf NAME M|new",
                "A mask layer from buffer N (masked where it is 0 or NaN), or a mask layer as "
                "buffer M (or a new buffer).",
                (),
                "mask frombuf frombuffer 1\nmask tobuf edges new",
            ),
            _c(
                "mask copy rename delete",
                "mask copy NAME FROM  ·  mask rename NAME NEW  ·  mask delete NAME",
                "Copy, rename or delete a mask layer.",
                (),
                "mask copy spare edges\nmask rename spare spare2\nmask delete spare2",
            ),
            _c(
                "mask union intersect",
                "mask union|intersect NAME A B ...",
                "A new layer with the pixels masked by any (union) or all (intersect) of the "
                "layers A, B, …",
                (),
                "mask union both edges published",
            ),
            _c(
                "mask circle ellipse box polygon brush",
                "mask circle NAME X Y R  ·  mask ellipse NAME X Y A B ANGLE  ·  "
                "mask box NAME X0 Y0 X1 Y1  ·  mask polygon NAME X1 Y1 X2 Y2 X3 Y3 ...  ·  "
                "mask brush NAME R X1 Y1 X2 Y2 ...   [ERASE]",
                "Draw a shape on a layer: a circle of radius R, an ellipse with semi-axes A, B "
                "turned by ANGLE degrees, a box between two corners, a polygon through its "
                "corners, or circles of radius R along a path. The mask tools of the window "
                "type these commands.",
                (("ERASE", "unmask the shape instead of masking it"),),
                "mask circle edges 812 440 12\nmask ellipse edges 300 300 70 22 35\n"
                "mask box edges 10 10 60 40\nmask polygon edges 100 100 150 100 125 140\n"
                "mask brush edges 3 400 400 420 410 440 400\nmask circle edges 812 440 12 ERASE",
            ),
            _c(
                "mask unpatch",
                "mask unpatch NAME X Y",
                "Unmask the whole connected masked area under X, Y.",
                (),
                "mask circle edges 812 440 12\nmask unpatch edges 812 440",
            ),
            _c(
                "mask nan range",
                "mask nan NAME N  ·  mask range NAME N LO HI",
                "Mask the pixels of buffer N that are not finite (NaN or infinite), or whose "
                "values lie from LO to HI.",
                (),
                "mask nan edges 1\nmask range edges 1 60000 1e9",
            ),
            _c(
                "mask sources segments",
                "mask sources NAME [RUN=k] [SCALE=1.5] [KEEP=centre|none|number] [X0= Y0=] "
                "[IDS=1,5,9] [WHERE='condition']  ·  mask segments NAME [RUN=k] [GROW=0] "
                "[KEEP=...] [X0= Y0=] [ADD]",
                "Mask the found sources: their Kron ellipses enlarged by SCALE (sources), or "
                "their own pixels from the segmentation grown by GROW pixels (segments: the "
                "layer becomes exactly the segments, everything else unmasked; ADD adds them "
                "instead). The "
                "galaxy at ELLIPROF's centre is kept unmasked unless KEEP= says otherwise.",
                (
                    ("RUN=", "the source-finder run (default: the last on that image)"),
                    ("SCALE=", "the Kron ellipses × SCALE (default 1.5)"),
                    (
                        "KEEP=",
                        "centre (the source at X0, Y0, default; before any fit, the largest "
                        "source), none, or a source NUMBER",
                    ),
                    (
                        "X0=, Y0=",
                        "the centre to keep (default: the last ELLIPROF centre); a list keeps "
                        "several, with the segment under each (X0=120,310 Y0=200,240: a pair)",
                    ),
                    ("IDS=", "only these source numbers"),
                    (
                        "WHERE=",
                        "only the sources meeting a condition: catalogue columns "
                        "compared with numbers, joined by and / or, read from left to right: "
                        "WHERE='MAG_AUTO < 22 and FWHM_IMAGE < 3'",
                    ),
                    ("GROW=", "pixels added around each segment (segments)"),
                ),
                "mask new found 1\nmask sources found SCALE=1.5\n"
                "mask sources found WHERE='FLUX_RADIUS < 2.5'\nmask segments found GROW=2\n"
                "mask segments found X0=120,310 Y0=200,240 GROW=2",
            ),
            _c(
                "mask grow shrink invert clear fill show hide",
                "mask grow|shrink NAME R  ·  mask invert|clear|fill NAME  ·  mask show|hide NAME",
                "Grow or shrink the masked areas by R pixels; swap masked and good; unmask "
                "or mask everything; show or hide the layer on the image.",
                (),
                "mask grow edges 2\nmask shrink edges 1\nmask hide edges\nmask show edges",
            ),
            _c(
                "mask role",
                "mask role NAME fit|sky|candidate|none",
                "Say what a layer is for. fit: left out of fits; sky: left out of the sky only "
                "(it covers the galaxies too, so a fit refuses it); candidate: proposed (from "
                "the residual, say) and not used until promoted with role fit; none: used "
                "wherever it is named. The pair tool lists the fit layers (and those without "
                "a role).",
                (),
                "mask role edges fit",
            ),
            _c(
                "masks",
                "masks",
                "List the mask layers: size and number of masked pixels.",
                (),
                "masks",
            ),
            _c(
                "undo redo",
                "undo  ·  redo",
                "Undo the last mask edit, or redo the edit undone last (the last 30 are kept). "
                "Ctrl+Z and Ctrl+Shift+Z do the same.",
                (),
                "undo\nredo",
            ),
        ],
    ),
    (
        "Files, folders and the session",
        "",
        [
            _c(
                "cd pwd",
                "cd [folder]  ·  pwd",
                "Change the folder that file names are relative to (cd alone: your home "
                "folder), or print it. Programs run from the terminal start there too.",
                (),
                "pwd",
            ),
            _c(
                "run",
                "run file",
                "Run the commands in a file, one per line (# starts a comment). File names in "
                "it are relative to the file's own folder. The run stops at the first error, "
                "naming the file and the line.",
                (),
                "run session.txt",
            ),
            _c(
                "help",
                "help [command]",
                "List the commands, or explain one. help mask lists the mask actions and help "
                "calc the arithmetic.",
                (),
                "help shift\nhelp mask",
            ),
            _c(
                "export",
                "export NAME=value ...",
                "Set environment variables for the programs run from the terminal.",
                (),
                "export CATALOGUES=catalogues",
            ),
        ],
    ),
]


def commands() -> List[Command]:
    return [c for _title, _intro, group in GROUPS for c in group]


def documented_names() -> set:
    """Every command name the reference explains (mask actions as 'mask ACTION')."""
    names = set()
    for c in commands():
        words = c.names.split()
        if words[0] == "mask" and len(words) > 1:
            names |= {"mask"} | {f"mask {w}" for w in words[1:]}
        else:
            names |= set(words)
    return names


def _e(text: str) -> str:
    return html.escape(text, quote=False)


def reference_html(website: bool = False) -> str:
    """Every command, group by group, as HTML for the documentation window (simple tags
    that Qt shows) or for the website (``website``: anchors and a contents list)."""
    out = []
    if website:
        out.append("<div class='toc'><b>Groups</b><ol>")
        for k, (title, _i, _g) in enumerate(GROUPS):
            out.append(f"<li><a href='#g{k}'>{_e(title)}</a></li>")
        out.append("</ol></div>")
    for k, (title, intro, group) in enumerate(GROUPS):
        anchor = f" id='g{k}'" if website else ""
        out.append(f"<h2{anchor}>{_e(title)}</h2>" if website else f"<h3>{_e(title)}</h3>")
        if intro:
            out.append(f"<p>{_e(intro)}</p>")
        for c in group:
            tag = " <span class='monsta'>MONSTA name</span>" if website and c.monsta else ""
            mark = (
                ""
                if website or not c.monsta
                else " <span style='color:#8b9099'>· MONSTA name</span>"
            )
            head = "h3" if website else "h4"
            out.append(f"<{head}><code>{_e(c.names)}</code>{tag}{mark}</{head}>")
            out.append(f"<p><code>{_e(c.syntax)}</code></p>")
            out.append(f"<p>{_e(c.what)}</p>")
            if c.keywords:
                rows = "".join(
                    f"<tr><td><code>{_e(k)}</code></td><td>{_e(v)}</td></tr>" for k, v in c.keywords
                )
                out.append(f"<table cellspacing='4'>{rows}</table>")
            if c.example:  # the website's code blocks; the documentation window's <pre>
                example = _e(c.example)
                out.append(
                    f"<pre><code>{example}</code></pre>" if website else f"<pre>{example}</pre>"
                )
    return "\n".join(out)


# ---- how the terminal works
TERMINAL_HTML = """
<p>The terminal at the bottom of the window is where AstroVista Lite keeps its record and
where you can type. Every button that changes data types a command there and runs it, so
you can learn the commands by watching, repeat them, and save a whole session as a file
that runs again.</p>

<h3>A command line</h3>
<pre>verb [buffer] [words] [KEY=value ...]   # a comment</pre>
<ul>
<li><b>The verb</b> comes first: <code>rd</code>, <code>stats</code>,
<code>elliprof</code>… It is not case-sensitive: <code>STATS 1</code> works.</li>
<li><b>Words</b> follow, separated by spaces: buffer numbers, file names, numbers and
flags such as <code>RMSTAR</code>, <code>ERASE</code> or <code>MEAN</code> (flags are not
case-sensitive).</li>
<li><b>Keywords</b> are written <code>KEY=value</code> with no spaces around the
<code>=</code>, in any order, anywhere after the verb. Their names are not
case-sensitive (<code>x0=567</code> is <code>X0=567</code>).</li>
<li><b>Quotes</b> keep spaces inside one word: <code>rd 1 "my galaxy.fits"</code>,
<code>WHERE='MAG_AUTO &lt; 22 and FWHM_IMAGE &lt; 3'</code>,
<code>ch 1 OBJECT='UGC 12517'</code>.</li>
<li><b>#</b> starts a comment, to the end of the line (unless it is inside quotes).</li>
<li><b>One command per line.</b> A semicolon does not separate commands. A line that ends
with <code>\\</code> goes on on the next line (in the terminal, in command files and in the
examples here): one command can be written over several lines.</li>
<li><b>tag="name"</b> at the end of any line that makes a buffer names that buffer:
<code>[2] = [1] - median([1]) tag="no sky"</code>.</li>
</ul>

<h3>Buffers, coordinates and units</h3>
<ul>
<li><b>Buffers</b> are numbered: a command takes the number (<code>stats 1</code>), a
formula writes <code>[1]</code> or <code>B1</code>.</li>
<li><b>ELLIPROF's coordinates</b> are used for X0, Y0, the mask shapes and
<code>gauss</code>: x and y run continuously, the first pixel spanning 0 to 1, so its
centre is at 0.5.</li>
<li><b>Columns and rows</b> counted from 0 are used by <code>print</code>,
<code>max</code>, <code>min</code> and <code>abx</code>: column c is the pixel between
x = c and c + 1.</li>
<li><b>The status bar</b> shows both, and the FITS pixel counted from 1, and the sky
position when the image has a WCS. Source catalogues use SExtractor's convention
(X_IMAGE of the first pixel's centre is 1).</li>
<li>Angles are in degrees, counter-clockwise; sizes and radii in pixels.</li>
</ul>

<h3>Formulas</h3>
<p>A line of the form <code>[M] = expression</code> (or <code>BM = expression</code>) is
image arithmetic, worked out pixel by pixel into buffer M:</p>
<pre><code>[3] = [1] - [2]
[4] = ([1] - 3246) * 2.5
[5] = median([1], [2], [3]) tag=stack</code></pre>
<p>Help ▸ Documentation ▸ <i>Image arithmetic</i> explains every operator and function;
<code>help calc</code> lists them.</p>

<h3>Variables</h3>
<pre><code>set xc=567 yc=562
elliprof 1 X0=xc Y0=yc R0=9 R1=347 NR=23
typ</code></pre>
<p><code>set</code> keeps numbers by name (names are not case-sensitive). A variable, or
an expression of variables such as <code>R1=2*re</code>, can stand for any keyword's value,
for a number in a command (<code>ac 1 sky</code>) and in a formula. <code>mn</code>,
<code>max</code> and <code>min</code> set variables too (mean, median, sigma; max, maxc,
maxr; min, minc, minr).</p>

<h3>Programs on your computer</h3>
<p>A line that is not an AstroVista Lite command runs as in a shell, when its first word
is a program on the PATH: <code>ls -l</code>, <code>grep</code>, <code>head
file.txt</code>, <code>python script.py</code>. A line starting with <code>!</code> always
goes to the shell, even when its first word is also a command name:
<code>! sort catalogue.csv | head</code>. The programs run in the terminal's folder
(<code>cd</code>, <code>pwd</code>), take no keyboard input, and are stopped by Ctrl+C or
after 10 minutes; up to 2000 lines of their output are shown. <code>ll</code> stands for
<code>ls -lrt</code> and <code>la</code> for <code>ls -A</code>.</p>
<p>A listing from <code>ls</code> is in colour: folders blue, FITS files green, pictures
purple, text files yellow, programs red. A click on a folder goes into it and lists it; a
click on a FITS file opens it.</p>

<h3>The keyboard</h3>
<table cellspacing="4">
<tr><td>Enter</td><td>run the line</td></tr>
<tr><td>↑ ↓</td><td>the lines typed before (the history)</td></tr>
<tr><td>Tab</td><td>complete what is being typed: commands, programs (after two letters),
mask actions and layer names, keywords such as <code>X0=</code> or
<code>DETECT_THRESH=</code>, buffer numbers after <code>TOMODEL=</code> and the like,
mask names after <code>MASK=</code>, and file and folder names; when several fit, a list
opens</td></tr>
<tr><td>Ctrl+C</td><td>stop the program running; when nothing runs, drop the commands
waiting and clear the line (with text selected, ⌘C or Ctrl+C copies it)</td></tr>
<tr><td>Ctrl+`</td><td>go to the terminal</td></tr>
<tr><td>Ctrl+Alt+= / Ctrl+Alt+−</td><td>larger or smaller text (View ▸ Terminal font
size)</td></tr>
</table>
<p>A click anywhere in the terminal puts the cursor on the command line. The
<i>Terminal</i> button in the status bar folds the terminal away; the command line then
stays in the bar at the bottom.</p>

<h3>How commands run</h3>
<ul>
<li>Commands run one after another, in the order typed or clicked. Reading files, fits
and the source finder show a busy cursor while they work.</li>
<li>When a command fails, its error is shown in red and the commands waiting after it are
dropped, since they may depend on it.</li>
<li><code>run file</code> runs a file of commands and stops at the first error, naming
the file and the line. File names in the file are relative to the file's own folder, so a
file can be shared with its data.</li>
<li>File ▸ Save the commands writes every command of the session (the lines starting with
›) to a file that <code>run</code>, File ▸ Run a command file or
<code>astrovista --run file</code> runs again.</li>
</ul>

<h3>A session, written as commands</h3>
<pre><code>rd 1 u12517j.fits
mask load u12517j u12517j.dmask 1
sources 1
mask new sources 1
mask sources sources SCALE=1.5
set xc=567 yc=562
elliprof 1 X0=xc Y0=yc R0=9 R1=347 NR=23 SKY=3246 RMSTAR MASK=u12517j,sources TOMODEL=3 TORESID=4
[5] = div([1] - [3], [3], 0) tag="residual / model"
stats 5 MASK=u12517j,sources
wd 4 residual.fits</code></pre>

<h3>Where the commands come from</h3>
<p>The names and the style of many commands (<code>rd</code>, <code>wd</code>,
<code>tv</code>, <code>ac</code>, <code>abx</code>, <code>shift</code>,
<code>smooth</code>, <code>clip</code>, <code>mask</code>, <code>elliprof</code>…) are those
of <b>MONSTA</b>, John Tonry's image analysis and plotting system, which joined
<b>Vista</b> (originally by Richard Stover and Tod Lauer, with image display by John Tonry)
and <b>Mongo</b> (by John Tonry). Astronomers who know them will feel at home. The
commands of AstroVista Lite are new code with options of their own, listed in the
<i>Command reference</i>, so MONSTA scripts do not run unchanged. ELLIPROF itself, the
isophote fitting, is John Tonry's, used through the elliprof package.</p>
"""
