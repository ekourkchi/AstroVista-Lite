"""The command line.  Every action of the window is a command, and commands can be typed.

    verb [buffer] [words] [KEY=value ...]     # a comment
    [3] = [1] - [2]                           arithmetic on buffers

Images live in numbered buffers.  Coordinates (X0, Y0, mask shapes) are ELLIPROF's: the
first pixel spans 0 to 1, so its centre is at 0.5.  ``help`` lists the commands.
"""

from __future__ import annotations

import hashlib
import os
import dataclasses
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import detection
from . import masks as maskops
from . import pair as pairs
from . import sky as skies
from .fitting import MODES, FitResult, FitSettings, fit
from .images import Image, image_hdus, read_fits, write_fits
from .masks import Mask
from .cmdbase import CommandError, Reply, integer, number, split_line
from .imageops import IMAGE_HELP, ImageCommands
from .imageops import KEYWORDS as IMAGE_KEYWORDS

HELP = {
    "help": "help [command]  -  list the commands, or explain one",
    "rd": "rd N file.fits[ext]  -  read an image into buffer N; ext is an HDU number or "
    "name: [1], [sci], [sci,2] (name and version); without it, the first image in the file",
    "fitsinfo": "fitsinfo file.fits  -  list what a FITS file holds: images (with their "
    "sizes), tables and headers, by number and name",
    "wd": "wd N file.fits  -  write buffer N as a FITS file",
    "tv": "tv N  -  show buffer N",
    "buffers": "buffers  -  list the buffers",
    "close": "close N  -  remove buffer N",
    "copy": "copy N M  -  copy buffer N into buffer M",
    "=": "[M] = expression  -  arithmetic on buffers: [3] = [1] - [2],  "
    "[4] = median([1], [2], [3]),  [5] = [1] - median([1]),  [6] = div([1], [2], 0),  "
    "[7] = [1] * mymask (a mask layer: NaN where it masks)  (help calc for everything)",
    "calc": "calc M EXPR='...'  -  the same as [M] = ...; help calc lists the functions",
    "stats": "stats N [MASK=a,b]  -  mean, median, standard deviation, minimum and maximum "
    "of buffer N, leaving out masked and non-finite pixels",
    "sky": "sky N [MASK=a,b]  -  the sky level of buffer N: a 3-sigma clipped median of the "
    "pixels the masks leave.   sky N SEGMENT [THRESH=0.5] [MINAREA=12] [GROW=2] [CELLS=8] "
    "[BACK_SIZE=0] [FWHM=0] [PAD=v] [MASK=a,b] [TOMASK=skymask] [TOMAP=M] [TOPLANE=M]  -  every "
    "source found down to THRESH sigma is left out (the layer TOMASK, role sky), and the sky "
    "is measured on the rest: level, cells, gradient, uncertainties and a verdict on the "
    "support (advice: the level is kept for you to judge).  FWHM= filters the detection with "
    "a Gaussian of that FWHM (0: a 3 x 3 kernel); BACK_SIZE= detects on a background mesh",
    "skyplane": "skyplane N [THRESH=0.5] [MINAREA=12] [GROW=2] [BOX=32] [PASSES=10] [FWHM=0] "
    "[PAD=v] [MASK=a,b] [TOMASK=skymask] [TOPLANE=M] [TOFLAT=M] [TOSUB=M]  -  the sky of buffer "
    "N as a plane: the sources are found on the image minus the plane's tilt and the plane is "
    "fitted to the medians of BOX x BOX boxes of what is left, until it settles.  TOFLAT: the "
    "image with the tilt removed and the level kept, with its sky measured (for elliprof and "
    "pair); TOSUB: the image minus the whole plane; TOPLANE: the plane",
    "elliprof": "elliprof N X0= Y0= R0= R1= NR= [NITER=10] [SKY=] [RMSTAR] [MASKFILE=file] "
    "[MASK=a,b] [MASKBUF=M,...] [TOMODEL=M] [TORESID=M] [RAW] [MINUS=M,...] [MODE=3,4] "
    "[COS3X=] [COS4X=] [FIXCTR=0|1|2] [RLAW=] [GAIN=] [TIE=]  -  fit elliptical isophotes to "
    "buffer N; MASK= names mask layers, MASKBUF= buffers used as masks (0 or NaN masked); "
    "MINUS= subtracts buffers first; RAW keeps every pixel of the residual; RESIDMASK=n "
    "[RESIDTHRESH=8] [RESIDGROW=2] [RESIDCORE=10]: n times, the faint sources the residual shows "
    "are found, masked (layer residsources) and the galaxy fitted again. Any of R1=AUTO NR=AUTO "
    "SKYITER= SKYPLANE= TRYFIX= HARMONICS=AUTO TRYCOS3X= TRYCOS4X= "
    "PARALLEL= RUNS= JITTER= TAIL= SHAPETOL= makes it an iterative fit with the pair fit's "
    "options for one galaxy (SKY= or a sky measurement needed)",
    "models": "models N [STEP=search|harmonics|tilt|residual|average] [SHOW=k,...] "
    "[AVERAGE=k,...]  -  the models of buffer N's latest iterative ELLIPROF fit, each with "
    "its parameters and its residual (RMS over the noise between 1.15 x R0 and 1.1 x R1); "
    "SHOW= puts models k (and their residuals) in buffers; AVERAGE= the mean of the models "
    "listed: mean model, residual = image - mean sky - mean model, and its residual",
    "pair": "pair N SKY=s [SKYFACTOR=1] [SKYITER=0] [STEPS=0] [MASK=a,b] [CYCLES=4] "
    "[NITER=10] [CORECAP=50] [VIEWS] [TOMODEL=M] [TORESID=M] [TOMODELA=M] [TOMODELB=M]  A: X0= "
    "Y0= R0= R1= [NR=25] [NAME=] [MODE=] "
    "[FIXCTR=0] [CORE=] [ADAPTIVE] [START=] [BLOCK=]  B: X0= Y0= R0= R1= ...  -  fit two overlapping galaxies: "
    "ELLIPROF on A and on B in turn, each with the other's model taken out (help pair)",
    "pairstep": "pairstep N K  -  put the model made by call K of the last pair fit of "
    "buffer N's image, and the image that call was fitted on (D, D - M_A or D - M_B), into "
    "two buffers (the iteration table's rows)",
    "pairreport": "pairreport N [file]  -  a plain-text report of the last pair fit of buffer "
    "N's image (settings, every call, growth, cycles, sky, residual rings around each galaxy, "
    "hints), shown here and written to file if one is given",
    "pairexport": "pairexport N folder  -  write the last pair fit of buffer N's image to a "
    "folder: inputs, settings, every call, profiles, regions, models and the residual",
    "window": "window N BOX=x0,y0,x1,y1 [OUT=M] [MASKS=a,b|all]  -  cut out the pixels "
    "data[y0:y1, x0:x1] (whole-pixel edges, ELLIPROF's coordinates) into buffer OUT (the next "
    "free one); the WCS follows, and the mask layers named are cut too",
    "sources": "sources N [KEYWORD=value ...] [MASK=a,b] [TOSEG=M]  -  find the sources "
    "(stars and galaxies) in buffer N.  Keywords (SExtractor's names): THRESH_TYPE, "
    "DETECT_THRESH, DETECT_MINAREA, FILTER, FILTER_FWHM, FILTER_TYPE, DEBLEND_NTHRESH, "
    "DEBLEND_MINCONT, CLEAN, CLEAN_PARAM, BACK_TYPE, BACK_VALUE, BACK_SIZE, BACK_FILTERSIZE, "
    "BACK_FILTTHRESH, PHOT_APERTURES, PHOT_AUTOPARAMS, PHOT_FLUXFRAC, MAG_ZEROPOINT, GAIN; "
    "MASK= leaves the pixels of mask layers out",
    "runs": "runs  -  list the runs of the Source finder",
    "centre": "centre N SOURCE=number [RUN=k]  -  take ELLIPROF's centre from a found source",
    "mask": "mask ACTION NAME ...  -  make, draw, combine, change and remove mask layers "
    "(help mask lists the actions)",
    "undo": "undo  -  undo the last mask edit (redo: do it again)",
    "redo": "redo  -  redo the mask edit undone last",
    "masks": "masks  -  list the mask layers",
    "run": "run file  -  run the commands in a file, one per line (file names in it are "
    "relative to its folder)",
    "cd": "cd folder  -  change the folder that file names are relative to",
    "pwd": "pwd  -  the folder that file names are relative to",
    "tag": 'tag N "name"  -  give buffer N a name of your own; any command or arithmetic '
    'that makes a buffer takes tag="name" too: [3] = [1] - [2] tag="difference"',
    "export": "export NAME=value  -  set an environment variable for the programs run here",
    "shell": "any program on the PATH runs as in a terminal (ls -l, ll, grep ...); a line "
    "starting with ! always goes to the shell; Ctrl+C in the command line stops it",
}
HELP.update(IMAGE_HELP)
# shorthands for shell commands
ALIASES = {"ll": "ls -lrt", "la": "ls -A"}
SHELL_TIMEOUT = 600  # seconds
SHELL_LINES = 2000  # lines of a program's output shown
MASK_HELP = """Making and removing mask layers
  mask new NAME [N]               an empty mask the size of buffer N
  mask load NAME file [N]         read a FITS or .dmask mask (0 = masked); it also shows
                                  as a buffer (1 = good, 0 = masked), kept up to date
  mask frombuf NAME N             a mask from buffer N (0 or NaN = masked)
  mask copy NAME FROM             a copy of the layer FROM
  mask union|intersect NAME A B ...   the pixels masked by any / all of A, B, ...
  mask rename NAME NEW;  mask delete NAME
  mask save NAME file             .dmask: a legacy bitmap; otherwise FITS (1 = good)
  mask tobuf NAME M|new           the mask as buffer M (1 = good, 0 = masked)
Drawing (ERASE unmasks instead)
  mask circle NAME X Y R          mask ellipse NAME X Y A B ANGLE
  mask box NAME X0 Y0 X1 Y1       mask polygon NAME X1 Y1 X2 Y2 X3 Y3 ...
  mask brush NAME R X1 Y1 X2 Y2 ...   circles of radius R along a path
  mask unpatch NAME X Y           unmask the connected masked area under X, Y
From the data and the sources
  mask nan NAME N                 the pixels of buffer N that are not finite
  mask range NAME N LO HI         the pixels of buffer N from LO to HI
  mask sources NAME [RUN=k] [SCALE=1.5] [KEEP=centre|none|number] [X0= Y0=]
                  [IDS=1,5,9] [WHERE='MAG_AUTO < 22 and FWHM_IMAGE < 3']
                                  the sources' Kron ellipses x SCALE, except the galaxy at
                                  X0, Y0 (by default the last ELLIPROF centre; before
                                  any fit, the largest source)
  mask segments NAME [RUN=k] [GROW=0] [WIDEN=0] [KEEP=...] [X0= Y0=] [ADD]
                                  the layer becomes the sources' pixels (their
                                  segmentation), everything else unmasked (ADD: add them
                                  to what the layer already masks); WIDEN=f grows each
                                  segment by GROW + f x its equivalent radius (bright
                                  stars' halos); X0= and Y0=
                                  may list several centres (X0=120,310 Y0=200,240): the
                                  sources there, and the segments under them, stay unmasked
Changing a layer
  mask grow|shrink NAME R;  mask invert|clear|fill NAME;  mask show|hide NAME
  mask role NAME fit|sky|candidate|none   what the layer is for: fit (left out of fits),
                                  sky (left out of the sky only; never used in a fit),
                                  candidate (proposed; promote it with role fit)
undo / redo undo or redo the last mask edits"""


class Session(ImageCommands):
    """The buffers, fits, source-finder runs and mask layers of a session."""

    def __init__(self):
        self.buffers: Dict[int, Image] = {}
        self.fits: Dict[str, FitResult] = {}  # the latest fit of each image (by its key)
        # every model of the latest iterative ELLIPROF fit of each image (models N)
        self.single_models: Dict[str, dict] = {}
        self.runs: List[detection.SourceRun] = []
        self.masks: Dict[str, Mask] = {}
        self.centre: Optional[Tuple[float, float]] = None
        self.folder = Path.cwd()
        self.variables: Dict[str, float] = {}
        self._process: Optional[subprocess.Popen] = None  # a shell command running
        self.mask_buffers: Dict[str, Tuple[int, Image]] = {}  # loaded masks, as buffers too
        self._undo: List[tuple] = []  # mask edits that can be undone, the last at the end
        self._redo: List[tuple] = []
        self.skies: Dict[str, skies.SkyResult] = {}  # sky measurements by their id
        self.sky_of: Dict[str, str] = {}  # the latest sky measurement of each image
        self.sky_buffer: Dict[str, tuple] = {}  # the buffer (number, image) each measured
        self.pairs: Dict[str, "PairRecord"] = {}  # the latest pair fit of each image
        self.planes: Dict[str, skies.PlaneFit] = {}  # the latest sky plane of each image
        self._pair_runs = 0
        self.stop_flag = False  # set by the window's Cancel for an iterative elliprof
        self._continued = ""  # the start of a line continued with \\

    # ---- helpers
    def buffer(self, text) -> Image:
        n = integer(text, "the buffer")
        if n not in self.buffers:
            raise CommandError(f"buffer {n} is empty")
        return self.buffers[n]

    def number_of(self, image: Image) -> Optional[int]:
        return next((n for n, b in self.buffers.items() if b is image), None)

    def free_buffer(self, after: int = 0) -> int:
        n = after + 1
        while n in self.buffers:
            n += 1
        return n

    def output_buffer(self, source: int, kind: str) -> int:
        """The buffer for a model or residual of ``source``: the one used before, or a free
        one."""
        image = self.buffers[source]
        name = f"{image.name} {kind}"
        for n, b in self.buffers.items():
            if b.name == name and b.family == image.key:
                return n
        return self.free_buffer(source)

    def path(self, text: str) -> Path:
        p = Path(text).expanduser()
        return p if p.is_absolute() else self.folder / p

    def mask_list(self, text: Optional[str], shape, fitting: bool = False) -> List[Mask]:
        """The layers named in ``text``.  For a fit (``fitting``), sky layers (which leave
        out the galaxies themselves) and candidates are refused."""
        if not text:
            return []
        found = []
        for name in [t for t in re.split(r"[,\s]+", text) if t]:
            if name not in self.masks:
                raise CommandError(f"there is no mask layer {name!r}")
            if self.masks[name].shape != tuple(shape):
                raise CommandError(f"the mask {name!r} is not the size of the image")
            role = self.masks[name].role
            if fitting and role == "sky":
                raise CommandError(
                    f"{name!r} is a sky-exclusion layer: it leaves out the galaxies too, so it "
                    "is never used in a fit (mask role NAME fit makes a layer a fitting one)"
                )
            if fitting and role == "candidate":
                raise CommandError(
                    f"{name!r} is a candidate: review it, then promote it (mask role {name} fit)"
                )
            found.append(self.masks[name])
        return found

    def run_for(self, image: Image, keys) -> detection.SourceRun:
        runs = [r for r in self.runs if r.image_key == image.key]
        if "RUN" in keys:
            k = integer(keys["RUN"], "RUN")
            run = next((r for r in self.runs if r.number == k), None)
            if run is None:
                raise CommandError(f"there is no source-finder run {k}")
            return run
        if not runs:
            raise CommandError(f"find the sources on {image.name} first (sources)")
        return runs[-1]

    # ---- running a line
    def execute(self, line: str) -> Reply:
        """Run a command line.  TAG= (or tag="..." after arithmetic) names the buffers it
        makes; every buffer made remembers the line that made it."""
        line = line.strip()
        if line.endswith("\\") and not line.startswith("!"):  # continued on the next line
            self._continued += line[:-1] + " "
            return Reply("…", show=None)
        if self._continued:
            line, self._continued = self._continued + line, ""
        tag = None
        m = re.search(r"""\s+tag\s*=\s*("[^"]*"|'[^']*'|\S+)""", line, flags=re.I)
        if m and not line.startswith("!"):  # tag= anywhere on the line
            tag = m.group(1).strip("'\"")
            line = (line[: m.start()] + line[m.end() :]).strip()
        before = {n: id(b) for n, b in self.buffers.items()}
        reply = self._execute(line)
        made = [n for n, b in self.buffers.items() if before.get(n) != id(b)]
        for n in made:
            image = self.buffers[n]
            if not image.note:
                image.note = line
                try:
                    image.header.add_history(f"AstroVista Lite: {line}"[:70])
                except Exception:
                    pass
            if tag:
                image.tag = tag
        if tag and made:
            reply.text += f"\ntagged {', '.join(str(n) for n in made)}: {tag}"
            reply.changed |= {"buffers"}
        elif tag and not made:
            reply.text += '\n(no new buffer to tag; tag N "name" tags a buffer)'
        return reply

    def _execute(self, line: str) -> Reply:
        line = re.sub(r"\\\s*\n\s*", " ", line)  # a line continued with \\
        if verb_of(line) == "pair":
            return self._pair_line(line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else "")
        m = re.fullmatch(r"(?:\[\s*(\d+)\s*\]|B(\d+))\s*=(?!=)\s*(.+)", line.split("#")[0].strip())
        if m:
            return self.c_arithmetic(int(m.group(1) or m.group(2)), m.group(3))
        if line.startswith("!"):
            return self.shell(line[1:].strip())
        first = line.split(None, 1)
        if first and first[0] in ALIASES:
            line = ALIASES[first[0]] + (" " + first[1] if len(first) > 1 else "")
        verb, words, keys = split_line(line)
        if not verb:
            return Reply()
        keys = self._with_variables(verb, keys)
        method = getattr(self, f"c_{verb}", None)
        if method is None:
            if shutil.which(verb):
                return self.shell(line)
            raise CommandError(
                f"unknown command {verb!r}: not an AstroVista command (help lists them) "
                "nor a program on the PATH"
            )
        return method(words, keys)

    def _with_variables(self, verb: str, keys):
        """Keyword values that are variables or expressions of them, as numbers (X0=xc,
        R1=2*re); other values (names, files, conditions) are left as they are."""
        if not self.variables or verb == "set":
            return keys
        from .imageops import evaluate

        out = dict(keys)
        for key, value in keys.items():
            names = {n.lower() for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value)}
            if names and names <= set(self.variables) | {
                "pi",
                "sqrt",
                "log10",
                "log",
                "exp",
                "abs",
            }:
                if names & set(self.variables):
                    try:
                        out[key] = repr(evaluate(value, self.variables))
                    except CommandError:
                        pass
        return out

    # ---- the shell
    def is_shell(self, line: str) -> bool:
        """Whether ``line`` goes to the shell (and so runs away from the window)."""
        line = line.strip()
        if line.startswith("!"):
            return True
        word = (line.split() or [""])[0]
        return word in ALIASES or (bool(word) and not hasattr(self, f"c_{word.lower()}"))

    def shell(self, line: str) -> Reply:
        """Run ``line`` in the shell, in the session's folder; its output is the reply."""
        if not line:
            return Reply()
        try:
            self._process = subprocess.Popen(
                line,
                shell=True,
                cwd=self.folder,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                errors="replace",
                start_new_session=True,  # Ctrl+C stops the program and its children
            )
            output, _ = self._process.communicate(timeout=SHELL_TIMEOUT)
            code = self._process.returncode
        except subprocess.TimeoutExpired:
            self.interrupt()
            raise CommandError(f"stopped after {SHELL_TIMEOUT} s: {line}")
        finally:
            self._process = None
        lines = output.rstrip("\n").splitlines()
        if len(lines) > SHELL_LINES:
            lines = lines[:SHELL_LINES] + [f"... ({len(lines) - SHELL_LINES} more lines)"]
        if code not in (0, None):
            lines.append(f"(exit {code})" if code > 0 else "(stopped)")
        reply = Reply("\n".join(lines), {"shell"})
        words = line.split()
        if words and words[0] == "ls":  # a listing: colours by kind, folders and FITS links
            folders = [w for w in words[1:] if not w.startswith("-")]
            base = (
                self.path(folders[-1])
                if len(folders) == 1 and self.path(folders[-1]).is_dir()
                else self.folder
            )
            reply.html = listing_html(lines, base)
        return reply

    def interrupt(self) -> bool:
        """Stop the shell command running (Ctrl+C); False when none is."""
        process = self._process
        if process is None or process.poll() is not None:
            return False
        try:
            os.killpg(process.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def c_tag(self, words, keys) -> Reply:
        """tag N "name": give buffer N a name of your own."""
        if len(words) < 2:
            raise CommandError(HELP["tag"])
        image = self.buffer(words[0])
        image.tag = " ".join(words[1:])
        return Reply(f"buffer {words[0]}: {image.tag}", {"buffers"})

    def c_pwd(self, words, keys) -> Reply:
        return Reply(str(self.folder))

    def c_export(self, words, keys) -> Reply:
        if not keys:
            raise CommandError(HELP["export"])
        os.environ.update({k: v for k, v in keys.items()})
        return Reply(" ".join(f"{k}={v}" for k, v in keys.items()))

    # ---- commands
    def c_help(self, words, keys) -> Reply:
        if words:
            name = words[0].lower()
            if name == "mask":
                return Reply(MASK_HELP)
            if name == "pair":
                return Reply(PAIR_HELP)
            if name in ("calc", "="):
                from . import calc

                return Reply(calc.__doc__.strip())
            if name not in HELP:
                raise CommandError(f"no help for {name!r}")
            return Reply(HELP[name])
        return Reply(
            "\n".join(HELP[k] for k in HELP)
            + "\n\nhelp mask: the mask commands.  help pair: fitting a pair.  Variables (set) can be used in keywords."
        )

    def c_cd(self, words, keys) -> Reply:
        folder = self.path(words[0]) if words else Path.home()
        if not folder.is_dir():
            raise CommandError(f"no folder {folder}")
        self.folder = folder.resolve()
        return Reply(str(self.folder))

    def c_rd(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(HELP["rd"])
        n = integer(words[0], "the buffer")
        name, ext = words[1], None
        m = re.fullmatch(r"(.+)\[([^\]]+)\]", name)
        if m and not self.path(name).exists():
            name, ext = m.group(1), m.group(2)
        path = self.path(name)
        if not path.exists():
            raise CommandError(f"no file {path}")
        hdu = None
        if ext is not None:
            hdu = int(ext) if ext.isdigit() else self._hdu_named(path, ext)
        try:
            image = read_fits(path, hdu)
        except (OSError, ValueError, IndexError) as e:
            raise CommandError(str(e))
        self.buffers[n] = image
        return Reply(
            f"buffer {n}: {image.name}, {image.shape[1]} x {image.shape[0]}",
            {"buffers", "data"},
            show=n,
        )

    @staticmethod
    def _hdu_named(path: Path, name: str) -> int:
        """The HDU called ``name`` (its EXTNAME), or ``name,version`` (EXTNAME, EXTVER)."""
        from astropy.io import fits

        extname, _, version = name.partition(",")
        with fits.open(path) as hdul:
            for k in image_hdus(path):
                h = hdul[k].header
                if str(h.get("EXTNAME", "")).strip().lower() != extname.strip().lower():
                    continue
                if version and int(h.get("EXTVER", 1)) != int(version):
                    continue
                return k
        raise CommandError(
            f"{path.name} has no image extension named {name!r} (fitsinfo lists them)"
        )

    def c_fitsinfo(self, words, keys) -> Reply:
        """The HDUs of a FITS file: which hold images, of what size, and which tables."""
        if not words:
            raise CommandError(HELP["fitsinfo"])
        from astropy.io import fits

        path = self.path(re.sub(r"\[[^\]]*\]$", "", words[0]))
        if not path.exists():
            raise CommandError(f"no file {path}")
        lines = [f"{path.name}:"]
        try:
            with fits.open(path) as hdul:
                for k, hdu in enumerate(hdul):
                    h = hdu.header
                    name = h.get("EXTNAME", "PRIMARY" if k == 0 else "")
                    version = f",{h['EXTVER']}" if "EXTVER" in h else ""
                    naxis = int(h.get("NAXIS", 0))
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        kind = f"table, {h.get('NAXIS2', 0)} rows x {h.get('TFIELDS', 0)} columns"
                    elif naxis >= 2:
                        size = " x ".join(str(h.get(f"NAXIS{a}")) for a in range(1, naxis + 1))
                        kind = f"image {size} (BITPIX {h.get('BITPIX')})"
                    else:
                        kind = "header only"
                    lines.append(f"  [{k}] {name}{version:<6}  {kind}")
        except OSError as e:
            raise CommandError(str(e))
        lines.append("Open one with  rd N file.fits[number]  or  rd N file.fits[NAME]")
        return Reply("\n".join(lines))

    def c_wd(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(HELP["wd"])
        image = self.buffer(words[0])
        path = write_fits(image, self.path(words[1]))
        return Reply(f"wrote {path}")

    def c_tv(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["tv"])
        self.buffer(words[0])
        return Reply(show=integer(words[0], "the buffer"))

    def c_buffers(self, words, keys) -> Reply:
        if not self.buffers:
            return Reply("no buffers")
        lines = [
            f"{n:3d}  {b.label}  {b.shape[1]} x {b.shape[0]}" + (f"   ({b.note})" if b.note else "")
            for n, b in sorted(self.buffers.items())
        ]
        return Reply("\n".join(lines))

    def c_close(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["close"])
        image = self.buffer(words[0])
        del self.buffers[integer(words[0], "the buffer")]
        return Reply(f"closed {image.name}", {"buffers"})

    def c_copy(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(HELP["copy"])
        image = self.buffer(words[0])
        m = integer(words[1], "the buffer")
        self.buffers[m] = Image(f"{image.name} copy", image.data.copy(), image.header.copy())
        return Reply(f"buffer {m}: copy of {image.name}", {"buffers", "data"}, show=m)

    def c_arithmetic(self, target: int, expression: str) -> Reply:
        """[target] = expression (calc.py)."""
        from .calc import Calculator

        expression = expression.split("#")[0].strip()
        calculator = Calculator(
            self.buffer,
            lambda name: self.masks[name].masked if name in self.masks else None,
            self.variables,
        )
        value = calculator.run(expression)
        if calculator.used:
            first = calculator.used[0]
            shape, header = first.shape, first.header.copy()
        else:
            masks = [m for m in self.masks.values() if m.name in expression]
            if not masks and target not in self.buffers:
                raise CommandError("the expression uses no buffer: its size is unknown")
            shape = masks[0].shape if masks else self.buffers[target].shape
            header = self.buffers[target].header.copy() if target in self.buffers else None
        if np.ndim(value) and np.shape(value) != tuple(shape):
            raise CommandError("the result is not the size of the buffers")
        data = np.broadcast_to(np.asarray(value, dtype=np.float32), shape).copy()
        image = Image(expression, data)
        if header is not None:
            image.header = header
        self.buffers[target] = image
        return Reply(f"buffer {target}: {expression}", {"buffers", "data"}, show=target)

    def c_calc(self, words, keys) -> Reply:
        if not words or "EXPR" not in keys:
            raise CommandError(HELP["calc"])
        return self.c_arithmetic(integer(words[0], "the buffer"), keys["EXPR"])

    def _good_pixels(self, image: Image, keys) -> np.ndarray:
        good = np.isfinite(image.data)
        for m in self.mask_list(keys.get("MASK"), image.shape):
            good &= ~m.masked
        if not good.any():
            raise CommandError("every pixel is masked")
        return good

    def c_stats(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["stats"])
        image = self.buffer(words[0])
        values = image.data[self._good_pixels(image, keys)].astype(np.float64)
        v = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "pixels": float(values.size),
        }
        text = "  ".join(f"{k} {val:.6g}" for k, val in v.items())
        return Reply(f"{image.name}: {text}", values=v)

    def c_sky(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["sky"])
        from astropy.stats import sigma_clipped_stats

        if any(w.upper() == "SEGMENT" for w in words[1:]):
            return self._sky_segment(integer(words[0], "the buffer"), self.buffer(words[0]), keys)
        image = self.buffer(words[0])
        values = image.data[self._good_pixels(image, keys)]
        mean, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=10)
        return Reply(
            f"{image.name}: sky {median:.6g} (clipped mean {mean:.6g}, sigma {std:.4g})",
            values={"sky": float(median), "sigma": float(std)},
        )

    def c_elliprof(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["elliprof"])
        n = integer(words[0], "the buffer")
        image = self.buffer(n)
        iterative = any(k in keys for k in ITERATIVE) or any(
            str(keys.get(k, "")).upper() == "AUTO" for k in ("R1", "NR")
        )
        growth = [k for k in ("START", "FACTOR", "TRYFACTOR", "STEP", "RADII") if k in keys]
        if growth:
            raise CommandError(
                f"{growth[0]}= is the pair fit's R1 growth; a single galaxy is fitted out to "
                "R1 at once"
            )
        for key in ("X0", "Y0", "R0", "R1") + (() if iterative else ("NR",)):
            if key not in keys:
                raise CommandError(f"{key}= is required ({HELP['elliprof']})")
        flags = {w.upper() for w in words[1:]}
        unknown = flags - {"RMSTAR", "RAW"}
        if unknown:
            raise CommandError(f"unknown word {sorted(unknown)[0]!r}")
        layers = self.mask_list(keys.get("MASK"), image.shape, fitting=True)
        minus = [t for t in re.split(r"[,\s]+", keys.get("MINUS", "")) if t]
        if minus:
            data = image.data.astype(np.float64)
            for text in minus:
                other = self.buffer(text)
                if other.shape != image.shape:
                    raise CommandError(f"buffer {text} is not the size of the image")
                data = data - other.data
            image = image.derived(image.name, data)
            image.family = self.buffer(words[0]).key
        for text in [t for t in re.split(r"[,\s]+", keys.get("MASKBUF", "")) if t]:
            # a buffer as a mask: masked where it is 0 or NaN (as elliprof reads masks)
            other = self.buffer(text)
            if other.shape != image.shape:
                raise CommandError(f"buffer {text} is not the size of the image")
            data = other.data
            layers.append(Mask(f"buffer {text}", (data == 0) | ~np.isfinite(data)))
        s = FitSettings(
            x0=number(keys["X0"], "X0"),
            y0=number(keys["Y0"], "Y0"),
            r0=number(keys["R0"], "R0"),
            r1=number(keys["R1"], "R1") if str(keys["R1"]).upper() != "AUTO" else 1e9,
            nr=(
                integer(keys["NR"], "NR")
                if str(keys.get("NR", "")).upper() not in ("", "AUTO")
                else 2
            ),
            niter=integer(keys.get("NITER", 10), "NITER"),
            sky=number(keys["SKY"], "SKY") if "SKY" in keys else None,
            rmstar="RMSTAR" in flags,
            mask_file=self.path(keys["MASKFILE"]) if "MASKFILE" in keys else None,
            extra_mask=maskops.union(layers),
            extra_mask_label=", ".join(m.name for m in layers),
            **self._native_keys(keys),
        )
        if s.mask_file is not None and not s.mask_file.exists():
            raise CommandError(f"no mask file {s.mask_file}")
        resid = self._resid_keys(keys)
        if iterative:
            return self._elliprof_iterative(n, image, keys, flags, s, resid, minus)
        try:
            result = fit(image, s)
            # RESIDMASK: the sources in the residual, masked, and the galaxy fitted again
            runs, added = [], np.zeros(image.shape, bool)
            for k in range(1, resid["rounds"] + 1):
                full = image.data.astype(np.float64) - (s.sky or 0.0) - result.model
                run, add = detection.residual_sources(
                    image,
                    full,
                    result.masked,
                    [(s.x0, s.y0)],
                    number=k,
                    thresh=resid["thresh"],
                    grow=resid["grow"],
                    core=resid["core"],
                )
                runs.append(run)
                if not add.any():
                    break
                added |= add
                s = dataclasses.replace(
                    s,
                    extra_mask=(s.extra_mask if s.extra_mask is not None else added) | added,
                    extra_mask_label=(s.extra_mask_label + ", residual sources").strip(", "),
                )
                result = fit(image, s)
        except (RuntimeError, ValueError) as e:
            raise CommandError(str(e))
        self.fits[image.key] = result
        self.centre = (s.x0, s.y0)
        reply = Reply(
            f"{len(result.profile)} isophotes in {result.seconds:.2f} s "
            f"(elliprof {result.version}); {int(result.masked.sum())} pixels masked",
            {"fits", "centre"},
        )
        residual = result.residual
        if "RAW" in flags:  # every pixel: the image minus the sky and the model
            residual = (image.data - (s.sky or 0.0) - result.model).astype(np.float32)
        if minus:
            reply.text += f"\n(fitted buffer {n} minus {', '.join(minus)})"
        if resid["rounds"]:
            reply.text += self._residual_outputs(
                self.buffer(n), runs, added if added.any() else None, "elliprof " + " ".join(words)
            )
            reply.changed |= {"sources", "masks"}
        for kind, key, data in (
            ("model", "TOMODEL", result.model),
            ("residual", "TORESID", residual),
        ):
            if key in keys:
                m = integer(keys[key], key)
                if m == n:
                    raise CommandError(f"{key} must not be the image's own buffer")
                self.buffers[m] = self.buffer(n).derived(f"{image.name} {kind}", data)
                reply.text += f"\n{kind} in buffer {m}"
                reply.changed |= {"buffers", "data"}
        return reply

    def _elliprof_iterative(self, n, image, keys, flags, fs, resid, minus) -> Reply:
        """elliprof with any of the pair fit's options (ITERATIVE): single.fit_single."""
        from . import pair as pairs
        from . import single

        masked = np.zeros(image.shape, bool)
        if fs.mask_file is not None:
            masked |= maskops.load(fs.mask_file, image.shape)
        if fs.extra_mask is not None:
            masked |= fs.extra_mask
        masked |= ~np.isfinite(image.data)
        measured = self.skies.get(self.sky_of.get(image.key, ""))
        if measured is not None and self.sky_stale(measured.id, image) is not None:
            measured = None
        if "SKY" in keys:
            sky = number(keys["SKY"], "SKY")
        elif measured is not None and np.isfinite(measured.level):
            sky = float(measured.level)
        else:
            raise CommandError(f"SKY= is needed (or a sky measurement: sky {n} SEGMENT)")
        factor = number(keys.get("SKYFACTOR", 1), "SKYFACTOR")
        sigma = measured.sigma if measured is not None else None
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            sigma = skies.local_noise(image.data, ~masked)
        x0, y0 = number(keys["X0"], "X0"), number(keys["Y0"], "Y0")
        light = np.asarray(image.data, float) - sky * factor
        notes = []
        if str(keys["R1"]).upper() == "AUTO":
            r1 = pairs.auto_r1(light, masked, (x0, y0), None, float(sigma))
            notes.append(f"R1=AUTO -> {r1:g} (where the light falls below the noise)")
        else:
            r1 = number(keys["R1"], "R1")
        r0 = number(keys["R0"], "R0")
        nr = 0 if str(keys.get("NR", "AUTO")).upper() == "AUTO" else integer(keys["NR"], "NR")

        def numbers(key, kind=float):
            return [kind(float(t)) for t in re.split(r"[,\s]+", keys.get(key, "")) if t]

        sky_iter = integer(keys.get("SKYITER", 0), "SKYITER")
        support = None
        if sky_iter:
            if measured is None or measured.support.sum() < 100:
                raise CommandError(
                    f"SKYITER needs the sky support of a current sky measurement (sky {n} SEGMENT)"
                )
            support = measured.support
        harmonics = str(keys.get("HARMONICS", "")).strip().upper()
        if harmonics and harmonics != "AUTO":
            raise CommandError("HARMONICS must be AUTO")
        tail = str(keys.get("TAIL", "elliprof")).strip().lower()
        if tail not in pairs.TAILS:
            raise CommandError("TAIL must be one of " + ", ".join(pairs.TAILS))
        st = single.SingleSettings(
            x0=x0,
            y0=y0,
            r0=r0,
            r1=r1,
            nr=nr,
            niter=fs.niter,
            sky=sky * factor,
            sky_base=sky,
            sky_factor=factor,
            sky_iter=sky_iter,
            sky_plane=integer(keys.get("SKYPLANE", 0), "SKYPLANE") != 0,
            rmstar=fs.rmstar,
            tie=fs.tie,
            rlaw=fs.rlaw,
            gain=fs.gain,
            fixctr=fs.fixctr,
            cos3x=fs.cos3x,
            cos4x=fs.cos4x,
            tail=tail,
            try_fix=numbers("TRYFIX", int),
            try_cos3x=numbers("TRYCOS3X", int),
            try_cos4x=numbers("TRYCOS4X", int),
            harmonic_search=harmonics == "AUTO",
            shape_tol=number(keys.get("SHAPETOL", 0), "SHAPETOL"),
            parallel=integer(keys.get("PARALLEL", 1), "PARALLEL"),
            runs=integer(keys.get("RUNS", 1), "RUNS"),
            jitter=number(keys.get("JITTER", 0.15), "JITTER"),
            seed=integer(keys["SEED"], "SEED") if "SEED" in keys else None,
            resid_rounds=resid["rounds"],
            resid_thresh=resid["thresh"],
            resid_grow=resid["grow"],
            resid_core=resid["core"],
        )
        if not 0 < r0 < r1:
            raise CommandError("0 < R0 < R1 is needed")
        if not 1 <= st.parallel <= pairs.MAX_PARALLEL or not 1 <= st.runs <= pairs.MAX_RUNS:
            raise CommandError(
                f"PARALLEL must be 1 to {pairs.MAX_PARALLEL}, RUNS 1 to {pairs.MAX_RUNS}"
            )
        self.stop_flag = False
        fitted_data = np.asarray(image.data, float)
        result = single.fit_single(
            fitted_data,
            masked,
            st,
            header=image.header,
            support=support,
            image_obj=image,
            cancelled=lambda: self.stop_flag,
            progress=getattr(self, "progress_hook", None),
        )
        if result.status == "cancelled" or self.stop_flag:
            return Reply("elliprof cancelled: nothing kept, no buffer changed")
        if not result.complete:
            raise CommandError(f"ELLIPROF found no fit that passes: {result.reason}")
        used = result.mask if result.mask is not None else masked
        residual = np.where(used, np.nan, result.residual).astype(np.float32)
        import elliprof

        fitres = FitResult(
            profile=result.profile,
            model=result.model.astype(np.float32),
            residual=residual,
            masked=used,
            ellipses=pairs.profile_ellipses(result.profile),
            command="elliprof " + " ".join(f"{k}={v}" for k, v in keys.items()),
            version=elliprof.__version__,
            seconds=result.seconds,
        )
        self.fits[image.key] = fitres
        self.single_models[image.key] = {
            "rows": result.models,
            "store": result.model_store,
            "data": fitted_data,
            "centre": (x0, y0),
            "annulus": (single.SCORE_INNER * r0, single.SCORE_OUTER * r1),
        }
        self.centre = (x0, y0)
        reply = Reply("\n".join(notes + [single.summary(result)]), {"fits", "centre", "models"})
        if "RAW" in flags:
            residual = result.residual.astype(np.float32)
        if minus:
            reply.text += f"\n(fitted buffer {n} minus {', '.join(minus)})"
        made = []
        for kind, key, data in (
            ("model", "TOMODEL", fitres.model),
            ("residual", "TORESID", residual),
        ):
            if key in keys:
                m = integer(keys[key], key)
                if m == n:
                    raise CommandError(f"{key} must not be the image's own buffer")
                self.buffers[m] = self.buffer(n).derived(f"{image.name} {kind}", data)
                made.append(f"{m} {kind}")
        source = self.buffer(n)
        stale = [
            m
            for m, b in self.buffers.items()
            if b.family == source.key and b.name.startswith(f"{source.name} elliprof mean")
        ]
        for m in stale:
            del self.buffers[m]
        e = result.ensemble
        extra = []
        if result.sky_map is not None:
            extra.append(("elliprof sky", result.sky_map))
        if e:
            extra += [
                ("elliprof mean model", e["model"]),
                ("elliprof mean residual", e["residual"]),
                ("elliprof mean model spread", e["spread"]),
            ]
            total = len(e["fits"])
            for k, f in enumerate(e["fits"], 1):
                extra.append((f"elliprof mean fit {k} of {total} model", f["model"]))
                extra.append((f"elliprof mean fit {k} of {total} residual", f["residual"]))
        taken = {n} | {integer(keys[k], k) for k in ("TOMODEL", "TORESID") if k in keys}
        for kind, data in extra:
            m = self.output_buffer(n, kind)
            while m in taken:
                m = self.free_buffer(m)
            taken.add(m)
            self.buffers[m] = source.derived(f"{source.name} {kind}", data)
            made.append(f"{m} {kind}")
        if made:
            reply.text += "\nbuffers: " + "; ".join(made)
            reply.changed |= {"buffers", "data"}
        if resid["rounds"]:
            reply.text += self._residual_outputs(
                source,
                result.resid_runs,
                result.resid_mask,
                "elliprof " + " ".join(f"{k}={v}" for k, v in keys.items()),
            )
            reply.changed |= {"sources", "masks"}
        return reply

    def c_models(self, words, keys) -> Reply:
        """The models of buffer N's latest iterative ELLIPROF fit: listed (STEP= one
        category), put in buffers (SHOW=), or averaged (AVERAGE=)."""
        from . import single

        if not words:
            raise CommandError(HELP["models"])
        n = integer(words[0], "the buffer")
        source = self.buffer(n)
        rec = self.single_models.get(source.key)
        if rec is None:
            raise CommandError(f"buffer {n} has no iterative ELLIPROF fit (elliprof ... R1=AUTO)")
        rows = rec["rows"]
        if "STEP" in keys:
            want = str(keys["STEP"]).strip().lower()
            if want not in {c[0] for c in single.CATEGORIES}:
                raise CommandError(
                    "STEP must be one of " + ", ".join(c[0] for c in single.CATEGORIES)
                )
            rows = [r for r in rows if single.category(r["stage"]) == want]

        def numbers(key):
            out = []
            for tok in re.split(r"[,\s]+", str(keys[key]).strip()):
                if not tok:
                    continue
                k = integer(tok.lstrip("#"), key)
                if k not in rec["store"]:
                    raise CommandError(f"model #{k} is not one of buffer {n}'s complete models")
                out.append(k)
            return out

        made, text = [], []
        taken = {n}

        def put(kind, data):
            m = self.output_buffer(n, kind)
            while m in taken:
                m = self.free_buffer(m)
            taken.add(m)
            self.buffers[m] = source.derived(f"{source.name} {kind}", data.astype(np.float32))
            made.append(f"{m} {kind}")

        by_n = {r["n"]: r for r in rec["rows"]}
        if "SHOW" in keys:
            for k in numbers("SHOW"):
                one = single.average_models(
                    rec["data"], rec["store"], [k], rec["centre"], *rec["annulus"]
                )
                tag = f"model #{k} {by_n[k]['stage']}"
                put(tag, one["model"])
                put(f"{tag} residual", one["residual"])
        if "AVERAGE" in keys:
            picked = numbers("AVERAGE")
            if len(picked) < 2:
                raise CommandError("AVERAGE needs two models or more")
            mean = single.average_models(
                rec["data"], rec["store"], picked, rec["centre"], *rec["annulus"]
            )
            tag = "models mean of " + ",".join(f"#{k}" for k in picked)
            put(f"{tag} model", mean["model"])
            put(f"{tag} residual", mean["residual"])
            sc = mean["score"]
            inner, outer = rec["annulus"]

            def alone(k):  # on the same annulus as the mean
                v = single.average_models(
                    rec["data"], rec["store"], [k], rec["centre"], *rec["annulus"]
                )["score"]
                return "-" if v is None else f"{v:.3f}"

            text.append(
                f"mean of {len(picked)} models ({', '.join(f'#{k}' for k in picked)}): residual "
                + (f"{sc:.3f}" if sc is not None else "-")
                + f" (RMS over the noise, {inner:.1f} to {outer:.1f} px); each alone: "
                + ", ".join(f"#{k} {alone(k)}" for k in picked)
            )
        if not made:
            text = single.models_table(rows)
        if made:
            text.append("buffers: " + "; ".join(made))
        return Reply("\n".join(text), {"buffers", "data"} if made else set())

    def _residual_outputs(self, image: Image, runs, added, line: str) -> str:
        """RESIDMASK: each round's sources in the residual as source runs, and the pixels
        they added to the fitting mask as the layer residsources (role fit), for later
        fits; what to say about them."""
        found = []
        for run in runs:
            run.number = max((r.number for r in self.runs), default=0) + 1
            run.image_key = image.key
            run.image_name = f"{image.name} residual"
            self.runs.append(run)
            found.append(f"run {run.number}: {len(run)}")
        text = "\nsources in the residual: " + "; ".join(found)
        if added is not None and added.any():
            name = "residsources"
            self._remember(name)
            self.masks[name] = Mask(
                name,
                added.copy(),
                self._colour(name),
                visible=False,
                source=f"sources in the residual of: {line}",
                role="fit",
            )
            text += f"\nmask layer {name}: {int(added.sum())} px (role fit)"
        return text

    @staticmethod
    def _resid_keys(keys) -> dict:
        """RESIDMASK= and its settings, checked."""
        out = {
            "rounds": integer(keys.get("RESIDMASK", 0), "RESIDMASK"),
            "thresh": number(keys.get("RESIDTHRESH", 8), "RESIDTHRESH"),
            "grow": integer(keys.get("RESIDGROW", 2), "RESIDGROW"),
            "core": number(keys.get("RESIDCORE", 10), "RESIDCORE"),
        }
        if not 0 <= out["rounds"] <= 5:
            raise CommandError("RESIDMASK must be 0 to 5 (rounds)")
        if not 1 <= out["thresh"] <= 100:
            raise CommandError("RESIDTHRESH must be 1 to 100 (times the local RMS)")
        return out

    @staticmethod
    def _native_keys(keys) -> dict:
        """The optional ELLIPROF keywords of a command (MODE= or COS3X=/COS4X=, FIXCTR=,
        RLAW=, GAIN=, TIE=)."""
        out = {}
        if "MODE" in keys:
            if "COS3X" in keys or "COS4X" in keys:
                raise CommandError("give MODE= or COS3X=/COS4X=, not both")
            mode = keys["MODE"].strip().lower()
            if mode not in MODES:
                raise CommandError(f"unknown MODE {mode!r} ({', '.join(MODES)})")
            out["cos3x"], out["cos4x"] = MODES[mode]
        for key in ("COS3X", "COS4X", "FIXCTR", "RLAW", "TIE"):
            if key in keys:
                out[key.lower()] = integer(keys[key], key)
        if "GAIN" in keys:
            out["gain"] = number(keys["GAIN"], "GAIN")
        if out.get("fixctr", 0) not in (0, 1, 2):
            raise CommandError("FIXCTR must be 0, 1 or 2")
        return out

    def _sky_segment(self, n: int, image: Image, keys) -> Reply:
        layers = self.mask_list(keys.get("MASK"), image.shape)
        settings = {
            "thresh": number(keys.get("THRESH", skies.DEFAULTS["THRESH"]), "THRESH"),
            "minarea": integer(keys.get("MINAREA", skies.DEFAULTS["MINAREA"]), "MINAREA"),
            "grow": number(keys.get("GROW", skies.DEFAULTS["GROW"]), "GROW"),
            "cells": integer(keys.get("CELLS", skies.DEFAULTS["CELLS"]), "CELLS"),
            "back_size": integer(keys.get("BACK_SIZE", 0), "BACK_SIZE"),
            "fwhm": number(keys.get("FWHM", 0), "FWHM"),
            "pad": number(keys["PAD"], "PAD") if "PAD" in keys else None,
        }
        outputs = {k: integer(keys[k], k) for k in ("TOMAP", "TOPLANE") if k in keys}
        if n in outputs.values():
            raise CommandError("TOMAP and TOPLANE must not be the measured buffer")
        if len(set(outputs.values())) < len(outputs):
            raise CommandError("TOMAP and TOPLANE must be different buffers")
        target = keys.get("TOMASK", "skymask")
        if target in self.masks and self.masks[target].role not in ("sky", ""):
            raise CommandError(
                f"{target!r} is a {maskops.ROLES[self.masks[target].role]} layer: give the "
                "sky mask another name (TOMASK=)"
            )
        try:
            result = skies.measure(
                image.data,
                image.label,
                exclude=maskops.union(layers),
                layers=",".join(m.name for m in layers),
                **settings,
            )
        except (RuntimeError, ValueError) as e:
            raise CommandError(str(e))
        result.settings["layer_checksums"] = {m.name: _mask_sum(m) for m in layers}
        self.skies[result.id] = result
        self.sky_of[image.key] = result.id
        self.sky_buffer[result.id] = (n, image)
        self._remember(target)
        self.masks[target] = Mask(
            target,
            result.exclusion.copy(),
            self._colour(target),
            visible=False,
            source=f"sky segmentation {result.id} ({settings['thresh']:g} sigma)",
            role="sky",
        )
        reply = Reply(result.summary(), {"masks", "sky"})
        reply.text += f"\nsky-exclusion layer {target} (shown on request; never used in a fit)"
        reply.values = {"sky_id": 0.0}
        if result.state == "usable_support":
            # "segment": a segmentation sky (the panels show it; Use this sky takes it)
            reply.values = {"sky": result.level, "sigma": result.sigma, "segment": 1.0}
        if "TOMAP" in outputs:
            m = np.full(image.shape, np.nan, np.float32)
            for c in result.cells:
                if c.usable:
                    m[c.y0 : c.y1, c.x0 : c.x1] = c.median
            self.buffers[outputs["TOMAP"]] = image.derived(f"{image.name} sky cells", m)
            reply.text += f"\ncell medians in buffer {outputs['TOMAP']}"
        if "TOPLANE" in outputs:
            if result.plane is None:
                reply.text += "\n(no plane to write)"
            else:
                self.buffers[outputs["TOPLANE"]] = image.derived(
                    f"{image.name} sky plane", skies.plane_image(result.plane, image.shape)
                )
                reply.text += f"\nfitted plane in buffer {outputs['TOPLANE']}"
        if outputs:
            reply.changed |= {"buffers", "data"}
        return reply

    def sky_stale(self, sky_id: str, image: Optional[Image] = None) -> Optional[str]:
        """Why a sky measurement no longer fits ``image`` (by default the buffer it was
        measured on; None when it still does)."""
        result = self.skies.get(sky_id)
        if result is None:
            return "no such measurement"
        if image is None:
            n, measured = self.sky_buffer.get(sky_id, (None, None))
            if n not in self.buffers or self.buffers[n] is not measured:
                return f"buffer {n}, which was measured, holds another image now"
            image = measured
        if image.shape != result.support.shape or skies.checksum(image.data) != result.image_id:
            return "the image has changed since the sky was measured"
        for name, value in result.settings.get("layer_checksums", {}).items():
            m = self.masks.get(name)
            if m is None or _mask_sum(m) != value:
                return f"the mask layer {name} has changed since the sky was measured"
        return None

    def c_skyplane(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["skyplane"])
        n = integer(words[0], "the buffer")
        image = self.buffer(n)
        layers = self.mask_list(keys.get("MASK"), image.shape)
        settings = {
            "thresh": number(keys.get("THRESH", skies.DEFAULTS["THRESH"]), "THRESH"),
            "minarea": integer(keys.get("MINAREA", skies.DEFAULTS["MINAREA"]), "MINAREA"),
            "grow": number(keys.get("GROW", skies.DEFAULTS["GROW"]), "GROW"),
            "box": integer(keys.get("BOX", skies.PLANE_DEFAULTS["BOX"]), "BOX"),
            "passes": integer(keys.get("PASSES", skies.PLANE_DEFAULTS["PASSES"]), "PASSES"),
            "fwhm": number(keys.get("FWHM", 0), "FWHM"),
            "pad": number(keys["PAD"], "PAD") if "PAD" in keys else None,
        }
        outputs = {k: integer(keys[k], k) for k in ("TOPLANE", "TOFLAT", "TOSUB") if k in keys}
        if n in outputs.values():
            raise CommandError("TOPLANE, TOFLAT and TOSUB must not be the measured buffer")
        if len(set(outputs.values())) < len(outputs):
            raise CommandError("TOPLANE, TOFLAT and TOSUB must be different buffers")
        target = keys.get("TOMASK", "skymask")
        if target in self.masks and self.masks[target].role not in ("sky", ""):
            raise CommandError(
                f"{target!r} is a {maskops.ROLES[self.masks[target].role]} layer: give the "
                "sky mask another name (TOMASK=)"
            )
        try:
            fit = skies.fit_plane_sky(
                image.data,
                image.label,
                exclude=maskops.union(layers),
                layers=",".join(m.name for m in layers),
                **settings,
            )
        except (RuntimeError, ValueError) as e:
            raise CommandError(str(e))
        fit.source = image.name
        self.planes[image.key] = fit
        reply = Reply(fit.summary(), {"masks", "sky"})
        self._remember(target)
        self.masks[target] = Mask(
            target,
            fit.sky.exclusion.copy(),
            self._colour(target),
            visible=False,
            source=f"sky plane of {image.label} ({settings['thresh']:g} sigma)",
            role="sky",
        )
        reply.text += f"\nsky-exclusion layer {target} (shown on request; never used in a fit)"
        if fit.plane is None:
            return reply
        p = fit.plane
        reply.values = {
            "sky": p["s0"],
            "sigma": fit.sky.sigma,
            "skygx": p["gx"],
            "skygy": p["gy"],
        }
        tilt = fit.tilt(image.shape)
        if "TOPLANE" in outputs:
            plane = image.derived(f"{image.name} sky plane", tilt + p["s0"])
            plane.display = "smooth"
            self.buffers[outputs["TOPLANE"]] = plane
            reply.text += f"\nthe plane in buffer {outputs['TOPLANE']}"
        if "TOSUB" in outputs:
            self.buffers[outputs["TOSUB"]] = Image(
                f"{image.name} minus sky plane",
                (image.data - tilt - p["s0"]).astype(np.float32),
                image.header.copy(),
            )
            reply.text += f"\nthe image minus the plane in buffer {outputs['TOSUB']} (its sky is 0)"
        if "TOFLAT" in outputs:
            out = outputs["TOFLAT"]
            flat = Image(f"{image.name} flat", fit.flat, image.header.copy())
            flat.note = f"skyplane {n}: the tilt removed, the level {p['s0']:.6g} kept"
            self.buffers[out] = flat
            # the measurement on the flat image is that image's sky (for elliprof and pair)
            result = fit.sky
            result.image_name = flat.label
            result.settings["layer_checksums"] = {m.name: _mask_sum(m) for m in layers}
            self.skies[result.id] = result
            self.sky_of[flat.key] = result.id
            self.sky_buffer[result.id] = (out, flat)
            reply.text += (
                f"\nthe flat image (the tilt removed, the level kept) in buffer {out}; its sky: "
                f"{result.level:.6g} [{result.state}], measurement {result.id}"
            )
            reply.show = out
        reply.changed |= {"buffers", "data"} if outputs else set()
        return reply

    def c_window(self, words, keys) -> Reply:
        if not words or "BOX" not in keys:
            raise CommandError(HELP["window"])
        n = integer(words[0], "the buffer")
        image = self.buffer(n)
        parts = [t for t in re.split(r"[,\s]+", keys["BOX"]) if t]
        if len(parts) != 4:
            raise CommandError("BOX=x0,y0,x1,y1: four whole numbers")
        x0, y0, x1, y1 = (integer(t, "BOX") for t in parts)
        ny, nx = image.shape
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, nx), min(y1, ny)
        if cx1 <= cx0 or cy1 <= cy0:
            raise CommandError(f"the box {x0},{y0},{x1},{y1} holds no pixel of the image")
        note = ""
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            note = f" (clipped to the image: {cx0},{cy0},{cx1},{cy1})"
        out = integer(keys["OUT"], "OUT") if "OUT" in keys else self.free_buffer(n)
        if out == n:
            raise CommandError("OUT must be another buffer: the original stays")
        header = image.header.copy()
        for k in ("CRPIX1", "CRPIX2"):
            if k in header:
                header[k] = header[k] - (cx0 if k.endswith("1") else cy0)
        header["LTV1"] = header.get("LTV1", 0) - cx0
        header["LTV2"] = header.get("LTV2", 0) - cy0
        for k in ("CNPIX1", "CNPIX2"):
            header.remove(k, ignore_missing=True)
        header["NAXIS1"], header["NAXIS2"] = cx1 - cx0, cy1 - cy0
        name = f"{image.name} window {cx0},{cy0},{cx1},{cy1}"
        self.buffers[out] = Image(name, image.data[cy0:cy1, cx0:cx1].copy(), header)
        reply = Reply(
            f"buffer {out}: {name}, {cx1 - cx0} x {cy1 - cy0}{note}; positions there are "
            f"those here minus ({cx0}, {cy0}) (LTV1, LTV2 in its header)",
            {"buffers", "data"},
            show=out,
        )
        names = keys.get("MASKS", "")
        layers = (
            [m for m in self.masks.values() if m.shape == image.shape]
            if names.lower() == "all"
            else self.mask_list(names, image.shape)
        )
        for m in layers:
            new = f"{m.name}_w"
            self._remember(new)
            self.masks[new] = Mask(
                new,
                m.masked[cy0:cy1, cx0:cx1].copy(),
                self._colour(new),
                m.visible,
                m.opacity,
                f"window of {m.name}",
                m.role,
            )
            reply.text += f"\nmask {new}: {m.name} cut to the window"
            reply.changed |= {"masks"}
        return reply

    # ---- pairs of galaxies
    def pair_prepare(self, text: str) -> "PairJob":
        """Read a pair command and fix everything its run uses (in the window's thread)."""
        try:
            words, shared, own = pairs.parse(text)
            shared = self._with_variables("pair", shared)
            own = {c: self._with_variables("pair", k) for c, k in own.items()}
            settings = pairs.settings_from(shared)
            auto = any(
                str(k.get(w, "")).upper() == "AUTO" for k in own.values() for w in ("R1", "START")
            )
            if not auto:
                components = pairs.components_from(own)
        except pairs.PairError as e:
            raise CommandError(f"{e}\n{HELP['pair']}")
        if len(words) != 1:
            raise CommandError("give one buffer: pair N ...")
        n = integer(words[0], "the buffer")
        image = self.buffer(n)
        layers = self.mask_list(shared.get("MASK"), image.shape, fitting=True)
        masked = maskops.union(layers)
        masked = np.zeros(image.shape, bool) if masked is None else masked.copy()
        label = [m.name for m in layers]
        maskbuf = [t for t in re.split(r"[,\s]+", shared.get("MASKBUF", "")) if t]
        for text_ in maskbuf:
            other = self.buffer(text_)
            if other.shape != image.shape:
                raise CommandError(f"buffer {text_} is not the size of the image")
            masked |= (other.data == 0) | ~np.isfinite(other.data)
            label.append(f"buffer {text_}")
        if "MASKFILE" in shared:
            path = self.path(shared["MASKFILE"])
            if not path.exists():
                raise CommandError(f"no mask file {path}")
            masked |= maskops.load(path, image.shape)
            label.append(path.name)
        uncovered = _uncover_centres(masked, own)
        # output buffers: given, or free ones; never the image, a mask buffer or each other
        taken = {n} | {integer(t, "MASKBUF") for t in maskbuf}
        outputs = {}
        for key in ("TOMODELA", "TOMODELB", "TOMODEL", "TORESID"):
            if key in shared:
                m = integer(shared[key], key)
                if m in taken:
                    raise CommandError(
                        f"{key}={m} is the image, a mask buffer or another output of this pair"
                    )
                outputs[key] = m
                taken.add(m)
        for key in ("TOMODELA", "TOMODELB", "TOMODEL", "TORESID"):
            if key not in outputs:
                kind = {
                    "TOMODELA": "pair A model",
                    "TOMODELB": "pair B model",
                    "TOMODEL": "pair A+B model",
                    "TORESID": "pair residual",
                }[key]
                m = self.output_buffer(n, kind)
                while m in taken:
                    m = self.free_buffer(m)
                outputs[key] = m
                taken.add(m)
        if "VIEWS" in shared:
            for key, kind in (
                ("VIEWI", "pair image - sky"),
                ("VIEWA", "pair image - sky - B"),
                ("VIEWB", "pair image - sky - A"),
            ):
                m = self.output_buffer(n, kind)
                while m in taken:
                    m = self.free_buffer(m)
                outputs[key] = m
                taken.add(m)
        run = next(
            (
                r
                for r in reversed(self.runs)
                if r.image_key == image.key
                and r.segmentation is not None
                and r.segmentation.shape == image.shape
            ),
            None,
        )
        sky_id, sigma = "", None
        candidate = self.skies.get(self.sky_of.get(image.key, ""))
        if candidate is not None and self.sky_stale(candidate.id, image) is not None:
            candidate = None  # its support no longer fits the image
        base = settings.sky_base
        if candidate is not None and np.isfinite(candidate.level):
            if abs(candidate.level - base) <= 1e-4 + 1e-9 * abs(base):  # the panel's 4 decimals
                sky_id, sigma = candidate.id, candidate.sigma
        if candidate is not None and candidate.support.sum() < 100:
            candidate = None
        auto_notes, auto_flags, sigma_auto = [], {}, None
        if auto:
            auto_flags = {
                c: (
                    str(own[c].get("R1", "")).upper() == "AUTO",
                    str(own[c].get("START", "")).upper() == "AUTO",
                )
                for c in ("A", "B")
            }
            sigma_auto = sigma
            if sigma_auto is None or not np.isfinite(sigma_auto) or sigma_auto <= 0:
                sigma_auto = skies.local_noise(image.data, np.isfinite(image.data) & ~masked)
            light = np.asarray(image.data, float) - settings.sky
            auto_notes = pairs.resolve_auto(own, light, masked, float(sigma_auto))
            try:
                components = pairs.components_from(own)
            except pairs.PairError as e:
                raise CommandError(f"{e}\n{HELP['pair']}")
        if settings.sky_iter and candidate is None:
            raise CommandError(
                "SKYITER needs the sky support of a current sky measurement of this image "
                f"(sky {n} SEGMENT)"
            )
        self._pair_runs += 1
        provenance = {
            "file": str(image.path) if image.path else None,
            "image": image.name,
            "buffer": n,
            "shape": list(image.shape),
            "dtype_in_buffer": str(image.data.dtype),
            "converted_to": "float64 for the pair arithmetic",
            "snapshot_sha256": hashlib.sha256(
                np.ascontiguousarray(image.data).tobytes()
            ).hexdigest(),
            "window_offset_ltv": [image.header.get("LTV1"), image.header.get("LTV2")],
            "source_run": (
                None
                if run is None
                else {
                    "number": run.number,
                    "command": "sources 1 "
                    + " ".join(
                        f"{k}={v}"
                        for k, v in run.settings.items()
                        if v != detection.DEFAULTS.get(k)
                    ),
                }
            ),
        }
        try:
            snap = pairs.prepare(
                image.data,
                image.label,
                image.header,
                components,
                settings,
                fit_mask=masked,
                fit_mask_label=",".join(label),
                run=run,
                run_id=self._pair_runs,
                sky_id=sky_id,
                sky_sigma=sigma,
                provenance=provenance,
            )
        except pairs.PairError as e:
            raise CommandError(str(e))
        snap.notes.extend(uncovered + auto_notes)
        if auto:
            snap.auto, snap.auto_sigma = auto_flags, float(sigma_auto)
        job = PairJob(n, image, snap, outputs, "pair " + text.strip())
        job.measurement = candidate
        return job

    def pair_commit(self, job: "PairJob", result: pairs.PairResult) -> Reply:
        """Put a finished pair run into the session (in the window's thread)."""
        image, snap, n = job.image, job.snap, job.number
        if job.number not in self.buffers or self.buffers[n] is not image:
            pass  # the buffer was replaced meanwhile: the result keeps its own snapshot
        if result.snapshot is not None:  # the fit kept: its own sky factor and variant
            job.snap = snap = result.snapshot
        if not result.quality:
            result.quality = pairs.quality(result, snap)
        measured = job.measurement
        if result.residual is not None and measured is not None:
            result.sky_check = skies.check(measured, result.residual)
            s = snap.settings
            level = skies.residual_level(
                result.residual, measured.support & ~snap.fit_mask, s.sky_base, s.sky_factor
            )
            result.sky_check.update({f"clipped_{k}": v for k, v in level.items()})
        self.pairs[image.key] = PairRecord(job, result)
        reply = Reply(pairs.summary(result, snap), {"pairs"})
        check = result.sky_check
        if "clipped_median" in check:
            reply.text += (
                f"\nresidual on the sky support: clipped median {check['clipped_median']:+.4g}"
                f", sigma {check['clipped_sigma']:.4g} ({check['clipped_pixels']} px)"
            )
            if "clipped_next_factor" in check:
                reply.text += (
                    f"; next sky factor {check['clipped_next_factor']:.5f} (sky "
                    f"{check['clipped_next_sky']:.6g})"
                )
                reply.values = {
                    "skymedian": check["clipped_median"],
                    "skyfactor": check["clipped_next_factor"],
                }
        if result.complete:
            a, b = result.commits["A"].model, result.commits["B"].model
            views = {
                "TOMODELA": (f"{image.name} pair A model ({snap.components['A'].label})", a),
                "TOMODELB": (f"{image.name} pair B model ({snap.components['B'].label})", b),
                "TOMODEL": (f"{image.name} pair A+B model", result.total),
                "TORESID": (f"{image.name} pair residual", result.residual),
                "VIEWI": (f"{image.name} pair image - sky", snap.image - snap.settings.sky),
                "VIEWA": (f"{image.name} pair image - sky - B", snap.image - snap.settings.sky - b),
                "VIEWB": (f"{image.name} pair image - sky - A", snap.image - snap.settings.sky - a),
            }
            made = []
            for key, m in job.outputs.items():
                name, data = views[key]
                self.buffers[m] = image.derived(name, data)
                self.buffers[m].note = job.line
                made.append(f"{m} {name.split(' pair ', 1)[1]}")
            e = result.ensemble
            # the mean's buffers of an earlier fit of this image go first (a fit with fewer
            # members, or none, must not leave them behind)
            stale = [
                m
                for m, b in self.buffers.items()
                if b.family == image.key
                and m not in job.outputs.values()
                and m != n
                and (
                    b.name.startswith(f"{image.name} pair mean")
                    or b.name == f"{image.name} pair model spread"
                )
            ]
            for m in stale:
                del self.buffers[m]
            if stale:
                made.append("removed the earlier mean's " + ", ".join(str(m) for m in stale))
            if e:
                taken = set(job.outputs.values()) | {n}
                outputs = [
                    ("pair mean A model", e["A"]),
                    ("pair mean B model", e["B"]),
                    ("pair mean A+B model", e["model"]),
                ]
                if e.get("sky_map") is not None:
                    outputs.append(("pair mean sky", e["sky_map"]))
                outputs += [
                    ("pair mean residual", e["residual"]),
                    ("pair model spread", e["spread"]),
                ]
                total = len(e.get("fits", []))
                for k, f in enumerate(e.get("fits", []), 1):
                    outputs.append((f"pair mean fit {k} of {total} model", f["model"]))
                    outputs.append((f"pair mean fit {k} of {total} residual", f["residual"]))
                for kind, data in outputs:
                    m = self.output_buffer(n, kind)
                    while m in taken:
                        m = self.free_buffer(m)
                    taken.add(m)
                    self.buffers[m] = image.derived(f"{image.name} {kind}", data)
                    self.buffers[m].note = job.line
                    made.append(f"{m} {kind.split('pair ', 1)[1]}")
            reply.text += "\nbuffers: " + "; ".join(made)
            reply.changed |= {"buffers", "data"}
            if result.resid_runs:
                reply.text += self._residual_outputs(
                    image, result.resid_runs, result.resid_mask, job.line
                )
                reply.changed |= {"sources", "masks"}
        return reply

    def _pair_line(self, text: str) -> Reply:
        """pair, run to the end in this thread (scripts, the terminal without a window)."""
        if not text.strip():
            raise CommandError(PAIR_HELP)
        job = self.pair_prepare(text)
        support = job.measurement.support if job.measurement is not None else None
        result = pairs.run_search(job.snap, support)
        return self.pair_commit(job, result)

    def c_pair(self, words, keys) -> Reply:  # reached only through _pair_line
        raise CommandError(PAIR_HELP)

    def c_pairstep(self, words, keys) -> Reply:
        if len(words) != 2:
            raise CommandError(HELP["pairstep"])
        image = self.buffer(words[0])
        record = self.pairs.get(image.key)
        if record is None:
            raise CommandError(f"no pair fit of {image.name} yet (pair {words[0]} ...)")
        k = integer(words[1], "the call")
        history = record.result.history
        if not 1 <= k <= len(history):
            raise CommandError(f"the last pair fit made calls 1 to {len(history)}")
        h = history[k - 1]
        snap = record.job.snap
        data = snap.image if h["companion_model"] is None else snap.image - h["companion_model"]
        what = f"pair call {k}: {h['galaxy']} cycle {h['cycle']} R1 {h['r1']:.1f}"
        n_in = self.output_buffer(integer(words[0], "the buffer"), f"{what} input")
        self.buffers[n_in] = image.derived(f"{image.name} {what} input {h['input']}", data)
        made = [f"{n_in} the input ({h['input']})"]
        show = n_in
        if h["model"] is not None:
            n_model = self.output_buffer(integer(words[0], "the buffer"), f"{what} model")
            while n_model == n_in:
                n_model = self.free_buffer(n_model)
            self.buffers[n_model] = image.derived(f"{image.name} {what} model", h["model"])
            made.append(f"{n_model} its model")
            show = n_model
        status = "passed" if h["passed"] else f"failed: {h['reason']}"
        reply = Reply(
            f"call {k}: {h['galaxy']} on {h['input']} (companion {h['companion']}), R1 "
            f"{h['r1']:.1f}, {status}; buffers: " + "; ".join(made),
            {"buffers", "data"},
        )
        reply.show = show
        return reply

    def c_pairreport(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["pairreport"])
        image = self.buffer(words[0])
        record = self.pairs.get(image.key)
        if record is None:
            raise CommandError(f"no pair fit of {image.name} yet (pair {words[0]} ...)")
        text = pairs.report(record.result, record.job.snap, record.job.line, record.job.measurement)
        if len(words) > 1:
            path = self.path(words[1])
            try:
                path.write_text(text + "\n")
            except OSError as e:
                raise CommandError(f"cannot write {path}: {e}")
            text += f"\n(written to {path})"
        return Reply(text)

    def c_pairexport(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(HELP["pairexport"])
        image = self.buffer(words[0])
        record = self.pairs.get(image.key)
        if record is None:
            raise CommandError(f"no pair fit of {image.name} yet (pair {words[0]} ...)")
        folder = self.path(words[1])
        snap, result = record.job.snap, record.result
        sky_result = self.skies.get(snap.sky_id) if snap.sky_id else None
        replay = []
        if snap.provenance.get("file"):
            ltv = snap.provenance.get("window_offset_ltv") or [None, None]
            replay.append(f"rd 1 {snap.provenance['file']}")
            if snap.provenance.get("source_run"):
                replay.append(snap.provenance["source_run"]["command"].strip())
            if ltv[0]:
                replay.append(
                    f"# then window 1 BOX=... (LTV1={ltv[0]}, LTV2={ltv[1]}: the window's "
                    "offset in the original)"
                )
            mask = " ".join(
                re.findall(r"\b(?:MASK|MASKBUF|MASKFILE)=\S+", record.job.line.split(" A:")[0])
            )
            replay.append(pairs.command_for(snap, mask))
        try:
            pairs.export(
                result,
                snap,
                folder,
                record.job.line,
                sky_report=sky_result.report() if sky_result else None,
                sky_exclusion=sky_result.exclusion if sky_result else None,
                replay=replay,
            )
        except OSError as e:
            raise CommandError(f"cannot write {folder}: {e}")
        return Reply(f"wrote the pair fit ({result.status}) to {folder}")

    def c_sources(self, words, keys) -> Reply:
        """Find the sources in a buffer (the Source finder)."""
        if not words:
            raise CommandError(HELP["sources"])
        n = integer(words[0], "the buffer")
        image = self.buffer(n)
        # the panel's defaults, then the settings given (numbers stay numbers)
        settings = dict(detection.DEFAULTS)
        for k, v in keys.items():
            if k in ("TOSEG", "MASK"):
                continue
            try:
                settings[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                settings[k] = v
        number_ = max((r.number for r in self.runs), default=0) + 1
        try:
            layers = self.mask_list(keys.get("MASK"), image.shape)
            run = detection.detect(image, settings, number_, maskops.union(layers))
        except (RuntimeError, ValueError) as e:
            raise CommandError(str(e))
        self.runs.append(run)
        reply = Reply(f"run {run.number}: {len(run)} sources in {run.seconds:.2f} s", {"sources"})
        if run.warnings:
            reply.text += "".join(f"\n  {w}" for w in run.warnings)
        if "TOSEG" in keys and run.segmentation is not None:
            m = integer(keys["TOSEG"], "TOSEG")
            self.buffers[m] = image.derived(
                f"{image.name} segmentation {run.number}", run.segmentation
            )
            reply.text += f"\nsegmentation in buffer {m}"
            reply.changed |= {"buffers", "data"}
        return reply

    def c_runs(self, words, keys) -> Reply:
        if not self.runs:
            return Reply("no source-finder runs yet")
        return Reply(
            "\n".join(
                f"{r.number:3d}  {r.image_name}  {len(r)} sources  "
                + " ".join(
                    f"{k}={v}" for k, v in r.settings.items() if v != detection.DEFAULTS.get(k)
                )
                for r in self.runs
            )
        )

    def c_centre(self, words, keys) -> Reply:
        if not words or "SOURCE" not in keys:
            raise CommandError(HELP["centre"])
        image = self.buffer(words[0])
        run = self.run_for(image, keys)
        source = integer(keys["SOURCE"], "SOURCE")
        rows = np.flatnonzero(run.columns["NUMBER"] == source)
        if not rows.size:
            raise CommandError(f"run {run.number} has no source {source}")
        row = int(rows[0])
        x, y, a, _b, _t = run.ellipses()[row]
        self.centre = (round(float(x), 2), round(float(y), 2))
        number_ = int(run.columns["NUMBER"][row])
        return Reply(
            f"centre X0={self.centre[0]:g} Y0={self.centre[1]:g} "
            f"(source {number_} of run {run.number}; its Kron ellipse is {a:.1f} pixels)",
            {"centre"},
            values={"x0": self.centre[0], "y0": self.centre[1], "source": number_, "radius": a},
        )

    def c_masks(self, words, keys) -> Reply:
        if not self.masks:
            return Reply("no mask layers")
        return Reply(
            "\n".join(
                f"{m.name}  {m.shape[1]} x {m.shape[0]}  {m.count()} pixels masked"
                + (f"  [{maskops.ROLES[m.role]}]" if m.role else "")
                + ("" if m.visible else "  (hidden)")
                for m in self.masks.values()
            )
        )

    # ---- mask layers
    def _snapshot(self, name: str):
        """The state of the layer ``name`` (None when there is none), compactly."""
        m = self.masks.get(name)
        if m is None:
            return (name, None)
        return (
            name,
            (np.packbits(m.masked), m.shape, m.colour, m.visible, m.opacity, m.source, m.role),
        )

    def _restore(self, snapshot) -> None:
        name, state = snapshot
        if state is None:
            self.masks.pop(name, None)
            return
        packed, shape, colour, visible, opacity, source, role = state
        masked = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        self.masks[name] = Mask(name, masked, colour, visible, opacity, source, role)

    def _remember(self, *names) -> None:
        """Keep the layers ``names`` as they are, so that the edit about to be made can
        be undone (the last 30 edits are kept)."""
        self._undo.append(tuple(self._snapshot(n) for n in names))
        del self._undo[:-30]
        self._redo.clear()

    def c_undo(self, words, keys) -> Reply:
        if not self._undo:
            raise CommandError("nothing to undo")
        edit = self._undo.pop()
        self._redo.append(tuple(self._snapshot(name) for name, _ in edit))
        for snapshot in edit:
            self._restore(snapshot)
        reply = Reply("undone: " + ", ".join(name for name, _ in edit), {"masks"})
        for name, _ in edit:
            if name in self.mask_buffers:
                self._mask_buffer(name, reply)
        return reply

    def c_redo(self, words, keys) -> Reply:
        if not self._redo:
            raise CommandError("nothing to redo")
        edit = self._redo.pop()
        self._undo.append(tuple(self._snapshot(name) for name, _ in edit))
        for snapshot in edit:
            self._restore(snapshot)
        reply = Reply("redone: " + ", ".join(name for name, _ in edit), {"masks"})
        for name, _ in edit:
            if name in self.mask_buffers:
                self._mask_buffer(name, reply)
        return reply

    def _mask_buffer(self, name: str, reply: Reply) -> None:
        """Keep the buffer of a loaded mask up to date with the layer (1 = good, 0 =
        masked), or give the layer one."""
        mask = self.masks.get(name)
        n, shown = self.mask_buffers.get(name, (None, None))
        if mask is None:
            self.mask_buffers.pop(name, None)  # the buffer stays, as an image
            return
        if n is None or self.buffers.get(n) is not shown:
            # no buffer yet, or something else was put in it since: a buffer of its own
            n = self.free_buffer()
            reply.text += f"\nalso as buffer {n}"
        like = next((b for b in self.buffers.values() if b.shape == mask.shape), None)
        image = Image(f"mask {name}", (~mask.masked).astype(np.float32))
        if like is not None:
            image.header = like.header.copy()
        self.buffers[n] = image
        self.mask_buffers[name] = (n, image)
        reply.changed |= {"buffers", "data"}

    def _layer(self, name: str) -> Mask:
        if name not in self.masks:
            raise CommandError(f"there is no mask layer {name!r} (mask new {name})")
        return self.masks[name]

    def _same_size(self, mask: Mask, image: Image) -> None:
        if image.shape != mask.shape:
            raise CommandError(f"buffer {self.number_of(image)} is not the mask's size")

    def c_mask(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(MASK_HELP)
        action, name, rest = words[0].lower(), words[1], words[2:]
        erase = any(w.upper() == "ERASE" for w in rest)
        rest = [w for w in rest if w.upper() != "ERASE"]
        value = not erase
        changed = {"masks"}

        def numbers(count: int, usage: str, at_least=False):
            if len(rest) < count or (not at_least and len(rest) != count):
                raise CommandError(f"usage: mask {usage}")
            return [number(w, "a coordinate") for w in rest]

        def make(masked, source):
            self.masks[name] = Mask(name, masked, self._colour(name), source=source)

        # layers made from something else
        if action in ("new", "load", "frombuf", "copy", "union", "intersect"):
            self._remember(name)
            if action == "new":
                make(np.zeros(self._shape(rest[0] if rest else None), bool), "drawn")
            elif action == "load":
                if not rest:
                    raise CommandError("usage: mask load NAME file [N]")
                path = self.path(rest[0])
                if not path.exists():
                    raise CommandError(f"no file {path}")
                try:
                    make(
                        maskops.load(path, self._shape(rest[1] if len(rest) > 1 else None)),
                        str(path),
                    )
                except ValueError as e:
                    raise CommandError(str(e))
            elif action == "frombuf":
                if not rest:
                    raise CommandError("usage: mask frombuf NAME N")
                data = self.buffer(rest[0]).data
                make((data == 0) | ~np.isfinite(data), f"buffer {rest[0]}")
            elif action == "copy":
                if not rest:
                    raise CommandError("usage: mask copy NAME FROM")
                make(self._layer(rest[0]).masked.copy(), f"copy of {rest[0]}")
            else:
                layers = [self._layer(n) for n in rest]
                if len(layers) < 2:
                    raise CommandError(f"usage: mask {action} NAME A B ...")
                if any(m.shape != layers[0].shape for m in layers):
                    raise CommandError("the layers differ in size")
                combined = layers[0].masked.copy()
                for m in layers[1:]:
                    combined = combined | m.masked if action == "union" else combined & m.masked
                make(combined, f"{action} of {', '.join(rest)}")
            reply = Reply(f"mask {name}: {self.masks[name].count()} pixels masked", changed)
            if action == "load" or name in self.mask_buffers:
                self._mask_buffer(name, reply)
            return reply

        mask = self._layer(name)
        if action == "save":
            if not rest:
                raise CommandError("usage: mask save NAME file")
            return Reply(f"wrote {maskops.save(mask, self.path(rest[0]))}")
        if action == "tobuf":
            if not rest:
                raise CommandError("usage: mask tobuf NAME M")
            m = self.free_buffer() if rest[0] == "new" else integer(rest[0], "the buffer")
            header = next((b.header for b in self.buffers.values() if b.shape == mask.shape), None)
            self.buffers[m] = Image(f"mask {name}", (~mask.masked).astype(np.float32))
            if header is not None:
                self.buffers[m].header = header.copy()
            return Reply(f"buffer {m}: mask {name} (1 = good)", {"buffers", "data"}, show=m)
        if action in ("show", "hide"):
            mask.visible = action == "show"
            return Reply(changed=changed)
        if action == "role":
            if not rest or rest[0].lower() not in ("fit", "sky", "candidate", "none"):
                raise CommandError("usage: mask role NAME fit|sky|candidate|none")
            self._remember(name)
            mask.role = "" if rest[0].lower() == "none" else rest[0].lower()
            return Reply(f"mask {name}: {maskops.ROLES[mask.role]}", changed)
        if action == "rename":
            if not rest or rest[0] in self.masks:
                raise CommandError("usage: mask rename NAME NEW (a name not in use)")
            self._remember(name, rest[0])
            mask.name = rest[0]
            self.masks = {(rest[0] if k == name else k): v for k, v in self.masks.items()}
            reply = Reply(f"mask {name} is now {rest[0]}", changed)
            if name in self.mask_buffers:
                self.mask_buffers[rest[0]] = self.mask_buffers.pop(name)
                self._mask_buffer(rest[0], reply)
            return reply
        if action == "delete":
            self._remember(name)
            del self.masks[name]
            self.mask_buffers.pop(name, None)  # its buffer stays, as an image
            return Reply(f"deleted mask {name}", changed)

        self._remember(name)
        before = mask.count()
        if action == "circle":
            x, y, r = numbers(3, "circle NAME X Y R [ERASE]")
            maskops.circle(mask.masked, x, y, r, value)
        elif action == "ellipse":
            x, y, a, b, angle = numbers(5, "ellipse NAME X Y A B ANGLE [ERASE]")
            maskops.ellipse(mask.masked, x, y, a, b, angle, value)
        elif action == "box":
            maskops.box(mask.masked, *numbers(4, "box NAME X0 Y0 X1 Y1 [ERASE]"), value)
        elif action == "polygon":
            xy = numbers(6, "polygon NAME X1 Y1 X2 Y2 X3 Y3 ... [ERASE]", at_least=True)
            if len(xy) % 2:
                raise CommandError("a polygon needs pairs of coordinates")
            maskops.polygon(mask.masked, list(zip(xy[::2], xy[1::2])), value)
        elif action == "brush":
            v = numbers(3, "brush NAME R X1 Y1 X2 Y2 ... [ERASE]", at_least=True)
            if len(v) % 2 == 0:
                raise CommandError("a brush needs a radius and pairs of coordinates")
            maskops.brush(mask.masked, list(zip(v[1::2], v[2::2])), v[0], value)
        elif action == "unpatch":
            x, y = numbers(2, "unpatch NAME X Y")
            patch = maskops.patch_at(mask.masked, x, y)
            if patch is None:
                self._undo.pop()
                raise CommandError(f"{name} does not mask the pixel at {x:g}, {y:g}")
            mask.masked[patch] = False
        elif action in ("grow", "shrink"):
            (r,) = numbers(1, f"{action} NAME R")
            mask.masked = (maskops.grow if action == "grow" else maskops.shrink)(mask.masked, r)
        elif action == "invert":
            mask.masked = ~mask.masked
        elif action == "clear":
            mask.masked[:] = False
        elif action == "fill":
            mask.masked[:] = True
        elif action in ("nan", "range"):
            if not rest:
                raise CommandError(
                    f"usage: mask {action} NAME N" + (" LO HI" if action == "range" else "")
                )
            image = self.buffer(rest[0])
            self._same_size(mask, image)
            if action == "nan":
                mask.masked |= ~np.isfinite(image.data)
            else:
                if len(rest) != 3:
                    raise CommandError("usage: mask range NAME N LO HI")
                lo, hi = number(rest[1], "LO"), number(rest[2], "HI")
                with np.errstate(invalid="ignore"):
                    mask.masked |= (image.data >= lo) & (image.data <= hi)
        elif action in ("sources", "segments"):
            image = next((b for b in self.buffers.values() if b.shape == mask.shape), None)
            if image is None:
                raise CommandError("no buffer has the mask's size")
            run = self.run_for(image, keys)
            if run.segmentation is not None and run.segmentation.shape != mask.shape:
                raise CommandError(f"run {run.number} was on an image of another size")
            scale = number(keys.get("SCALE", 1.5), "SCALE")
            centres = [self.centre]
            if "X0" in keys and "Y0" in keys:
                xs = [number(t, "X0") for t in str(keys["X0"]).split(",") if t.strip()]
                ys = [number(t, "Y0") for t in str(keys["Y0"]).split(",") if t.strip()]
                if len(xs) != len(ys) or not xs:
                    raise CommandError("X0= and Y0= need the same number of values")
                centres = list(zip(xs, ys))
            centre = centres[0]
            kept = [self._kept_source(run, keys.get("KEEP", "centre"), scale, c) for c in centres]
            if run.segmentation is not None and str(keys.get("KEEP", "centre")).lower() == "centre":
                # the segment under each centre too (a pair blended into one segment keeps it)
                ny, nx = run.segmentation.shape
                for c in centres:
                    if c is None:
                        continue
                    i, j = int(np.floor(c[0])), int(np.floor(c[1]))
                    if 0 <= i < nx and 0 <= j < ny and run.segmentation[j, i] > 0:
                        rows = np.flatnonzero(run.columns["NUMBER"] == run.segmentation[j, i])
                        kept += [int(r) for r in rows]
            kept = sorted({k for k in kept if k is not None})
            chosen = np.ones(len(run), bool)
            if "IDS" in keys:
                ids = [integer(t, "IDS") for t in re.split(r"[,\s]+", keys["IDS"]) if t]
                chosen &= np.isin(run.columns["NUMBER"], ids)
            if "WHERE" in keys:
                try:
                    chosen &= maskops.where(run.columns, keys["WHERE"])
                except ValueError as e:
                    raise CommandError(str(e))
            if kept:
                chosen[kept] = False
                numbers_kept = ", ".join(str(int(run.columns["NUMBER"][k])) for k in kept)
                kept_note = f" (source{'s' if len(kept) > 1 else ''} {numbers_kept} kept unmasked"
                if centre is None and str(keys.get("KEEP", "centre")).lower() == "centre":
                    kept_note += ": the largest, as there is no ELLIPROF centre yet"
                kept_note += ")"
            else:
                kept_note = ""
            if action == "sources":
                mask.masked |= detection.source_mask(run, mask.shape, None, scale, chosen)
                mask.source = f"sources of run {run.number} x {scale:g}"
            else:
                if run.segmentation is None:
                    raise CommandError(f"run {run.number} has no segmentation")
                numbers_ = run.columns["NUMBER"][chosen]
                pixels = np.isin(run.segmentation, numbers_)
                g = number(keys.get("GROW", 0), "GROW")
                widen = number(keys.get("WIDEN", 0), "WIDEN")
                if widen < 0 or g < 0:
                    raise CommandError("GROW and WIDEN cannot be negative")
                if widen > 0:  # each segment grown by GROW + WIDEN x its equivalent radius
                    pixels = _grow_by_size(run.segmentation, numbers_, g, widen)
                else:
                    pixels = maskops.grow(pixels, g) if g > 0 else pixels
                if any(w.upper() == "ADD" for w in rest):
                    mask.masked |= pixels
                else:  # the layer becomes the segments: everything else unmasked
                    mask.masked[...] = pixels
                mask.source = f"segments of run {run.number}" + (f" grown {g:g}" if g else "")
                if widen > 0:
                    mask.source += f" + {widen:g} x their radius"
        else:
            self._undo.pop()
            raise CommandError(f"unknown mask action {action!r}\n{MASK_HELP}")
        after = mask.count()
        reply = Reply(f"mask {name}: {after} pixels masked ({after - before:+d})", changed)
        if action in ("sources", "segments"):
            reply.text += kept_note
        if name in self.mask_buffers:
            self._mask_buffer(name, reply)
        return reply

    def _shape(self, n) -> Tuple[int, int]:
        if n is not None:
            return self.buffer(n).shape
        if not self.buffers:
            raise CommandError("read an image first")
        return self.buffers[min(self.buffers)].shape

    def _colour(self, name: str) -> str:
        if name in self.masks:
            return self.masks[name].colour
        return maskops.COLOURS[len(self.masks) % len(maskops.COLOURS)]

    def _kept_source(self, run, keep: str, scale: float, centre) -> Optional[int]:
        """The row of the source not to mask: the galaxy at ELLIPROF's centre, a source
        number, or none."""
        keep = str(keep).lower()
        if keep == "none":
            return None
        if keep.isdigit():
            rows = np.flatnonzero(run.columns["NUMBER"] == int(keep))
            return int(rows[0]) if rows.size else None
        if not len(run):
            return None
        if centre is None:  # no centre yet: the largest source, nearly always the galaxy
            return int(np.argmax(run.columns["ISOAREA_IMAGE"]))
        row = run.nearest(*centre)
        x, y, a, _b, _t = run.ellipses(scale)[row]
        return row if np.hypot(x - centre[0], y - centre[1]) <= max(a, 5.0) else None

    def c_run(self, words, keys) -> Reply:
        if not words:
            raise CommandError(HELP["run"])
        path = self.path(words[0])
        if not path.exists():
            raise CommandError(f"no file {path}")
        total = Reply()
        # file names in the file are relative to the file's own folder while it runs
        folder, self.folder = self.folder, path.parent.resolve()
        try:
            lines = path.read_text().splitlines()
            joined = []  # lines ending in \\ continue on the next
            for k, line in enumerate(lines, 1):
                if joined and joined[-1][1].rstrip().endswith("\\"):
                    joined[-1] = (joined[-1][0], joined[-1][1].rstrip()[:-1] + " " + line.strip())
                else:
                    joined.append((k, line))
            for k, line in joined:
                if not line.split("#")[0].strip():
                    continue
                try:
                    reply = self.execute(line)
                except CommandError as e:
                    raise CommandError(f"{path.name}, line {k}: {e}")
                total.text += f"> {line}\n" + (reply.text + "\n" if reply.text else "")
                total.changed |= reply.changed
                total.show = reply.show if reply.show is not None else total.show
                total.values.update(reply.values)
        finally:
            if self.folder == path.parent.resolve():  # unless the file changed it (cd)
                self.folder = folder
        total.text = total.text.rstrip()
        return total


# keywords offered by Tab completion after each command
KEYWORDS = {
    "models": "STEP= SHOW= AVERAGE=",
    "elliprof": "X0= Y0= R0= R1= NR= NITER= SKY= RMSTAR MASK= MASKBUF= MASKFILE= TOMODEL= "
    "TORESID= RAW MINUS= MODE= COS3X= COS4X= FIXCTR= RLAW= GAIN= TIE= RESIDMASK= RESIDTHRESH= "
    "RESIDGROW= RESIDCORE= START= FACTOR= SKYITER= SKYPLANE= SKYFACTOR= TRYFIX= TRYFACTOR= "
    "TRYCOS3X= TRYCOS4X= HARMONICS= PARALLEL= RUNS= JITTER= SEED= TAIL= SHAPETOL=",
    "pair": "SKY= SKYFACTOR= SKYITER= STEPS= MASK= CYCLES= NITER= CORECAP= VIEWS TOMODEL= TORESID= TOMODELA= TOMODELB= "
    "A: B: X0= Y0= R0= R1= NR= NAME= MODE= FIXCTR= CORE= ADAPTIVE BLOCK= START= STEP= FACTOR= "
    "RADII= CORES= CONVERGE= CONSECUTIVE= ONFAIL= DAMP= POLISH= TRYDAMP= TRYFIX= TRYFACTOR= TAIL= "
    "TIE= RMSTAR SKYPLANE= PARALLEL= COS3X= COS4X= TRYCOS3X= TRYCOS4X= RUNS= JITTER= SEED= "
    "HARMONICS= WINGS= METHOD= PULLCHECK= PULLPX= PULLFRAC= PULLRUN= PULLR0= FIRST= JOEGROW= "
    "JOECORE= JOECOREMIN= JOECOREMAX= SHAPETOL= RESIDMASK= RESIDTHRESH= RESIDGROW= RESIDCORE=",
    "window": "BOX= OUT= MASKS=",
    "sources": "TOSEG= MASK= " + " ".join(f"{k}=" for k in detection.DEFAULTS),
    "centre": "RUN= SOURCE=",
    "stats": "MASK=",
    "sky": "MASK= SEGMENT THRESH= MINAREA= GROW= CELLS= BACK_SIZE= FWHM= PAD= TOMASK= TOMAP= TOPLANE=",
    "skyplane": "MASK= THRESH= MINAREA= GROW= BOX= PASSES= FWHM= PAD= TOMASK= TOPLANE= TOFLAT= TOSUB=",
}
KEYWORDS.update(IMAGE_KEYWORDS)
MASK_ACTIONS = (
    "new load save frombuf tobuf copy union intersect rename delete circle ellipse box "
    "polygon brush unpatch nan range sources segments grow shrink invert clear fill show hide "
    "role"
).split()
FILE_COMMANDS = {"rd": 2, "wd": 2, "run": 1, "cd": 1, "fitsinfo": 1}  # the word that is a file name


def completions(session: "Session", line: str):
    """What Tab can complete at the end of ``line``: (where the word being typed starts,
    the candidates for it)."""
    start = max(line.rfind(" "), line.rfind("\t")) + 1
    word = line[start:]
    before = line[:start].split()
    if line.startswith("!"):
        return start, _files(session, word)
    if not before:  # the command
        verbs = sorted({n[2:] for n in dir(session) if n.startswith("c_")} | set(ALIASES))
        found = [v for v in verbs if v.startswith(word)]
        if len(word) >= 2:  # programs on the PATH, after two letters
            found += [p for p in _programs() if p.startswith(word) and p not in found]
        return start, found
    verb = before[0].lower()
    if "=" in word:  # a keyword's value
        key, _, value = word.partition("=")
        offset = start + len(key) + 1
        if key.upper() == "MASK":
            names = list(session.masks)
            head, _, last = value.rpartition(",")
            return offset + (len(head) + 1 if head else 0), [n for n in names if n.startswith(last)]
        if key.upper() in (
            "MASKBUF",
            "TOMODEL",
            "TORESID",
            "TOSEG",
            "OUT",
            "TOMODELA",
            "TOMODELB",
            "TOMAP",
            "TOPLANE",
            "MINUS",
        ):
            return offset, [str(n) for n in sorted(session.buffers) if str(n).startswith(value)]
        if key.upper() == "MASKFILE":
            return offset, _files(session, value)
        return offset, []
    if verb == "mask":
        if len(before) == 1:
            return start, [a for a in MASK_ACTIONS if a.startswith(word)]
        if len(before) == 2 and before[1] not in ("new", "load", "frombuf", "union", "intersect"):
            return start, [n for n in session.masks if n.startswith(word)]
        if before[1] in ("load", "save") and len(before) == 3:
            return start, _files(session, word)
    if verb == "help" and len(before) == 1:
        return start, [k for k in HELP if k.startswith(word) and k != "="]
    if FILE_COMMANDS.get(verb) == len(before) or not hasattr(session, f"c_{verb}"):
        return start, _files(session, word)  # a file name, or a shell command's argument
    keys = [k for k in KEYWORDS.get(verb, "").split() if k.startswith(word.upper()) and word]
    if keys:
        return start, keys
    if word.isdigit() or not word:
        buffers = [str(n) for n in sorted(session.buffers) if str(n).startswith(word)]
        return start, buffers + ([] if word else KEYWORDS.get(verb, "").split())
    return start, []


def _files(session: "Session", word: str):
    """File and folder names completing ``word``, relative to the session's folder."""
    path = Path(word).expanduser()
    folder = path.parent if word and not word.endswith("/") else path
    folder = folder if folder.is_absolute() else session.folder / folder
    stem = "" if word.endswith("/") or not word else path.name
    head = word[: len(word) - len(stem)]
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith(stem) and (stem.startswith(".") or not name.startswith(".")):
            out.append(head + name + ("/" if (folder / name).is_dir() else ""))
    return out[:200]


_PROGRAMS: Optional[List[str]] = None


def _programs() -> List[str]:
    """The programs on the PATH (looked up once)."""
    global _PROGRAMS
    if _PROGRAMS is None:
        found = set()
        for d in os.environ.get("PATH", "").split(os.pathsep):
            try:
                for name in os.listdir(d):
                    if os.access(os.path.join(d, name), os.X_OK):
                        found.add(name)
            except OSError:
                continue
        _PROGRAMS = sorted(found)
    return _PROGRAMS


# colours of a listing, by the kind of file
KINDS = (
    ((".fits", ".fit", ".fts", ".fz", ".fits.gz", ".fit.gz", ".dmask"), "#98c379"),
    ((".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".pdf", ".eps", ".svg"), "#c678dd"),
    ((".txt", ".csv", ".dat", ".cat", ".reg", ".md", ".json", ".log", ".pro"), "#e5c07b"),
)
LONG_FORMAT = re.compile(r"^[-dlcbps][-rwxsStT]{9}[@+.]?\s")


def listing_html(lines, base: Path) -> str:
    """An ls listing with colours (folders blue, FITS green, pictures purple, text yellow,
    programs red) and links: a folder's goes into it, a FITS file's opens it."""
    import html as _html

    out = []
    for text in lines:
        if LONG_FORMAT.match(text):  # ls -l: the name is what follows the eighth field
            fields = text.split(None, 8)
            name = fields[-1].split(" -> ")[0] if len(fields) == 9 else ""
            head = text[: len(text) - len(fields[-1])] if name else text
        else:
            name, head = text.strip(), ""
        target = base / name if name else None
        if target is None or not (target.exists() or target.is_symlink()):
            out.append(_html.escape(text))
            continue
        rest = text[len(head) + len(name) :] if name else ""
        shown = _html.escape(name)
        lower = name.lower()
        if target.is_dir():
            piece = f"<a href='cd:{_html.escape(str(target))}' style='color:#61afef; font-weight:600; text-decoration:none'>{shown}/</a>"
        elif lower.endswith(KINDS[0][0]):
            piece = f"<a href='open:{_html.escape(str(target))}' style='color:{KINDS[0][1]}; text-decoration:none'>{shown}</a>"
        else:
            colour = next((c for ends, c in KINDS[1:] if lower.endswith(ends)), None)
            if colour is None and os.access(target, os.X_OK):
                colour = "#e06c75"
            piece = f"<span style='color:{colour}'>{shown}</span>" if colour else shown
        out.append(_html.escape(head) + piece + _html.escape(rest))
    return "<br>".join(out)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


SLOW = {
    "elliprof",
    "sources",
    "run",
    "rd",
    "pair",
    "sky",
    "skyplane",
    "pairexport",
    "pairreport",
    "window",
}  # commands shown as busy while they run


def verb_of(line: str) -> str:
    try:
        tokens = shlex.split(line, comments=True)
    except ValueError:
        return ""
    return tokens[0].lower() if tokens else ""


PAIR_HELP = """Fitting a pair of galaxies
  pair N SKY=s [shared keywords]  A: [A's keywords]  B: [B's keywords]
Shared (before A:)
  SKY=          the sky, frozen for the whole run (required; sky N SEGMENT measures it)
  SKYFACTOR=1   the sky used is SKYFACTOR x SKY (0.5 to 1.5)
  SKYITER=0     refit up to this many times, each time with the factor + m / SKY, where m
                is the clipped median of image - sky - A - B on the sky support of the
                image's sky measurement; the fit with the smallest |m| is the result
  MASK=a,b      fitting layers (role fit, or no role; sky layers are refused)
  MASKFILE= MASKBUF=   as for elliprof
  CORES=0       mask nothing around the companion: only its model is subtracted
  STEPS=0       growth steps (when a galaxy has START but no STEP, FACTOR or RADII)
  CONVERGE=0.02 CONSECUTIVE=0   stop after CONSECUTIVE cycles at the final radii in which
                A, B and A + B each changed by less than CONVERGE (0: run every cycle)
  DAMP=1        damping: subtract alpha x new + (1 - alpha) x previous companion model;
                below 1, POLISH=2 undamped cycles end the run (the result: ELLIPROF's models)
  TRYDAMP=1,0.5 TRYFIX=1,0 TRYFACTOR=1.2,1.35   fit every combination and keep the one
                whose two models describe the pair best: stable fits first, then the
                smallest residual, fixed centres unless free ones lower it by > 0.01
  ONFAIL=reduce on a failed fit keep the last valid model and: reduce (try halfway, then
                a quarter of the way, and go on growing), hold (stay at its radius), retry
                (try the radius again next cycle) or stop (end the run)
  CYCLES=4      A then B, this many times after the initialisation (2 + 2 x CYCLES calls)
  NITER=10      iterations inside each elliprof call
  CORECAP=50    the largest core (pixels) left out around the other galaxy
  RLAW= GAIN= TIE= RMSTAR TIMEOUT=300   passed to every call
  TAIL=elliprof the model beyond each galaxy's last isophote, which is subtracted from the
                other's image: elliprof (its r^1/4 extrapolation), sersic (a Sersic law
                fitted to the outer profile), exp (the outer slope), cut (a taper to zero),
                measured (the light measured in rings of the last isophote's shape)
  SKYPLANE=1    a plane fitted to the residual of the fit kept; if significant, the search
                again on the image minus its tilt (kept when the residual is lower)
  PARALLEL=1    N fits, all averaged (1 to 16): the fit kept and N - 1 repeats with each
                galaxy's R0 and R1 changed at random by 2/3 JITTER to JITTER (JITTER=0.15:
                10 to 15 %), run N at a time, each refining its own sky; the mean A, B and
                A + B models, mean residual (image - mean sky - mean A + B), the models'
                spread and each fit's own model and residual go to buffers  [SEED=]
  RUNS=         the number of fits averaged, when not PARALLEL's (RUNS=10 PARALLEL=5)
  HARMONICS=AUTO   try every COS3X and COS4X for each galaxy, A's then B's, on the fit
                kept (as TRYCOS3X=-3,-2,-1,0,1,2 TRYCOS4X=0,1,2)
  WINGS=1       mask the light beyond the masks of bright sources (star wings, spikes)
                that the residual of a first pass shows, joined to their masks
  METHOD=JOE    Joe's mode: each galaxy's centre left out while the other is fitted, the
                smaller galaxy first and not far out, then the two in turn, each R1 grown
                only as far as its ellipses are not pulled towards the other (free
                centres judge it); fast, no search
  FIRST=SMALLER|A|B  JOEGROW=1.2  JOECORE=0.2 JOECOREMIN=6 JOECOREMAX=30   Joe's mode:
                the galaxy fitted first, the R1 tried each cycle (x the last), each centre
                left out (a share of the separation, within these pixels; 0: none)
  PULLPX=0.5 PULLFRAC=0.02 PULLRUN=2 PULLR0=8   a pull (Joe's mode and the check): a
                move towards the other of more than PULLPX px or PULLFRAC of the radius,
                PULLRUN isophotes in a row, judged by a free-centre fit from R0 >= PULLR0
  SHAPETOL=0.12  Joe's mode, at the end: an outermost ellipse off the trend by more than
                this (or a model anomalous beyond R1) refitted with coarser isophotes or a
                smaller R1, the largest clean fit kept (0: off)
  RESIDMASK=n   n times: the faint sources in the residual of the fit kept (globular
                clusters, stars and galaxies on the galaxies, which the first source finding
                cannot separate) found [RESIDTHRESH=8 x the local RMS], grown [RESIDGROW=2 px]
                and masked, not within RESIDCORE=10 px of a centre; the fit kept fitted again.
                The layer residsources and a source run per round are made
  PULLCHECK=0   no pull check (by default each galaxy of a complete fit is fitted once more
                with free centres and the report says whether its ellipses move towards
                the other, and from which radius)
  STOP          stop early once A, B and A + B all change by less than 2 %
  VIEWS         also make the image - sky - B and image - sky - A buffers
  TOMODELA= TOMODELB= TOMODEL= TORESID=   the result buffers (free ones by default)
Each galaxy (after A: or B:)
  X0= Y0= R0= R1=   its centre (ELLIPROF's coordinates) and radii (semi-major, pixels)
  NR=25  NAME="..."  FIXCTR=0|1|2
  COS3X=2 COS4X=2   ELLIPROF's harmonics in the model: 0 none, 1 the median over all
                isophotes, 2 each isophote its own (the default); COS3X=-2, -1, -3 fit
                the 6th order instead (each, median, none); or MODE=none|3|4|3,4|4,6|...
  TRYCOS3X=0,2 TRYCOS4X=0,1,2   harmonics to try on the fit kept: A's combinations with
                B as it is, then B's with A's best; kept when the residual falls by > 0.01
  START=r STEP=dr | FACTOR=f | RADII=r1,r2,...   its radius schedule: the first R1,
                then + dr or x f each cycle, or the listed radii; always ending at R1
  CORE=r        the core centred on THIS galaxy, left out while the other is fitted
  BLOCK=r       left out around THIS galaxy during the other's first fit (block the
                smaller galaxy to fit the larger one alone first)
  START=r       the R1 of its first fit; STEPS= (shared) grows it to R1 in steps
  ADAPTIVE      if its outer isophotes fail, try a smaller R1 (at most 3 times)
  R1=AUTO       where its light, on the side away from the other, falls below the
                noise sigma; measured again on the image minus the other's model of a
                first pass before the fit
  START=AUTO    a third of R1, at most 0.35 of the separation, at least 9 px; grows x1.5
  NR=AUTO       isophotes about 10 % apart from R0 to each R1 (8 to 40)
A line can go on after a \\ at its end.  The result is one of stable_valid_pair,
valid_pair_not_stable, retained_after_failed_update, incomplete_initialization, cancelled."""


def _uncover_centres(masked: np.ndarray, own) -> List[str]:
    """A fitting mask over a galaxy's own centre (a mask made to fit the other galaxy alone
    hides this one): the masked patch that holds the centre is left out of the mask, in
    place, and a note says so."""
    from scipy import ndimage

    notes = []
    ny, nx = masked.shape
    for cid in ("A", "B"):
        try:
            x, y = float(own[cid]["X0"]), float(own[cid]["Y0"])
        except (KeyError, ValueError):
            continue
        i, j = int(np.floor(x)), int(np.floor(y))
        if not (0 <= i < nx and 0 <= j < ny) or not masked[j, i]:
            continue
        labels, _n = ndimage.label(masked)
        patch = labels == labels[j, i]
        masked[patch] = False
        notes.append(
            f"the fitting mask covered {cid}'s centre: its masked patch there "
            f"({int(patch.sum())} px) is not used"
        )
    return notes


def _grow_by_size(segmentation, numbers, grow: float, widen: float, largest: float = 60.0):
    """The segments ``numbers``, each grown by grow + widen x sqrt(its area / pi) pixels (at
    most ``largest``), grouped by whole radii."""
    areas = np.bincount(segmentation.ravel())
    out = np.zeros(segmentation.shape, bool)
    numbers = np.asarray(numbers, int)
    numbers = numbers[(numbers > 0) & (numbers < areas.size)]
    lut = np.full(areas.size, -1.0)  # each segment's radius (-1: not masked)
    lut[numbers] = np.round(np.minimum(grow + widen * np.sqrt(areas[numbers] / np.pi), largest))
    per_pixel = lut[segmentation]
    for r in np.unique(lut[numbers]):
        pixels = per_pixel == r
        out |= maskops.grow(pixels, float(r)) if r > 0 else pixels
    return out


def _mask_sum(mask: Mask) -> str:
    return hashlib.sha1(np.packbits(mask.masked).tobytes()).hexdigest()[:12]


# elliprof's keywords that make it an iterative fit (single.py), as the pair fit's
ITERATIVE = {
    "SKYITER",
    "SKYPLANE",
    "SKYFACTOR",
    "TRYFIX",
    "TRYCOS3X",
    "TRYCOS4X",
    "HARMONICS",
    "PARALLEL",
    "RUNS",
    "JITTER",
    "SEED",
    "TAIL",
    "SHAPETOL",
}


class PairJob:
    """A pair command read and fixed, ready to run (see pair.py)."""

    def __init__(self, number: int, image: Image, snap, outputs: Dict[str, int], line: str):
        self.number = number
        self.image = image
        self.snap = snap
        self.outputs = outputs
        self.line = line
        self.measurement = None  # the image's current sky measurement (its support)


class PairRecord:
    """A pair fit kept in the session: the job and its result."""

    def __init__(self, job: PairJob, result):
        self.job = job
        self.result = result
