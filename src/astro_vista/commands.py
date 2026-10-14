"""The command line.  Every action of the window is a command, and commands can be typed.

    verb [buffer] [words] [KEY=value ...]     # a comment
    [3] = [1] - [2]                           arithmetic on buffers

Images live in numbered buffers.  Coordinates (X0, Y0, mask shapes) are ELLIPROF's: the
first pixel spans 0 to 1, so its centre is at 0.5.  ``help`` lists the commands.
"""

from __future__ import annotations

import hashlib
import os
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
    "elliprof": "elliprof N X0= Y0= R0= R1= NR= [NITER=10] [SKY=] [RMSTAR] [MASKFILE=file] "
    "[MASK=a,b] [MASKBUF=M,...] [TOMODEL=M] [TORESID=M] [RAW] [MINUS=M,...] [MODE=3,4] "
    "[COS3X=] [COS4X=] [FIXCTR=0|1|2] [RLAW=] [GAIN=] [TIE=]  -  fit elliptical isophotes to "
    "buffer N; MASK= names mask layers, MASKBUF= buffers used as masks (0 or NaN masked); "
    "MINUS= subtracts buffers first; RAW keeps every pixel of the residual",
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
  mask segments NAME [RUN=k] [GROW=0] [KEEP=...] [X0= Y0=]
                                  the sources' pixels (their segmentation)
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
        self._pair_runs = 0
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
        for key in ("X0", "Y0", "R0", "R1", "NR"):
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
            r1=number(keys["R1"], "R1"),
            nr=integer(keys["NR"], "NR"),
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
        try:
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
            reply.values = {"sky": result.level, "sigma": result.sigma}
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
            components = pairs.components_from(own)
            settings = pairs.settings_from(shared)
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
        job = PairJob(n, image, snap, outputs, "pair " + text.strip())
        job.measurement = candidate
        return job

    def pair_commit(self, job: "PairJob", result: pairs.PairResult) -> Reply:
        """Put a finished pair run into the session (in the window's thread)."""
        image, snap, n = job.image, job.snap, job.number
        if job.number not in self.buffers or self.buffers[n] is not image:
            pass  # the buffer was replaced meanwhile: the result keeps its own snapshot
        if result.sky_iterations:  # the fit chosen: its snapshot has its sky factor
            job.snap = snap = pairs.with_sky(snap, pairs.chosen_factor(result, snap.settings))
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
            reply.text += "\nbuffers: " + "; ".join(made)
            reply.changed |= {"buffers", "data"}
        return reply

    def _pair_line(self, text: str) -> Reply:
        """pair, run to the end in this thread (scripts, the terminal without a window)."""
        if not text.strip():
            raise CommandError(PAIR_HELP)
        job = self.pair_prepare(text)
        support = job.measurement.support if job.measurement is not None else None
        result = pairs.run_sky(job.snap, support)
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
            line = re.sub(r"^pair\s+\S+", "pair 1", record.job.line)
            if result.sky_iterations:  # the factor chosen, without iterating again
                line = re.sub(r"\s(SKYFACTOR|SKYITER)=\S+", "", line, flags=re.I)
                line = re.sub(
                    r"(\sSKY=\S+)", rf"\1 SKYFACTOR={snap.settings.sky_factor:.10g}", line, count=1
                )
            replay.append(line)
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
            centre = self.centre
            if "X0" in keys and "Y0" in keys:
                centre = (number(keys["X0"], "X0"), number(keys["Y0"], "Y0"))
            keep = self._kept_source(run, keys.get("KEEP", "centre"), scale, centre)
            chosen = np.ones(len(run), bool)
            if "IDS" in keys:
                ids = [integer(t, "IDS") for t in re.split(r"[,\s]+", keys["IDS"]) if t]
                chosen &= np.isin(run.columns["NUMBER"], ids)
            if "WHERE" in keys:
                try:
                    chosen &= maskops.where(run.columns, keys["WHERE"])
                except ValueError as e:
                    raise CommandError(str(e))
            if keep is not None:
                chosen[keep] = False
                kept_note = f" (source {int(run.columns['NUMBER'][keep])} kept unmasked"
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
                mask.masked |= maskops.grow(pixels, g) if g > 0 else pixels
                mask.source = f"segments of run {run.number}" + (f" grown {g:g}" if g else "")
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
    "elliprof": "X0= Y0= R0= R1= NR= NITER= SKY= RMSTAR MASK= MASKBUF= MASKFILE= TOMODEL= "
    "TORESID= RAW MINUS= MODE= COS3X= COS4X= FIXCTR= RLAW= GAIN= TIE=",
    "pair": "SKY= SKYFACTOR= SKYITER= STEPS= MASK= CYCLES= NITER= CORECAP= VIEWS TOMODEL= TORESID= TOMODELA= TOMODELB= "
    "A: B: X0= Y0= R0= R1= NR= NAME= MODE= FIXCTR= CORE= ADAPTIVE BLOCK= START= STEP= FACTOR= "
    "RADII= CORES= CONVERGE= CONSECUTIVE= ONFAIL=",
    "window": "BOX= OUT= MASKS=",
    "sources": "TOSEG= MASK= " + " ".join(f"{k}=" for k in detection.DEFAULTS),
    "centre": "RUN= SOURCE=",
    "stats": "MASK=",
    "sky": "MASK= SEGMENT THRESH= MINAREA= GROW= CELLS= BACK_SIZE= FWHM= PAD= TOMASK= TOMAP= TOPLANE=",
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
  ONFAIL=hold   on a failed fit keep the last valid model and: hold (stay at its radius),
                retry (try the radius again next cycle) or stop (end the run)
  CYCLES=4      A then B, this many times after the initialisation (2 + 2 x CYCLES calls)
  NITER=10      iterations inside each elliprof call
  CORECAP=50    the largest core (pixels) left out around the other galaxy
  RLAW= GAIN= TIE= RMSTAR TIMEOUT=300   passed to every call
  STOP          stop early once A, B and A + B all change by less than 2 %
  VIEWS         also make the image - sky - B and image - sky - A buffers
  TOMODELA= TOMODELB= TOMODEL= TORESID=   the result buffers (free ones by default)
Each galaxy (after A: or B:)
  X0= Y0= R0= R1=   its centre (ELLIPROF's coordinates) and radii (semi-major, pixels)
  NR=25  NAME="..."  MODE=none|3|4|3,4|4,6|3m|4m|6m,4m (or COS3X= COS4X=)  FIXCTR=0|1|2
  START=r STEP=dr | FACTOR=f | RADII=r1,r2,...   its radius schedule: the first R1,
                then + dr or x f each cycle, or the listed radii; always ending at R1
  CORE=r        the core centred on THIS galaxy, left out while the other is fitted
  BLOCK=r       left out around THIS galaxy during the other's first fit (block the
                smaller galaxy to fit the larger one alone first)
  START=r       the R1 of its first fit; STEPS= (shared) grows it to R1 in steps
  ADAPTIVE      if its outer isophotes fail, try a smaller R1 (at most 3 times)
A line can go on after a \\ at its end.  The result is one of stable_valid_pair,
valid_pair_not_stable, retained_after_failed_update, incomplete_initialization, cancelled."""


def _mask_sum(mask: Mask) -> str:
    return hashlib.sha1(np.packbits(mask.masked).tobytes()).hexdigest()[:12]


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
