"""Image commands in the style of the VISTA and MONSTA command languages: the names and
the way of writing them (``ac 1 100``, ``shift 2 DX=1.5``, ``abx 1``) are those
astronomers know, and every operation is written here, from its usual definition, with
NumPy and SciPy.

Pixel positions use columns and rows counted from 0 (column ``c`` is the pixel between
x = c and c + 1).  Results that make a new image go to OUT= (a buffer number);
otherwise the buffer itself changes.
"""

from __future__ import annotations

import ast
import math
from typing import Dict, Tuple

import numpy as np

from .cmdbase import CommandError, Reply, integer
from .images import Image

IMAGE_HELP = {
    "ac": "ac N value  -  add a constant to buffer N (sc subtracts, mc multiplies, dc divides)",
    "ai": "ai N M  -  add buffer M to buffer N (si subtracts, mi multiplies, di divides)",
    "log": "log N  -  base-10 logarithm of buffer N (ln: natural; sqrt; inverse: 1/x; "
    "round: to whole numbers); pixels where it is undefined become NaN",
    "mn": "mn N  -  the mean, median, standard deviation and number of pixels",
    "max": "max N  -  the largest pixel and where it is (min: the smallest)",
    "abx": "abx N  -  sum, mean, minimum and maximum (with their positions)",
    "print": "print N C R [SIZE=5]  -  the pixel values around column C, row R",
    "shift": "shift N DX= DY= [ORDER=3]  -  move the image by DX, DY pixels (fractions too)",
    "rotate": "rotate N ANGLE= [ORDER=3]  -  rotate about the centre, degrees counter-"
    "clockwise (multiples of 90 are exact)",
    "flip": "flip N X|Y  -  mirror left-right (X) or top-bottom (Y)",
    "bin": "bin N k [MEAN]  -  add up (or average) k x k blocks of pixels",
    "mash": "mash N ROWS|COLS [OUT=M]  -  add up the rows (or columns) into one",
    "smooth": "smooth N FWHM=  or  smooth N BOXCAR=k  -  smooth with a Gaussian or a boxcar",
    "filt": "filt N MEDIAN=k  -  replace each pixel by the median of the k x k around it",
    "zap": "zap N [SIG=5] [SIZE=5]  -  replace pixels more than SIG sigma above the local "
    "median (cosmic rays, hot pixels) by that median",
    "clip": "clip N [MIN=] [MAX=] [VALUE=]  -  pixels below MIN or above MAX become the "
    "limit (or VALUE, or NaN with VALUE=nan)",
    "skysub": "skysub N [VALUE=]  -  subtract the sky (the value given, or a clipped median)",
    "fft": "fft N OUT=M  -  the power spectrum of buffer N (centred) into buffer M",
    "noise": "noise N SIGMA= [SEED=]  -  add Gaussian noise",
    "gauss": "gauss N X= Y= FWHM= PEAK=  -  add a Gaussian star at X, Y",
    "ch": "ch N KEY=value ...  -  set FITS header keywords of buffer N",
    "chead": "chead N [WORD]  -  the header of buffer N (the cards containing WORD)",
    "set": "set NAME=expression ...  -  variables, usable in any keyword (X0=xc)",
    "typ": "typ [NAME ...]  -  print variables (all of them without names)",
    "buf": "buf  -  list the buffers (as buffers)",
}
KEYWORDS = {
    "shift": "DX= DY= ORDER=",
    "rotate": "ANGLE= ORDER=",
    "smooth": "FWHM= BOXCAR=",
    "filt": "MEDIAN=",
    "zap": "SIG= SIZE=",
    "clip": "MIN= MAX= VALUE=",
    "skysub": "VALUE=",
    "fft": "OUT=",
    "noise": "SIGMA= SEED=",
    "gauss": "X= Y= FWHM= PEAK=",
    "mash": "OUT=",
    "print": "SIZE=",
}
MATH = {
    "log": np.log10,
    "ln": np.log,
    "sqrt": np.sqrt,
    "inverse": lambda a: np.where(a == 0, np.nan, 1.0 / np.where(a == 0, 1.0, a)),
    "round": np.round,
}


def evaluate(text: str, variables: Dict[str, float]) -> float:
    """A number from an expression of numbers, variables, + - * / ** and a few functions."""
    functions = {
        "sqrt": math.sqrt,
        "log10": math.log10,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id.lower() in variables:  # variable names in any case
                return float(variables[node.id.lower()])
            if node.id == "pi":
                return math.pi
            raise CommandError(f"no variable {node.id!r} (set {node.id}=...)")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            ops = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}
            if type(node.op) in ops:
                return ops[type(node.op)]
            if isinstance(node.op, ast.Div):
                return a / b if b else float("nan")
            if isinstance(node.op, ast.Pow):
                return a**b
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and len(node.args) == 1
        ):
            return float(functions[node.func.id](walk(node.args[0])))
        raise CommandError(f"cannot work out {text!r}")

    try:
        return walk(ast.parse(text.strip(), mode="eval").body)
    except SyntaxError:
        raise CommandError(f"cannot read {text!r}")


class ImageCommands:
    """The image commands, mixed into the Session (which has buffers and variables)."""

    # ---- helpers
    def _image(self, words) -> Tuple[int, Image]:
        if not words:
            raise CommandError("which buffer?")
        n = integer(words[0], "the buffer")
        return n, self.buffer(n)

    def _put(self, n: int, image: Image, data, name: str, keys=None) -> Reply:
        """The result: into buffer OUT= when given, else in place of buffer n."""
        target = integer(keys["OUT"], "OUT") if keys and "OUT" in keys else n
        new = Image(
            name if target != n else image.name,
            np.asarray(data, np.float32),
            image.header.copy(),
            image.path if target == n else None,
        )
        self.buffers[target] = new
        return Reply(f"buffer {target}: {name}", {"buffers", "data"}, show=target)

    @staticmethod
    def _region(keys, image):
        """The pixels a statistic looks at: the whole image."""
        return image.data, 0, 0

    def value_of(self, text: str) -> float:
        """A keyword's value: a number, a variable or an expression of them."""
        try:
            return float(text)
        except (TypeError, ValueError):
            return evaluate(str(text), self.variables)

    # ---- arithmetic
    def _constant(self, words, op, name) -> Reply:
        if len(words) < 2:
            raise CommandError(IMAGE_HELP["ac"])
        n, image = self._image(words)
        c = self.value_of(words[1])
        with np.errstate(all="ignore"):
            data = op(image.data.astype(np.float64), c)
        image.data = data.astype(np.float32)
        return Reply(f"buffer {n}: {name} {c:g}", {"data"}, show=n)

    def c_ac(self, words, keys):
        return self._constant(words, np.add, "+")

    def c_sc(self, words, keys):
        return self._constant(words, np.subtract, "-")

    def c_mc(self, words, keys):
        return self._constant(words, np.multiply, "×")

    def c_dc(self, words, keys):
        if len(words) > 1 and self.value_of(words[1]) == 0:
            raise CommandError("dividing by 0")
        return self._constant(words, np.divide, "÷")

    def _images(self, words, op, name) -> Reply:
        if len(words) < 2:
            raise CommandError(IMAGE_HELP["ai"])
        n, a = self._image(words)
        m, b = self._image(words[1:])
        if a.shape != b.shape:
            raise CommandError(f"buffers {n} and {m} differ in size")
        with np.errstate(all="ignore"):
            if op is np.divide:  # NaN where the divisor is 0, never infinite
                from .calc import safe_divide

                out = safe_divide(a.data, b.data)
            else:
                out = op(a.data.astype(np.float64), b.data)
        a.data = np.where(np.isinf(out), np.nan, out).astype(np.float32)
        return Reply(f"buffer {n}: {name} buffer {m}", {"data"}, show=n)

    def c_ai(self, words, keys):
        return self._images(words, np.add, "+")

    def c_si(self, words, keys):
        return self._images(words, np.subtract, "-")

    def c_mi(self, words, keys):
        return self._images(words, np.multiply, "×")

    def c_di(self, words, keys):
        return self._images(words, np.divide, "÷")

    def _math(self, words, name) -> Reply:
        n, image = self._image(words)
        with np.errstate(all="ignore"):
            out = MATH[name](image.data.astype(np.float64))
        image.data = np.where(np.isinf(out), np.nan, out).astype(np.float32)  # log(0): NaN
        return Reply(f"buffer {n}: {name}", {"data"}, show=n)

    def c_log(self, words, keys):
        return self._math(words, "log")

    def c_ln(self, words, keys):
        return self._math(words, "ln")

    def c_sqrt(self, words, keys):
        return self._math(words, "sqrt")

    def c_inverse(self, words, keys):
        return self._math(words, "inverse")

    def c_round(self, words, keys):
        return self._math(words, "round")

    # ---- statistics
    def c_mn(self, words, keys) -> Reply:
        n, image = self._image(words)
        region, _c, _r = self._region(keys, image)
        v = region[np.isfinite(region)].astype(np.float64)
        if not v.size:
            raise CommandError("no finite pixels")
        values = {
            "mean": v.mean(),
            "median": float(np.median(v)),
            "sigma": v.std(),
            "pixels": v.size,
        }
        self.variables.update(
            {"mean": values["mean"], "median": values["median"], "sigma": values["sigma"]}
        )
        text = "  ".join(f"{k} {val:.6g}" for k, val in values.items())
        return Reply(
            f"buffer {n}: {text}   (also the variables mean, median, sigma)", values=values
        )

    def _extreme(self, words, keys, largest: bool) -> Reply:
        n, image = self._image(words)
        region, c0, r0 = self._region(keys, image)
        if not np.isfinite(region).any():
            raise CommandError("no finite pixels")
        k = np.nanargmax(region) if largest else np.nanargmin(region)
        r, c = np.unravel_index(k, region.shape)
        value = float(region[r, c])
        name = "max" if largest else "min"
        self.variables.update({name: value, f"{name}c": c + c0, f"{name}r": r + r0})
        return Reply(
            f"buffer {n}: {name} {value:.6g} at column {c + c0}, row {r + r0}",
            values={name: value},
        )

    def c_max(self, words, keys):
        return self._extreme(words, keys, True)

    def c_min(self, words, keys):
        return self._extreme(words, keys, False)

    def c_abx(self, words, keys) -> Reply:
        n, image = self._image(words)
        region, c0, r0 = self._region(keys, image)
        finite = np.isfinite(region)
        if not finite.any():
            raise CommandError("no finite pixels")
        v = region[finite].astype(np.float64)
        hi = np.unravel_index(np.nanargmax(region), region.shape)
        lo = np.unravel_index(np.nanargmin(region), region.shape)
        return Reply(
            f"buffer {n}: sum {v.sum():.6g}  mean {v.mean():.6g}  pixels {v.size}\n"
            f"  max {region[hi]:.6g} at column {hi[1] + c0}, row {hi[0] + r0}\n"
            f"  min {region[lo]:.6g} at column {lo[1] + c0}, row {lo[0] + r0}",
            values={"sum": float(v.sum()), "mean": float(v.mean())},
        )

    def c_print(self, words, keys) -> Reply:
        if len(words) < 3:
            raise CommandError(IMAGE_HELP["print"])
        n, image = self._image(words)
        c, r = integer(self.value_of(words[1]), "C"), integer(self.value_of(words[2]), "R")
        h = integer(keys.get("SIZE", 5), "SIZE") // 2
        rows = range(min(r + h, image.shape[0] - 1), max(r - h, 0) - 1, -1)
        cols = range(max(c - h, 0), min(c + h, image.shape[1] - 1) + 1)
        lines = ["row\\col " + "".join(f"{k:>11d}" for k in cols)]
        for j in rows:
            lines.append(f"{j:>7d} " + "".join(f"{image.data[j, k]:>11.5g}" for k in cols))
        return Reply("\n".join(lines))

    # ---- geometry
    def c_shift(self, words, keys) -> Reply:
        from scipy import ndimage

        n, image = self._image(words)
        dx, dy = self.value_of(keys.get("DX", 0)), self.value_of(keys.get("DY", 0))
        order = integer(keys.get("ORDER", 3), "ORDER")
        data = np.nan_to_num(image.data.astype(np.float64))
        out = ndimage.shift(data, (dy, dx), order=order, mode="constant", cval=np.nan)
        return self._put(n, image, out, f"{image.name} shifted {dx:g}, {dy:g}", keys)

    def c_rotate(self, words, keys) -> Reply:
        from scipy import ndimage

        n, image = self._image(words)
        angle = self.value_of(keys.get("ANGLE", words[1] if len(words) > 1 else 0))
        if angle % 90 == 0:  # exact
            out = np.rot90(image.data, int(angle // 90) % 4)
        else:
            order = integer(keys.get("ORDER", 3), "ORDER")
            data = np.nan_to_num(image.data.astype(np.float64))
            out = ndimage.rotate(data, angle, reshape=False, order=order, cval=np.nan)
        return self._put(n, image, out, f"{image.name} rotated {angle:g}°", keys)

    def c_flip(self, words, keys) -> Reply:
        n, image = self._image(words)
        axis = words[1].upper() if len(words) > 1 else "X"
        if axis not in ("X", "Y"):
            raise CommandError(IMAGE_HELP["flip"])
        out = image.data[:, ::-1] if axis == "X" else image.data[::-1, :]
        return self._put(n, image, out.copy(), f"{image.name} flipped {axis}", keys)

    def c_bin(self, words, keys) -> Reply:
        if len(words) < 2:
            raise CommandError(IMAGE_HELP["bin"])
        n, image = self._image(words)
        k = integer(self.value_of(words[1]), "the block size")
        if k < 1:
            raise CommandError("the block size must be at least 1")
        ny, nx = (image.shape[0] // k) * k, (image.shape[1] // k) * k
        blocks = image.data[:ny, :nx].reshape(ny // k, k, nx // k, k).astype(np.float64)
        mean = any(w.upper() == "MEAN" for w in words[2:])
        out = np.nanmean(blocks, axis=(1, 3)) if mean else np.nansum(blocks, axis=(1, 3))
        return self._put(n, image, out, f"{image.name} binned {k}", keys)

    def c_mash(self, words, keys) -> Reply:
        n, image = self._image(words)
        how = words[1].upper() if len(words) > 1 else "ROWS"
        if how not in ("ROWS", "COLS"):
            raise CommandError(IMAGE_HELP["mash"])
        region, _c, _r = self._region(keys, image)
        out = (
            np.nansum(region, axis=0)[None, :]
            if how == "ROWS"
            else np.nansum(region, axis=1)[:, None]
        )
        if "OUT" not in keys:
            keys = dict(keys, OUT=str(self.free_buffer(n)))
        return self._put(n, image, out, f"{image.name} mashed {how.lower()}", keys)

    # ---- smoothing and cleaning
    def c_smooth(self, words, keys) -> Reply:
        from scipy import ndimage

        n, image = self._image(words)
        data = image.data.astype(np.float64)
        finite = np.isfinite(data)
        values = np.where(finite, data, 0.0)
        if "BOXCAR" in keys:
            k = integer(self.value_of(keys["BOXCAR"]), "BOXCAR")
            total = ndimage.uniform_filter(values, k)
            weight = ndimage.uniform_filter(finite.astype(np.float64), k)
            label = f"boxcar {k}"
        else:
            fwhm = self.value_of(keys.get("FWHM", 2.0))
            sigma = fwhm / 2.3548
            total = ndimage.gaussian_filter(values, sigma)
            weight = ndimage.gaussian_filter(finite.astype(np.float64), sigma)
            label = f"Gaussian FWHM {fwhm:g}"
        with np.errstate(all="ignore"):
            out = np.where(weight > 0, total / weight, np.nan)
        out[~finite] = np.nan  # pixels without data stay without
        return self._put(n, image, out, f"{image.name} smoothed ({label})", keys)

    def c_filt(self, words, keys) -> Reply:
        from scipy import ndimage

        n, image = self._image(words)
        k = integer(self.value_of(keys.get("MEDIAN", 5)), "MEDIAN")
        out = (
            ndimage.generic_filter(image.data.astype(np.float64), np.nanmedian, size=k)
            if np.isnan(image.data).any()
            else ndimage.median_filter(image.data, size=k)
        )
        return self._put(n, image, out, f"{image.name} median {k}", keys)

    def c_zap(self, words, keys) -> Reply:
        from scipy import ndimage

        n, image = self._image(words)
        sig = self.value_of(keys.get("SIG", 5.0))
        k = integer(self.value_of(keys.get("SIZE", 5)), "SIZE")
        data = image.data.astype(np.float64)
        filled = np.where(np.isfinite(data), data, np.nanmedian(data))
        median = ndimage.median_filter(filled, size=k)
        mad = ndimage.median_filter(np.abs(filled - median), size=k) * 1.4826
        floor = np.nanmedian(mad[mad > 0]) if (mad > 0).any() else 1.0
        bad = (data - median) > sig * np.maximum(mad, floor)
        out = np.where(bad, median, data)
        reply = self._put(n, image, out, f"{image.name} zapped", keys)
        reply.text += f" ({int(bad.sum())} pixels replaced)"
        return reply

    def c_clip(self, words, keys) -> Reply:
        n, image = self._image(words)
        data = image.data.astype(np.float64)
        value = keys.get("VALUE")
        fill = (
            float("nan")
            if str(value).lower() == "nan"
            else (self.value_of(value) if value is not None else None)
        )
        count = 0
        with np.errstate(invalid="ignore"):
            if "MIN" in keys:
                lo = self.value_of(keys["MIN"])
                low = data < lo
                count += int(low.sum())
                data = np.where(low, lo if fill is None else fill, data)
            if "MAX" in keys:
                hi = self.value_of(keys["MAX"])
                high = data > hi
                count += int(high.sum())
                data = np.where(high, hi if fill is None else fill, data)
        reply = self._put(n, image, data, f"{image.name} clipped", keys)
        reply.text += f" ({count} pixels)"
        return reply

    def c_skysub(self, words, keys) -> Reply:
        n, image = self._image(words)
        if "VALUE" in keys:
            sky = self.value_of(keys["VALUE"])
        else:
            from astropy.stats import sigma_clipped_stats

            _mean, sky, _sigma = sigma_clipped_stats(image.data[np.isfinite(image.data)], sigma=3.0)
        image.data = (image.data.astype(np.float64) - sky).astype(np.float32)
        return Reply(
            f"buffer {n}: sky {sky:.6g} subtracted", {"data"}, show=n, values={"sky": float(sky)}
        )

    def c_fft(self, words, keys) -> Reply:
        n, image = self._image(words)
        if "OUT" not in keys:
            raise CommandError(IMAGE_HELP["fft"])
        data = np.nan_to_num(image.data.astype(np.float64) - np.nanmean(image.data))
        power = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
        return self._put(n, image, power, f"{image.name} power spectrum", keys)

    def c_noise(self, words, keys) -> Reply:
        n, image = self._image(words)
        sigma = self.value_of(keys.get("SIGMA", 1.0))
        rng = np.random.default_rng(int(self.value_of(keys["SEED"])) if "SEED" in keys else None)
        image.data = (image.data + rng.normal(0.0, sigma, image.shape)).astype(np.float32)
        return Reply(f"buffer {n}: Gaussian noise of sigma {sigma:g} added", {"data"}, show=n)

    def c_gauss(self, words, keys) -> Reply:
        n, image = self._image(words)
        for key in ("X", "Y", "FWHM", "PEAK"):
            if key not in keys:
                raise CommandError(IMAGE_HELP["gauss"])
        x, y = self.value_of(keys["X"]), self.value_of(keys["Y"])
        sigma = self.value_of(keys["FWHM"]) / 2.3548
        peak = self.value_of(keys["PEAK"])
        jj, ii = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        star = peak * np.exp(-((ii + 0.5 - x) ** 2 + (jj + 0.5 - y) ** 2) / (2 * sigma**2))
        image.data = (image.data + star).astype(np.float32)
        return Reply(f"buffer {n}: a Gaussian star added at {x:g}, {y:g}", {"data"}, show=n)

    # ---- the header and the variables
    def c_ch(self, words, keys) -> Reply:
        n, image = self._image(words)
        if not keys:
            raise CommandError(IMAGE_HELP["ch"])
        for key, value in keys.items():
            try:
                image.header[key] = (
                    float(value) if "." in value or "e" in value.lower() else int(value)
                )
            except ValueError:
                image.header[key] = value
        return Reply(f"buffer {n}: " + " ".join(f"{k}={v}" for k, v in keys.items()))

    def c_chead(self, words, keys) -> Reply:
        n, image = self._image(words)
        cards = image.header.tostring(sep="\n").rstrip().splitlines()
        if len(words) > 1:
            cards = [c for c in cards if words[1].upper() in c.upper()]
        return Reply("\n".join(c.rstrip() for c in cards) or "(no such card)")

    def c_set(self, words, keys) -> Reply:
        if not keys:
            raise CommandError(IMAGE_HELP["set"])
        for key, text in keys.items():
            self.variables[key.lower()] = self.value_of(text)
        return Reply("  ".join(f"{k.lower()} = {self.variables[k.lower()]:.6g}" for k in keys))

    def c_typ(self, words, keys) -> Reply:
        names = [w.lower() for w in words] or sorted(self.variables)
        if not names:
            return Reply("no variables")
        missing = [n for n in names if n not in self.variables]
        if missing:
            raise CommandError(f"no variable {missing[0]!r}")
        return Reply("  ".join(f"{n} = {self.variables[n]:.6g}" for n in names))

    def c_buf(self, words, keys):
        return self.c_buffers(words, keys)
