"""Synthetic images with a known answer: a pair of overlapping elliptical galaxies (de
Vaucouleurs' law) with stars, a sky with a small gradient and Gaussian noise.  They make the
example of the pair fit and the tests of the pair fit and of the sky tool.

Positions are ELLIPROF's (the first pixel's centre is at 0.5); angles are degrees from +x,
counter-clockwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

B4 = 7.669  # de Vaucouleurs' law: I(r) = Ie exp(-B4 ((r / re)^(1/4) - 1))


@dataclass
class Galaxy:
    x: float
    y: float
    ie: float  # intensity at the effective radius
    re: float  # effective radius (semi-major), pixels
    ellip: float  # 1 - b/a
    angle: float  # major axis, degrees from +x

    def intensity(self, rmaj):
        rmaj = np.maximum(np.asarray(rmaj, float), 1e-3)
        return self.ie * np.exp(-B4 * ((rmaj / self.re) ** 0.25 - 1.0))

    def image(self, shape, oversample: int = 3, core: float = 6.0) -> np.ndarray:
        """The galaxy on a grid of pixels: each pixel's mean, by sampling it ``oversample``
        x ``oversample`` times within ``core`` re/10 of the centre, once elsewhere."""
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        out = self._at(xx + 0.5, yy + 0.5)
        r = int(np.ceil(core)) + 2
        i0, i1 = max(int(self.x) - r, 0), min(int(self.x) + r + 1, nx)
        j0, j1 = max(int(self.y) - r, 0), min(int(self.y) + r + 1, ny)
        if i0 < i1 and j0 < j1:
            k = oversample
            offsets = (np.arange(k) + 0.5) / k
            acc = np.zeros((j1 - j0, i1 - i0))
            jj, ii = np.mgrid[j0:j1, i0:i1]
            for oy in offsets:
                for ox in offsets:
                    acc += self._at(ii + ox, jj + oy)
            out[j0:j1, i0:i1] = acc / (k * k)
        return out

    def _at(self, x, y):
        t = np.radians(self.angle)
        dx, dy = x - self.x, y - self.y
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        q = 1.0 - self.ellip
        return self.intensity(np.sqrt(u * u + (v / q) ** 2))


@dataclass
class Scene:
    data: np.ndarray  # float32
    sky: float
    gradient: Tuple[float, float]  # counts per pixel along x and y (about the centre)
    noise: float
    galaxies: Dict[str, Galaxy]
    stars: List[Tuple[float, float, float]] = field(default_factory=list)  # x, y, peak
    star_mask: Optional[np.ndarray] = None  # True around the stars (a fitting mask)


def sky_plane(shape, sky: float, gradient: Tuple[float, float]) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return sky + gradient[0] * (xx + 0.5 - nx / 2) + gradient[1] * (yy + 0.5 - ny / 2)


def stars(shape, count: int, rng, peak=(80.0, 2000.0), fwhm: float = 2.5, margin: int = 8):
    ny, nx = shape
    out = []
    for _ in range(count):
        out.append(
            (
                rng.uniform(margin, nx - margin),
                rng.uniform(margin, ny - margin),
                float(np.exp(rng.uniform(np.log(peak[0]), np.log(peak[1])))),
            )
        )
    image = np.zeros(shape)
    s = fwhm / 2.3548
    for x, y, p in out:
        r = int(5 * s) + 2
        i0, i1 = max(int(x) - r, 0), min(int(x) + r + 1, nx)
        j0, j1 = max(int(y) - r, 0), min(int(y) + r + 1, ny)
        jj, ii = np.mgrid[j0:j1, i0:i1]
        image[j0:j1, i0:i1] += p * np.exp(
            -((ii + 0.5 - x) ** 2 + (jj + 0.5 - y) ** 2) / (2 * s * s)
        )
    return out, image


def pair(
    seed: int = 12517,
    shape=(560, 700),
    separation: float = 140.0,
    noise: float = 3.0,
    sky: float = 1000.0,
    gradient: Tuple[float, float] = (0.002, -0.001),
    n_stars: int = 30,
    with_noise: bool = True,
) -> Scene:
    """Two overlapping galaxies: A (the larger, re 60 px, e 0.25, at 30 deg) and B (re 25
    px, e 0.40, at -50 deg), ``separation`` pixels apart, with stars, a sky with a gradient
    and noise.  At 140 px (2.3 re of A) the pair fit settles; at 95 px (1.6 re) the blend
    is hard, and with noise it does not."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    ax, ay = nx * 0.42 + 0.3, ny * 0.47 + 0.6
    t = np.radians(25.0)
    a = Galaxy(ax, ay, ie=40.0, re=60.0, ellip=0.25, angle=30.0)
    b = Galaxy(
        ax + separation * np.cos(t),
        ay + separation * np.sin(t),
        ie=30.0,
        re=25.0,
        ellip=0.40,
        angle=-50.0,
    )
    data = sky_plane(shape, sky, gradient) + a.image(shape) + b.image(shape)
    star_list, _ = stars(shape, n_stars, rng)
    # keep the stars off the two centres
    star_list = [s for s in star_list if min(np.hypot(s[0] - g.x, s[1] - g.y) for g in (a, b)) > 15]
    star_image = np.zeros(shape)
    sig = 2.5 / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = np.zeros(shape, bool)
    for x, y, p in star_list:
        star_image += p * np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2 * sig * sig))
        mask |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= 6.0**2
    data = data + star_image
    if with_noise:
        data = data + rng.normal(0.0, noise, shape)
    return Scene(
        data.astype(np.float32),
        sky,
        gradient,
        noise if with_noise else 0.0,
        {"A": a, "B": b},
        star_list,
        mask,
    )


def blank(
    seed: int = 1,
    shape=(600, 800),
    sky: float = 500.0,
    noise: float = 10.0,
    gradient=(0.0, 0.0),
    n_stars: int = 40,
    correlated: float = 0.0,
) -> Scene:
    """A field with stars and sky only (``correlated``: the FWHM, in pixels, of a Gaussian
    that smooths the noise, rescaled to keep its sigma)."""
    rng = np.random.default_rng(seed)
    noise_image = rng.normal(0.0, noise, shape)
    if correlated > 0:
        from scipy import ndimage

        noise_image = ndimage.gaussian_filter(noise_image, correlated / 2.3548)
        noise_image *= noise / noise_image.std()
    star_list, star_image = stars(shape, n_stars, rng)
    data = sky_plane(shape, sky, gradient) + star_image + noise_image
    return Scene(data.astype(np.float32), sky, gradient, noise, {}, star_list, None)


def halo(seed: int = 2, shape=(600, 800), sky: float = 500.0, noise: float = 10.0) -> Scene:
    """A diffuse galaxy whose faint outskirts fill most of the frame."""
    rng = np.random.default_rng(seed)
    g = Galaxy(shape[1] / 2, shape[0] / 2, ie=60.0, re=250.0, ellip=0.1, angle=0.0)
    data = sky + g.image(shape) + rng.normal(0.0, noise, shape)
    return Scene(data.astype(np.float32), sky, (0.0, 0.0), noise, {"A": g})


def write_example(folder) -> Path:
    """The pair example: pair.fits, its fitting mask (stars) as pair.dmask, a README."""
    from astropy.io import fits

    from .masks import Mask, save

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scene = pair()
    header = fits.Header()
    header["OBJECT"] = "synthetic pair"
    header["COMMENT"] = "Made by astro_vista.synthetic.pair() (seed 12517); see README.md"
    for cid, g in scene.galaxies.items():
        header[f"{cid}_X"] = (round(g.x, 3), f"{cid}: centre x (first pixel's centre at 0.5)")
        header[f"{cid}_Y"] = (round(g.y, 3), f"{cid}: centre y")
        header[f"{cid}_RE"] = (g.re, f"{cid}: effective radius (semi-major), px")
        header[f"{cid}_IE"] = (g.ie, f"{cid}: intensity at re")
        header[f"{cid}_E"] = (g.ellip, f"{cid}: ellipticity 1 - b/a")
        header[f"{cid}_PA"] = (g.angle, f"{cid}: major axis, deg from +x")
    header["SKY"] = (scene.sky, "sky level at the frame's centre")
    header["SKYGX"] = (scene.gradient[0], "sky gradient along x, counts per px")
    header["SKYGY"] = (scene.gradient[1], "sky gradient along y, counts per px")
    header["NOISE"] = (scene.noise, "Gaussian noise sigma")
    fits.PrimaryHDU(scene.data, header=header).writeto(folder / "pair.fits", overwrite=True)
    save(Mask("stars", scene.star_mask), folder / "pair.dmask")
    return folder
