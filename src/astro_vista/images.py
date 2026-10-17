"""Images held by AstroVista Lite: FITS images read into memory, and the images made from them.

Coordinates.  The viewer puts pixel ``data[j, i]`` between ``i`` and ``i + 1`` (and ``j``
and ``j + 1``), so its centre is at ``(i + 0.5, j + 0.5)``.  ELLIPROF uses the same
coordinates for its centre (X0, Y0).  FITS, DS9 and SExtractor put the centre of the first
pixel at 1, so they are 0.5 larger: ``fits = view + 0.5``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits

# keywords that describe how the data were stored on disk, not the data themselves
STORAGE_KEYS = ("BSCALE", "BZERO", "BLANK")


@dataclass
class Image:
    name: str
    data: np.ndarray  # 2-D float32
    header: fits.Header = field(default_factory=fits.Header)
    path: Optional[Path] = None
    family: Optional[str] = None  # the image a model or residual was made from
    tag: str = ""  # a name given by the user (tag="...")
    note: str = ""  # how it was made (the command or the arithmetic)
    display: str = ""  # how to show it first: "" (the display chosen) or "smooth" (min to max)
    about: str = ""  # for a model or residual of an iterative fit: the model's parameters

    @property
    def label(self) -> str:
        """What the buffer is called on screen: its tag, or its name."""
        return self.tag or self.name

    @property
    def key(self) -> str:
        """Images made from one image share its fits and sources (the same pixels)."""
        return self.family or self.name

    @property
    def shape(self):
        return self.data.shape

    def wcs(self):
        """The celestial WCS of the header, or None when it has none."""
        if not hasattr(self, "_wcs"):
            self._wcs = None
            try:
                from astropy.wcs import WCS

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = WCS(self.header, naxis=2)
                if w.has_celestial:
                    self._wcs = w.celestial
            except Exception:  # an unreadable WCS: show pixel coordinates only
                self._wcs = None
        return self._wcs

    def derived(self, name: str, data: np.ndarray) -> "Image":
        """A new image of the same size and WCS (a model, a residual, a segmentation)."""
        data = np.asarray(data, dtype=np.float32)
        return Image(name, data, self.header.copy(), None, self.key)


def first_header(path) -> Optional[str]:
    """The first 2880-byte header block of a FITS file (gzip-compressed or not), whatever
    the file is called; None when the file is not FITS."""
    import gzip

    try:
        with open(path, "rb") as f:
            head = f.read(2)
        opener = gzip.open if head == b"\x1f\x8b" else open
        with opener(path, "rb") as f:
            block = f.read(2880)
    except OSError:
        return None
    return block.decode("ascii", "replace") if block.startswith(b"SIMPLE  =") else None


def read_bitmap(path) -> np.ndarray:
    """A legacy bitmap mask's values (1 = good, 0 = masked), gzip-compressed or not."""
    import gzip
    import tempfile

    from elliprof import load_mask

    path = Path(path)
    with open(path, "rb") as f:
        zipped = f.read(2) == b"\x1f\x8b"
    if not zipped:
        return np.asarray(load_mask(str(path)), np.float32)
    with tempfile.TemporaryDirectory() as folder:  # elliprof reads plain files only
        plain = Path(folder) / "mask.dmask"
        with gzip.open(path, "rb") as f:
            plain.write_bytes(f.read())
        return np.asarray(load_mask(str(plain)), np.float32)


def is_fits(path) -> bool:
    return first_header(path) is not None


def is_bitmap_mask(path) -> bool:
    """A legacy BITPIX = 1 bitmap mask (such as a .dmask), which astropy cannot read."""
    header = first_header(path)
    if header is None:
        return False
    cards = [header[k : k + 80] for k in range(0, len(header), 80)]
    return any(c.startswith("BITPIX") and c[10:30].strip() == "1" for c in cards)


def image_hdus(path) -> List[int]:
    """The HDUs of a FITS file that hold an image of two or more dimensions."""
    if is_bitmap_mask(path):
        return [0]
    found = []
    with fits.open(path) as hdul:
        for k, hdu in enumerate(hdul):
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
                if hdu.header.get("NAXIS", 0) >= 2:
                    found.append(k)
    return found


def read_fits(path, ext: Optional[int] = None) -> Image:
    """Read an image from a FITS file: HDU ``ext``, or the first one that holds an image
    (a header-only primary HDU, as in HST and JWST files, is skipped)."""
    path = Path(path).expanduser()
    if not is_fits(path):
        raise ValueError(f"{path.name} is not a FITS file")
    if is_bitmap_mask(path):  # a legacy bitmap: 1 = good, 0 = masked
        return Image(path.name, read_bitmap(path), fits.Header(), path)
    if ext is None:
        found = image_hdus(path)
        if not found:
            raise ValueError(f"{path.name} holds no image (only tables or headers)")
        ext = found[0]
    with fits.open(path) as hdul:
        hdu = hdul[ext]
        if hdu.data is None:
            raise ValueError(f"HDU {ext} of {path.name} holds no data")
        data = np.squeeze(np.asarray(hdu.data))
        header = hdu.header.copy()
    if data.ndim != 2:
        raise ValueError(f"HDU {ext} of {path.name} is {data.ndim}-dimensional, not an image")
    label = str(header.get("EXTNAME", "")).strip() or str(ext)
    if label != str(ext) and int(header.get("EXTVER", 1) or 1) > 1:
        label += f",{int(header['EXTVER'])}"
    name = path.name if ext == 0 else f"{path.name}[{label}]"
    return Image(name, data.astype(np.float32), header, path)


def write_fits(image: Image, path, extra: Optional[dict] = None) -> Path:
    """Write ``image`` as a single-HDU FITS file (float32)."""
    path = Path(path).expanduser()
    header = image.header.copy()
    for key in STORAGE_KEYS:
        header.remove(key, ignore_missing=True)
    for key, value in (extra or {}).items():
        header[key] = value
    fits.PrimaryHDU(image.data.astype(np.float32), header=header).writeto(
        path, overwrite=True, output_verify="silentfix"
    )
    return path
