"""The work kept between sessions: every buffer (a FITS file with its name, tag, note and
model parameters) and every mask layer (its pixels, colour, role and how it was made), in a
folder of their own (the session folder, by default .temp next to the settings).  Saved when
buffers are closed and when the window closes, and from time to time while working; read
back with ``session restore``.  A saved session is never replaced by an empty one, and the
one before is kept as FOLDER.previous."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from .images import Image
from .masks import Mask

MANIFEST = "session.json"
VERSION = 1


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60] or "layer"


def save(session, folder: Path) -> Optional[str]:
    """Write the session's buffers and mask layers to ``folder`` (written beside it first,
    then swapped in, so a failed save leaves the last one whole).  None when there is
    nothing to save."""
    if not session.buffers and not session.masks:
        return None
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    work = folder.with_name(folder.name + ".saving")
    shutil.rmtree(work, ignore_errors=True)
    (work / "buffers").mkdir(parents=True)
    (work / "masks").mkdir()
    buffers, masks = [], []
    for n, b in sorted(session.buffers.items()):
        file = f"buffers/{n}.fits"
        fits.PrimaryHDU(np.asarray(b.data, np.float32), b.header).writeto(
            work / file, output_verify="silentfix"
        )
        buffers.append(
            {
                "n": n,
                "file": file,
                "name": b.name,
                "path": str(b.path) if b.path else "",
                "family": b.family,
                "tag": b.tag,
                "note": b.note,
                "display": b.display,
                "about": b.about,
            }
        )
    for k, (name, m) in enumerate(session.masks.items()):
        file = f"masks/{k}_{_safe(name)}.npz"
        np.savez_compressed(work / file, bits=np.packbits(m.masked), shape=m.masked.shape)
        masks.append(
            {
                "name": name,
                "file": file,
                "colour": m.colour,
                "visible": m.visible,
                "opacity": m.opacity,
                "source": m.source,
                "role": m.role,
            }
        )
    manifest = {
        "version": VERSION,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "folder": str(session.folder),
        "buffers": buffers,
        "masks": masks,
        "project": {
            "name": session.project.name,
            "shape": list(session.project.shape) if session.project.shape else None,
            "files": list(session.project.files),
        },
    }
    (work / MANIFEST).write_text(json.dumps(manifest, indent=1))
    # the save before this one is kept beside it (FOLDER.previous), in case this one is
    # not what was wanted
    previous = folder.with_name(folder.name + ".previous")
    if folder.exists():
        shutil.rmtree(previous, ignore_errors=True)
        folder.rename(previous)
    work.rename(folder)
    return f"saved {len(buffers)} buffers and {len(masks)} mask layers to {folder}"


def describe(folder: Path) -> Optional[dict]:
    """What a session folder holds: {saved, buffers, masks} counts, or None (none, or
    unreadable)."""
    try:
        manifest = json.loads((Path(folder) / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    return {
        "saved": manifest.get("saved", ""),
        "buffers": len(manifest.get("buffers", [])),
        "masks": len(manifest.get("masks", [])),
    }


def restore(session, folder: Path) -> str:
    """Read a saved session back into ``session``: its buffers (to their own numbers, or
    free ones when taken) and its mask layers (renamed when the name is taken)."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST).read_text())
    # one project: its images all of one size
    saved = manifest.get("project") or {}
    shapes = {tuple(int(v) for v in saved["shape"])} if saved.get("shape") else set()
    for entry in manifest.get("buffers", []):
        with fits.open(folder / entry["file"], memmap=False) as hdul:
            shapes.add(tuple(hdul[0].data.shape))
    project = session.project
    if project.shape is not None and shapes - {tuple(project.shape)}:
        size = " and ".join(f"{s[1]} x {s[0]}" for s in sorted(shapes))
        raise ValueError(
            f"its images are {size}, this project's {project.shape[1]} x {project.shape[0]}: "
            "close the project first (project close)"
        )
    if saved.get("name") and not project.name:
        project.name = saved["name"]
    if project.shape is None and len(shapes) == 1:
        project.shape = next(iter(shapes))
    for f in saved.get("files", []):
        if f not in project.files:
            project.files.append(f)
    numbers = []
    for entry in manifest.get("buffers", []):
        with fits.open(folder / entry["file"], memmap=False) as hdul:
            data = np.asarray(hdul[0].data, np.float32)
            header = hdul[0].header.copy()
        image = Image(
            entry["name"],
            data,
            header,
            path=Path(entry["path"]) if entry.get("path") else None,
            family=entry.get("family"),
            tag=entry.get("tag", ""),
            note=entry.get("note", ""),
            display=entry.get("display", ""),
            about=entry.get("about", ""),
        )
        n = int(entry["n"])
        if n in session.buffers:
            n = session.free_buffer(n)
        session.buffers[n] = image
        numbers.append(n)
    names = []
    for entry in manifest.get("masks", []):
        got = np.load(folder / entry["file"])
        shape = tuple(int(v) for v in got["shape"])
        masked = np.unpackbits(got["bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        name = entry["name"]
        while name in session.masks:
            name += "_restored"
        session.masks[name] = Mask(
            name,
            masked,
            entry.get("colour", "#ff0000"),
            entry.get("visible", True),
            entry.get("opacity", 0.45),
            entry.get("source", ""),
            entry.get("role", ""),
        )
        names.append(name)
    return (
        f"restored the session saved {manifest.get('saved', '')}: {len(numbers)} buffers"
        + (f" ({', '.join(map(str, numbers))})" if numbers else "")
        + f" and {len(names)} mask layers"
        + (f" ({', '.join(names)})" if names else "")
    )
