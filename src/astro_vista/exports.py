"""Files written for other programs: the source list as a comma-separated table whose columns
line up, and the profile and source plots as PNG figures made with matplotlib.

The CSV files are ordinary CSV with spaces before each value so that the columns align in a
text editor; read them with ``pandas.read_csv(path, skipinitialspace=True)``,
``astropy.table.Table.read(path, format="ascii.csv")`` or any reader that ignores the
spaces around values.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

# number formats by column (others: integers as integers, the rest with 6 significant
# figures); fixed decimals keep the decimal points under each other
FORMATS = {
    "X_IMAGE": "{:.3f}",
    "Y_IMAGE": "{:.3f}",
    "ALPHA_J2000": "{:.7f}",
    "DELTA_J2000": "{:.7f}",
    "THETA_IMAGE": "{:.2f}",
    "ELLIPTICITY": "{:.4f}",
    "CLASS_STAR": "{:.3f}",
}
PREFIX_FORMATS = (  # by the start of the name
    ("MAGERR", "{:.4f}"),
    ("MAG_", "{:.4f}"),
    ("FLUXERR", "{:.5e}"),
    ("FLUX_RADIUS", "{:.3f}"),
    ("FLUX_", "{:.5e}"),
    ("FWHM", "{:.3f}"),
    ("KRON", "{:.3f}"),
    ("A_", "{:.3f}"),
    ("B_", "{:.3f}"),
)


def _format(name: str, values: np.ndarray):
    """How the values of one column are written."""
    if name in FORMATS:
        return FORMATS[name]
    for prefix, fmt in PREFIX_FORMATS:
        if name.startswith(prefix):
            return fmt
    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if values.dtype.kind in "iub" or (finite.size and np.all(finite == np.round(finite))):
        return "{:.0f}"
    return "{:.6g}"


def aligned_csv(path, columns: Dict[str, Sequence], formats: Optional[dict] = None) -> None:
    """Write ``columns`` (name -> values) as CSV whose columns are right-aligned: each
    field is padded to the width of its column, so the commas line up.  ``formats`` gives
    the number format of some columns by name (the others: as for the sources)."""
    names = list(columns)
    cells = []
    for name in names:
        values = np.asarray(columns[name])
        if values.dtype.kind in "iufb":
            fmt = (formats or {}).get(name) or _format(
                name, values.astype(float) if values.dtype.kind == "b" else values
            )
            text = [
                "nan" if isinstance(v, float) and not math.isfinite(v) else fmt.format(v)
                for v in values.astype(float).tolist()
            ]
        else:  # text: commas would break the columns
            text = [str(v).replace(",", ";") for v in values.tolist()]
        cells.append(text)
    widths = [max([len(n)] + [len(t) for t in col]) for n, col in zip(names, cells)]
    rows = len(cells[0]) if cells else 0
    with open(path, "w") as f:
        f.write(", ".join(n.rjust(w) for n, w in zip(names, widths)) + "\n")
        for k in range(rows):
            f.write(", ".join(col[k].rjust(w) for col, w in zip(cells, widths)) + "\n")


def devauc_fit(r: np.ndarray, i0: np.ndarray) -> dict:
    """The de Vaucouleurs law I(R) = Ie exp(-7.669 [(R/Re)^¼ - 1]) fitted as a straight line
    of ln I against R^¼.  Returns re, ie, scatter (magnitudes), n and the line's slope and
    intercept, or {"error": why} when there is no law to fit."""
    good = (i0 > 0) & np.isfinite(i0) & (r > 0)
    if good.sum() < 3:
        return {"error": "The R¼ law needs three ellipses or more with I0 > 0."}
    slope, intercept = np.polyfit(r[good] ** 0.25, np.log(i0[good]), 1)
    if slope >= 0:
        return {"error": "The profile does not fall with radius: no R¼ law to fit."}
    model = np.exp(intercept + slope * r[good] ** 0.25)
    return dict(
        re=(-7.669 / slope) ** 4,
        ie=float(np.exp(intercept - 7.669)),
        scatter=float(np.std(2.5 * np.log10(i0[good] / model))),
        n=int(good.sum()),
        slope=float(slope),
        intercept=float(intercept),
        rmin=float(r[good].min()),
        rmax=float(r[good].max()),
    )


def _figure():
    """A matplotlib figure that needs no window (the Agg canvas)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.4, 4.6), dpi=150)
    FigureCanvasAgg(fig)
    return fig


def profile_png(
    path,
    profile,
    label: str,
    column: str,
    logarithmic: bool,
    quarter: bool,
    devauc: bool,
    title: str = "",
) -> None:
    """The ELLIPROF profile as shown in Results ▸ Profile: ``column`` (called ``label``)
    against Rmaj or Rmaj^¼, with the R¼ law when asked (drawn over the intensity)."""
    r = profile["Rmaj"].to_numpy(float)
    y = profile[column].to_numpy(float)
    x = r**0.25 if quarter else r
    if logarithmic:
        keep = y > 0
        x, y = x[keep], y[keep]
    fig = _figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o-", color="#2f7fd8", markersize=4, linewidth=1.3, label=label)
    if logarithmic:
        ax.set_yscale("log")
    note = ""
    if devauc:
        law = devauc_fit(r, profile["I0"].to_numpy(float))
        if "error" in law:
            note = law["error"]
        else:
            note = (
                f"R¼ law: Re = {law['re']:.1f} px, Ie = {law['ie']:.4g}, "
                f"scatter {law['scatter']:.3f} mag"
            )
            if column == "I0":
                rr = np.linspace(law["rmin"], law["rmax"], 200)
                curve = np.exp(law["intercept"] + law["slope"] * rr**0.25)
                ax.plot(rr**0.25 if quarter else rr, curve, "--", color="#e8891c", label="R¼ law")
    ax.set_xlabel("Rmaj$^{1/4}$ (pixels$^{1/4}$)" if quarter else "Rmaj (pixels)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    if devauc and column == "I0" and "error" not in law:
        ax.legend(fontsize=8)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path)


def scatter_png(
    path,
    x: np.ndarray,
    y: np.ndarray,
    xname: str,
    yname: str,
    marked: Optional[int] = None,
    title: str = "",
) -> None:
    """A scatter plot of two source columns as in Results ▸ Plots, with the selected
    source (``marked``, a row) circled."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    fig = _figure()
    ax = fig.add_subplot()
    ax.scatter(x[ok], y[ok], s=8, color="#1aa3c8", alpha=0.7, linewidths=0)
    if marked is not None and 0 <= marked < len(x) and ok[marked]:
        ax.scatter(
            [x[marked]], [y[marked]], s=90, facecolors="none", edgecolors="#e8a800", linewidths=1.6
        )
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
