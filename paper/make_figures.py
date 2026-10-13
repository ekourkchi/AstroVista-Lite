"""The figures and numbers of the AstroVista Lite papers, made from the UGC 12517 example
with the installed package:

    python paper/make_figures.py

Writes paper/arxiv/figures/*.pdf and paper/numbers.json.  The elliprof command-line
program (installed with elliprof) is run too, for the comparison with elliprof's own
published fit.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from astro_vista import __version__, exports  # noqa: E402
from astro_vista.commands import Session  # noqa: E402

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "examples" / "u12517"
FIGURES = HERE / "arxiv" / "figures"
FIT = "X0=567 Y0=562 R0=9 R1=347 NR=23 NITER=10 SKY=3246 RMSTAR"


def figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=200)
    FigureCanvasAgg(fig)
    return fig


def session() -> Session:
    s = Session()
    s.folder = EXAMPLE
    s.execute("rd 1 u12517j.fits")
    s.execute("mask load u12517j u12517j.dmask 1")
    return s


def elliprof_reference(work: Path) -> pd.DataFrame:
    """The profile of elliprof's published example command (examples/u12517 in elliprof)."""
    program = shutil.which("elliprof")
    if program is None:
        raise SystemExit("the elliprof command is not on the PATH")
    subprocess.run(
        [program, str(EXAMPLE / "u12517j.fits"), "--mask", str(EXAMPLE / "u12517j.dmask")]
        + ["--sky", "3246.0"]
        + FIT.replace("SKY=3246 ", "").split()
        + ["-o", str(work / "u.dat"), "--csv", str(work / "u.csv")],
        check=True,
        capture_output=True,
    )
    lines = (work / "u.csv").read_text().splitlines()
    head = [line for line in lines if line.startswith("#")][-1].lstrip("#")
    body = [line for line in lines if not line.startswith("#")]
    table = pd.read_csv(io.StringIO(head + "\n" + "\n".join(body)), skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    return table


def timed(s: Session, line: str, repeat: int = 3) -> float:
    """The best of ``repeat`` runs of a command, in seconds."""
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        s.execute(line)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    numbers: dict = {"version": __version__}

    # 1. the published fit, through AstroVista Lite's command, against elliprof's own program
    a = session()
    a.execute(f"elliprof 1 {FIT} MASK=u12517j TOMODEL=3 TORESID=4")
    fit_a = next(iter(a.fits.values()))
    with tempfile.TemporaryDirectory() as tmp:
        ref = elliprof_reference(Path(tmp))
    rel = {}
    for column in ("Rmaj", "x0", "y0", "I0", "ellip", "alpha", "I4", "slope"):
        x, y = fit_a.profile[column].to_numpy(float), ref[column].to_numpy(float)
        rel[column] = float(np.nanmax(np.abs(x - y) / np.maximum(np.abs(y), 1e-30)))
    numbers["validation_max_relative_difference"] = rel
    numbers["isophotes"] = int(len(fit_a.profile))
    numbers["published_mask_pixels"] = int(a.masks["u12517j"].count())

    # 2. the sources found, masked, and the fit repeated
    b = session()
    b.execute("sources 1")
    run = b.runs[-1]
    b.execute("mask new sources 1")
    b.execute("mask sources sources SCALE=1.5")
    b.execute(f"elliprof 1 {FIT} MASK=u12517j,sources TOMODEL=3 TORESID=4")
    fit_b = next(iter(b.fits.values()))
    numbers["sources_found"] = int(len(run))
    numbers["sources_mask_pixels"] = int(b.masks["sources"].count())
    numbers["masked_pixels_both"] = int(fit_b.masked.sum())
    image_pixels = int(np.prod(b.buffers[1].shape))
    numbers["image_pixels"] = image_pixels
    for key, fit in (("published_mask", fit_a), ("both_masks", fit_b)):
        p = fit.profile
        law = exports.devauc_fit(p["Rmaj"].to_numpy(float), p["I0"].to_numpy(float))
        numbers[f"devauc_{key}"] = {
            k: round(float(law[k]), 4) for k in ("re", "ie", "scatter", "n") if k in law
        }
    # the residual as a fraction of the model, on the same pixels for both fits: those
    # neither mask covers, inside the last isophote's semi-major axis
    yy, xx = np.mgrid[0 : image_pixels // b.buffers[1].shape[1], 0 : b.buffers[1].shape[1]]
    inside = np.hypot(xx + 0.5 - 567, yy + 0.5 - 562) < float(fit_b.profile["Rmaj"].iloc[-1])
    common = inside & np.isfinite(fit_a.residual) & np.isfinite(fit_b.residual)
    numbers["residual_pixels_compared"] = int(common.sum())
    for key, fit, s in (("published_mask", fit_a, a), ("both_masks", fit_b, b)):
        model = s.buffers[3].data
        ok = common & (model > 0)
        numbers[f"residual_median_abs_fraction_{key}"] = round(
            float(np.median(np.abs(fit.residual[ok] / model[ok]))), 5
        )
    i_a, i_b = fit_a.profile["I0"].to_numpy(float), fit_b.profile["I0"].to_numpy(float)
    numbers["outer_intensity_change_percent"] = [
        round(float(100 * (i_b[k] - i_a[k]) / i_a[k]), 2) for k in range(-4, 0)
    ]
    numbers["outer_radii"] = [round(float(r), 1) for r in fit_a.profile["Rmaj"][-4:]]

    # 3. how long the commands take on this 1025 x 1022 image
    t = session()
    t.execute("copy 1 7")
    t.execute("copy 1 9")
    t.buffers[9].data[::50, ::50] = np.nan
    commands = {
        "read the image (gzip-free FITS)": "rd 1 u12517j.fits",
        "ELLIPROF, 23 isophotes, model and residual": f"elliprof 1 {FIT} MASK=u12517j "
        "TOMODEL=3 TORESID=4",
        "source finder": "sources 1",
        "arithmetic: [5] = [1] - median([1])": "[5] = [1] - median([1])",
        "median of three images": "[6] = median([1], [3], [4])",
        "Gaussian smoothing, FWHM 3": "smooth 7 FWHM=3 OUT=8",
        "median filter 5 x 5, with NaN pixels": "filt 9 MEDIAN=5 OUT=8",
        "cosmic-ray cleaning (zap)": "zap 7 OUT=8",
        "rotation by 12.5 degrees": "rotate 7 ANGLE=12.5 OUT=8",
    }
    numbers["timings_s"] = {label: round(timed(t, line), 3) for label, line in commands.items()}

    # figure: the profiles, with the published mask and with the sources masked as well
    fig = figure(7.0, 2.6)
    axes = fig.subplots(1, 3)
    for fit, label, style in (
        (fit_a, "published mask", dict(color="#7f7f7f", marker="o", ms=3, lw=1)),
        (fit_b, "+ found sources", dict(color="#2f7fd8", marker="s", ms=3, lw=1)),
    ):
        p = fit.profile
        r4 = p["Rmaj"].to_numpy(float) ** 0.25
        axes[0].plot(r4, -2.5 * np.log10(p["I0"].to_numpy(float)), label=label, **style)
        axes[1].plot(p["Rmaj"], p["ellip"], **style)
        axes[2].plot(p["Rmaj"], p["I4"], **style)
    law = exports.devauc_fit(
        fit_b.profile["Rmaj"].to_numpy(float), fit_b.profile["I0"].to_numpy(float)
    )
    rr = np.linspace(law["rmin"], law["rmax"], 100)
    axes[0].plot(
        rr**0.25,
        -2.5 * (law["intercept"] + law["slope"] * rr**0.25) / np.log(10),
        "--",
        color="#e8891c",
        lw=1,
        label=r"$R^{1/4}$ law",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$R_{\rm maj}^{1/4}$ (pixel$^{1/4}$)")
    axes[0].set_ylabel(r"$-2.5\,\log_{10} I_0$")
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_xlabel(r"$R_{\rm maj}$ (pixel)")
    axes[1].set_ylabel(r"ellipticity $1 - b/a$")
    axes[2].set_xlabel(r"$R_{\rm maj}$ (pixel)")
    axes[2].set_ylabel(r"$\cos 4\theta$ amplitude $I_4$")
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "profiles.pdf")

    # figure: the image, the model and the residual (both masks), on matching stretches
    data = b.buffers[1].data - 3246.0
    model, resid = b.buffers[3].data, fit_b.residual
    fig = figure(7.0, 2.5)
    axes = fig.subplots(1, 3)
    lim = np.nanpercentile(data, 99.5)
    for ax, img, title in ((axes[0], data, "image $-$ sky"), (axes[1], model, "ELLIPROF model")):
        ax.imshow(
            np.arcsinh(np.clip(img, 0, None) / 30),
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=np.arcsinh(lim / 30),
        )
        ax.set_title(title, fontsize=8)
    r_lim = np.nanpercentile(np.abs(resid), 98)
    shown = np.where(np.isfinite(resid), resid, np.nan)
    axes[2].imshow(shown, origin="lower", cmap="RdBu_r", vmin=-r_lim, vmax=r_lim)
    axes[2].set_facecolor("#bbbbbb")
    axes[2].set_title("residual (masked: grey)", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "images.pdf")

    # figure: the found sources, size against magnitude
    c = run.columns
    fig = figure(3.4, 2.7)
    ax = fig.add_subplot()
    ok = np.isfinite(c["MAG_AUTO"]) & np.isfinite(c["FLUX_RADIUS"])
    ax.scatter(c["MAG_AUTO"][ok], c["FLUX_RADIUS"][ok], s=4, color="#1aa3c8", lw=0)
    ax.set_xlabel("MAG_AUTO (instrumental)", fontsize=8)
    ax.set_ylabel("FLUX_RADIUS (pixel)", fontsize=8)
    ax.set_yscale("log")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "sources.pdf")

    # the window, from the website's screenshots
    for name in ("hero", "mask_rotate", "terminal"):
        shutil.copy(HERE.parent / "docs" / "images" / f"{name}.png", FIGURES / f"{name}.png")

    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
