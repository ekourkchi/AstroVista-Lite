"""Make the website's terminal.html and reference.html from astro_vista/reference.py, the
source the documentation window uses too:

    python docs/build.py

The tests check that the pages in docs/ are what this script makes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from astro_vista import __version__, reference

DOCS = Path(__file__).resolve().parent

NAV = (
    ("index.html", "Home"),
    ("tutorial.html", "Tutorial"),
    ("pairs.html", "Pairs"),
    ("terminal.html", "Terminal"),
    ("reference.html", "Reference"),
    ("commands.html", "Recipes"),
)


def page(name: str, title: str, description: str, kicker: str, heading: str, body: str) -> str:
    """A page of the website, with the bar at the top and the footer of the others."""
    here = ' class="here"'
    nav = "\n".join(
        f'      <a{here if href == name else ""} href="{href}">{label}</a>' for href, label in NAV
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="icon" href="images/logo.svg" type="image/svg+xml">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="top">
  <div class="wrap">
    <a class="brand" href="index.html"><img src="images/logo.svg" alt="">AstroVista Lite</a>
    <nav>
{nav}
      <a href="https://github.com/ekourkchi/AstroVista-Lite">GitHub</a>
    </nav>
  </div>
</header>

<main class="wrap narrow">
<section class="doc">
  <p class="kicker">{kicker}</p>
  <h1>{heading}</h1>
{body}
</section>
</main>

<footer>
  <div class="wrap">
    <span>AstroVista Lite {__version__} · Ehsan Kourkchi · MIT licence</span>
    <span><a href="index.html">Home</a> · <a href="tutorial.html">Tutorial</a> · <a href="https://github.com/ekourkchi/AstroVista-Lite">GitHub</a></span>
  </div>
</footer>
</body>
</html>
"""


def pages() -> dict:
    """The generated pages: file name -> HTML."""
    terminal = page(
        "terminal.html",
        "AstroVista Lite Terminal",
        "How the AstroVista Lite terminal works: how a command line is written, buffers, "
        "coordinates, formulas, variables, programs on your computer, the keyboard and "
        "command files.",
        "Guide",
        "The terminal",
        reference.TERMINAL_HTML + "\n<p>Every command, with its keywords and an example: the "
        '<a href="reference.html">Command reference</a>.</p>',
    )
    ref = page(
        "reference.html",
        "AstroVista Lite Command Reference",
        "Every command of the AstroVista Lite terminal, with its syntax, its keywords and an "
        "example: images and buffers, arithmetic, statistics, geometry, smoothing, the "
        "header, variables, ELLIPROF, the source finder and masks.",
        "Reference",
        "Command reference",
        "<p>Every command the terminal knows, with how it is written, what it does, its "
        "keywords and an example; the examples run as written, in order, on the example "
        'image. <a href="terminal.html">The terminal</a> explains how a line is written. '
        "In the syntax, <code>[ ]</code> marks what is optional and <code>|</code> a "
        "choice; N and M are buffer numbers. Commands marked <span class='monsta'>MONSTA "
        "name</span> share their name with a command of MONSTA, John Tonry's image analysis "
        "and plotting system; their options are AstroVista Lite's own, as given here.</p>\n"
        + reference.reference_html(website=True),
    )
    return {"terminal.html": terminal, "reference.html": ref}


def main() -> int:
    for name, text in pages().items():
        (DOCS / name).write_text(text)
        print("wrote", DOCS / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
