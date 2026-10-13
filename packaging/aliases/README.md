# Other names for AstroVista

The program is published on PyPI as **AstroVista**. PyPI treats capitals, `-`, `_` and `.`
as the same, so `AstroVista` also covers `astrovista`, `Astro_Vista` and so on. The name
`astro-vista` is different, and is kept as a working package that installs AstroVista
(PyPI may give away names that hold empty placeholders):

| Package | Installs |
|---|---|
| `astro-vista` | AstroVista, the same version |

It is built and published with the main package by `.github/workflows/release.yml`, its
version pinned to the same release (`python -m build packaging/aliases/astro-vista`).
