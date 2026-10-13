# Other names on PyPI

The program is published as **AstroVista**. PyPI ignores capitals, `-`, `_` and `.` when it
compares names, so AstroVista also covers `astrovista`, `astro-vista`, `Astro_Vista` and so
on: nobody else can register those.

`astrovista-lite` is a different name, kept as a small package that installs AstroVista
(PyPI may give away names that hold empty placeholders). It is built and published by
`.github/workflows/name-packages.yml`, started by hand (Actions ▸ name-packages ▸ Run
workflow). It needs a new release only if its own files change.
