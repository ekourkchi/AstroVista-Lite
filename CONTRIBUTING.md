# Contributing to AstroVista Lite

Thank you for helping. AstroVista Lite is maintained by Ehsan Kourkchi; questions, bug
reports, ideas and pull requests are all welcome.

## Getting help and reporting problems

- **Questions and bugs:** open an issue at
  <https://github.com/ekourkchi/AstroVista-Lite/issues>. Please say which system and Python
  you use (`astrovista --version`, `python3 --version`), what you did, what you expected
  and what happened; the terminal's red error text and a small FITS file that shows the
  problem help most.
- **Ideas:** open an issue describing what you would like to measure or teach and how.
- **Support:** the project is maintained on a best-effort basis; issues are usually
  answered within a week or two.

## Contributing code or documentation

1. Fork the repository and make a branch.
2. Set up a development environment:

   ```sh
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e '.[test]'
   ```

3. Make the change, with a test when it changes behaviour (`tests/test_demo.py`); the tests
   run offscreen with `pytest -q`, and `python tests/smoke.py out` runs a whole session in
   the real window.
4. Format with Black at line length 100 (`black -l 100 src tests`).
5. If you add or change a terminal command, describe it in `src/astro_vista/reference.py`
   (the tests check that every command is documented and that every example runs) and run
   `python docs/build.py` to rebuild the website's reference pages.
6. Open a pull request saying what it changes and why. The tests run on Linux and macOS
   for every pull request.

Code is contributed under the project's MIT licence. AstroVista Lite holds no code from
MONSTA or other packages: new features are written fresh, and the libraries it uses are
installed as dependencies.

## Conduct

Be kind and constructive. Everyone is welcome, whatever their experience; questions from
students are as welcome as patches.
