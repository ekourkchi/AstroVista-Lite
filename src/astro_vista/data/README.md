# The example

- `u12517j.fits.gz`: HST WFC3/IR F110W image of the galaxy UGC 12517, 1025 x 1022 pixels,
  0.128 arcsec per pixel (gzip-compressed).
- `u12517j.dmask`: its pixel mask, a legacy BITPIX = 1 bitmap (0 = masked, 1 = good).

Both come from the example of the `elliprof` package
(https://github.com/ekourkchi/elliprof, `examples/u12517`), where the fit parameters are
published: X0=567 Y0=562 R0=9 R1=347 NR=23 NITER=10 RMSTAR, sky 3246.
- `pair.fits.gz`, `pair.dmask`: a synthetic pair of overlapping galaxies with stars, made by
  `astro_vista.synthetic.pair()` (seed 12517); the true centres, radii, ellipticities and
  angles are in its FITS header (see `examples/pair_synthetic` in the repository).
