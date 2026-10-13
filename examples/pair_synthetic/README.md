# A synthetic pair of galaxies

`pair.fits` (700 x 560 pixels) holds two overlapping elliptical galaxies that follow de
Vaucouleurs' law, 30 stars, a sky of 1000 with a small gradient, and Gaussian noise
(sigma 3). Because the image is made by `astro_vista.synthetic.pair()` (seed 12517), the
right answer is known. It is written in the FITS header:

| | centre (x, y) | effective radius | ellipticity | major axis |
|---|---|---|---|---|
| A | 294.3, 263.8 | 60 px | 0.25 | 30 deg from +x |
| B | 421.18, 322.97 | 25 px | 0.40 | -50 deg from +x |

Positions are ELLIPROF's coordinates, where the first pixel's centre is at 0.5.
`pair.dmask` masks the stars, in the legacy bitmap format.

`session.txt` runs the whole fit: the sky by segmentation, then the pair. The same files
ship inside the package, and **File ▸ Open the pair example** loads them in the window.
