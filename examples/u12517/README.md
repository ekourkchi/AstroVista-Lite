# Example: UGC 12517

An HST WFC3/IR F110W image of the galaxy UGC 12517 and its pixel mask, from the example
of the [`elliprof`](https://github.com/ekourkchi/elliprof) package.

| File | What it is |
|---|---|
| `u12517j.fits` | The image: 1025 × 1022 pixels, 0.128″ per pixel. |
| `u12517j.dmask` | Its mask, a legacy BITPIX = 1 bitmap: 0 = masked, 1 = good (103,402 pixels masked). |
| `session.txt` | The commands of the example: read, mask, find the sources, fit. |

```sh
cd examples/u12517
astrovista --run session.txt      # or: astrovista u12517j.fits, then File ▸ Open u12517j.dmask
```

The published fit: X0=567 Y0=562 R0=9 R1=347 NR=23 NITER=10 RMSTAR, with a sky of 3246
subtracted. The same two files ship inside the package, so `astrovista --example` works
without this folder.
