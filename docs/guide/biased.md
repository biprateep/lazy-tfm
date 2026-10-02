# Biased training sets

DC1 gives every training galaxy a redshift. Real spectroscopic samples do not
look like this, since they are bright, incomplete and cut in redshift by which
features fall in the observed window, and this mismatch, rather than the
estimator, is usually what limits the redshifts of a survey.
{func}`~lazy.datasets.fetch_dc1_biased` merges both DC1 files and cuts the
realistic case out of them:

```python
from lazy.datasets import fetch_dc1_biased

split = fetch_dc1_biased()          # 35,011 biased / 10,000 calibration / 19,383 test
split.biased                        # what a HSC-like campaign would have got
split.calibration                   # representative, yours to repair the model with
split.test                          # representative, held out from all of it
print(split.summary())              # each subset against the parent catalog
```

Both files are merged and reshuffled, and then cut in two. The first part is
run through the selection, and the galaxies it keeps form the training set,
while the second part is divided at random into the calibration sample and the
test set. The position of the cut is solved for rather than chosen, because
the selection keeps a nearly fixed fraction of whatever it sees, so the cut is
controlled through `n_train`. The calibration sample is taken *out* of the
hold-out rather than added on top of it, so that no galaxy is both given to a
model and scored on.

`n_train` cannot reach the 43,486 galaxies of the DC1 training file. The
selection keeps 8.66 percent of a representative sample, so all 434,476
galaxies yield at most 37,626, and that with nothing left to test on. 35,000
is the largest round number that still leaves a usable hold-out.

`control=True` adds `split.unbiased`, which holds exactly as many galaxies
drawn at random from the same pool. This is the control that separates the
effect of the selection from that of the sample size. It is drawn last, so
asking for it changes nothing else in the split.

The selection is a port of RAIL's HSC `GridSelection`, in which each galaxy is
kept with the HSC spectroscopic success rate of its (i, g−z) pixel, below a
color-dependent redshift ceiling. It is reproduced galaxy for galaxy against
the runs this work reports. The selection also acts on any photometry, so a
catalog of the user's own can be biased in the same way:

```python
from lazy.selection import grid_selection, selection_summary

keep, diagnostics = grid_selection(catalog.raw, catalog.redshift)
print(selection_summary(keep, catalog.raw["I"], catalog.redshift))
#  i in [16.0, 20.0)  ...  fraction 0.661
#  i in [24.0, 25.3)  ...  fraction 0.0005
```

`diagnostics` carries the pixel success rate and the redshift ceiling of each
galaxy, which is everything the selection knew, for use by a method that tries
to estimate the selection back.
