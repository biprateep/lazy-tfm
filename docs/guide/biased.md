# Biased training sets

DC1 hands every training galaxy a redshift. Real spectroscopic samples do not
look like that — they are bright, incomplete, and cut in redshift by which
features fall in the observed window — and that mismatch, not the estimator, is
usually what limits a survey's redshifts. {func}`~lazy.datasets.fetch_dc1_biased` merges both DC1
files and cuts the realistic case out of them:

```python
from lazy.datasets import fetch_dc1_biased

split = fetch_dc1_biased()          # 35,011 biased / 10,000 calibration / 19,383 test
split.biased                        # what a HSC-like campaign would have got
split.calibration                   # representative, yours to repair the model with
split.test                          # representative, held out from all of it
print(split.summary())              # each subset against the parent catalogue
```

Both files are merged and reshuffled, then cut in two: the first part is run
through the selection and what it keeps is the training set, the second is
divided at random into the calibration sample and the test set. Where the cut
falls is solved for, not chosen, because the selection keeps a near-fixed
fraction of whatever it sees — `n_train` is the knob. The calibration sample
comes *out* of the hold-out rather than on top of it, so no galaxy is both given
to a model and scored on.

`n_train` cannot reach the DC1 training file's 43,486: the selection keeps 8.66
per cent of a representative sample, so all 434,476 galaxies yield at most
37,626, and that with nothing left to test on. 35,000 is the largest round
number that still leaves a usable hold-out.

`control=True` adds `split.unbiased`, exactly as many galaxies drawn at random
from the same pool — the control that separates the selection from the sample
size. It is drawn last, so asking for it changes nothing else in the split.

The selection is a port of RAIL's HSC `GridSelection`: each galaxy is kept with
the HSC spectroscopic success rate of its (i, g−z) pixel, below a
colour-dependent redshift ceiling. It is reproduced galaxy-for-galaxy against
the runs this work reports. It also acts on any photometry, so a catalogue of
your own can be biased the same way:

```python
from lazy.selection import grid_selection, selection_summary

keep, diagnostics = grid_selection(catalog.raw, catalog.redshift)
print(selection_summary(keep, catalog.raw["I"], catalog.redshift))
#  i in [16.0, 20.0)  ...  fraction 0.661
#  i in [24.0, 25.3)  ...  fraction 0.0005
```

`diagnostics` carries each galaxy's pixel success rate and redshift ceiling —
everything the selection knew, for a method that tries to estimate it back.
