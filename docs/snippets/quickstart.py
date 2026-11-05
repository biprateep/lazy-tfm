"""The landing page's quick start; make_figures.py runs it for the figure."""

import lazy
from lazy import datasets

X_train, X_test, y_train, y_test = datasets.load_dataset(
    "chirp", split=True, return_X_y=True, random_state=0
)

# v3.5-fast is not the default (v3.5); it runs on a CPU.
model = lazy.LazyModel("tabpfn", version="v3.5-fast")
model.fit(X_train, y_train)

y_mean = model.predict(X_test, method="mean")
y_low, y_high = model.predict_interval(X_test, coverage=0.68).T
print(y_mean[:3], y_low[:3], y_high[:3], sep="\n")
