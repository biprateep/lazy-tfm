# Using scikit-learn

Every model in LAZY is a scikit-learn regressor. Its tags declare that it
accepts `NaN` features and requires a target. Four methods differ from those
of a usual regressor:

| Method          | A usual scikit-learn estimator      | A LAZY model                                                                                       |
| --------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `fit`           | Trains the model.                   | Stores the context and trains no weights (see [the FAQ](faq.md#what-does-fit-do)).                             |
| `predict`       | The prediction.                     | The mode of each row's density by default; `method=` picks another point estimate.                 |
| `predict_proba` | A classifier's class probabilities. | Densities on a grid, shape `(n_samples, n_bins)`.                                                  |
| `score`         | R² for a regressor.                 | The negative CDE loss, so higher is better.                                                        |

{doc}`distributions` lists every method and what it returns.

## Parameters

As in scikit-learn, the constructor only stores its arguments. `fit`
validates them and loads the checkpoint; TabFM loads its checkpoint at the
first prediction instead. `get_params`, `set_params` and
{func}`sklearn.base.clone` behave as scikit-learn's do, and a clone of a
fitted model is unfitted. `LazyModel` flattens the parameters of its backend
into its own, so `LazyModel("tabpfn").set_params(n_estimators=4)` and
`GridSearchCV(model, {"n_estimators": [4, 8]})` need no prefix. A pool
names the parameters of its members `<name>__<parameter>` (see
{doc}`ensembles`).

## scikit-learn tools

The table lists what each tool does with a LAZY model.

| Tool | Result | Notes |
| ---- | ------ | ----- |
| {func}`~sklearn.base.clone` | Works | Copies the parameters; the copy is unfitted. |
| {class}`~sklearn.pipeline.Pipeline` | Works, with a caveat | `fit`, `predict`, `predict_proba` and `score` pass through. A pipeline has no `predict_pdf`, `predict_quantiles`, `predict_interval`, `predict_pit`, `predict_cdf`, `predict_distribution` or `evaluate`. Call them on the last step with the transformed features: `pipe[-1].predict_quantiles(pipe[:-1].transform(X), q)`. |
| {class}`~sklearn.compose.ColumnTransformer` | Works | As a pipeline step before the model. |
| {class}`~sklearn.model_selection.GridSearchCV`, {func}`~sklearn.model_selection.cross_val_score`, default scoring | Works | They maximize `score`, the negative CDE loss, so they prefer the better densities. Each candidate is scored on its default grid; a `y_grid` in the constructor scores every candidate on one grid. |
| The same, with `scoring=` | Works | A point scorer, such as `"neg_mean_absolute_error"`, scores `predict`, which is the mode. |
| The same, with `n_jobs` | Works | Each job runs in a process of its own (see [Parallel jobs](scaling.md#parallel-jobs)). |
| {func}`~sklearn.model_selection.cross_val_predict` | Works, with a caveat | `method="predict"` returns the modes. `method="predict_proba"` does not work: scikit-learn treats it as a classifier's method and raises `AttributeError`. Loop over the folds instead. |
| {class}`~sklearn.compose.TransformedTargetRegressor` | Works, with a caveat | Only `predict` is transformed back. It has no `predict_proba`. |
| {class}`~sklearn.ensemble.StackingRegressor`, {class}`~sklearn.ensemble.VotingRegressor`, {class}`~sklearn.ensemble.BaggingRegressor` | Works, with a caveat | They combine `predict`, the modes, not the densities. {doc}`ensembles` pools densities. |
| {func}`~sklearn.inspection.partial_dependence` | Works | On `predict`. |
| {func}`~sklearn.utils.estimator_checks.check_estimator` | Fails two checks on purpose | `check_regressors_no_decision_function` fails because `predict_proba` returns densities. `check_dtype_object` fails because a non-numeric cell raises `ValueError`, not `TypeError`. Pass both in `expected_failed_checks`. LAZY's tests run the whole suite on models that use LAZY's shared code with a small stand-in for the network. The checks fit the model, so a real backend loads its checkpoint. |
| {mod}`pickle`, `joblib` | Works | See {doc}`saving`. |
