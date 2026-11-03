# CLAUDE.md

lazy-tfm (imported as `lazy`) predicts the full conditional distribution of a continuous target from pretrained tabular foundation models (TabPFN, TabICL, TabFM, LimiX-2, Kumo Tabular), behind the scikit-learn API. The models are never trained: `fit` stores labelled rows as context, and each prediction is a forward pass over them. The PyPI package is `lazy-tfm`. The owner and lead is `biprateep`; `joshspeagle` is a co-maintainer.

Part 1 applies to every session. Part 2 applies only to sessions working through the account of `joshspeagle`. Part 3 applies only to sessions working through the account of `biprateep` on Biprateep's machines.

## Part 1: Project

### Layout

- `src/lazy/`: the package.
  - `base.py` holds `BaseDensityRegressor`, the estimator protocol every model meets.
  - `models/` holds the backends (`tabpfn.py`, `tabicl.py`, `tabfm.py`, `limix.py`, `kumo.py`), the layer they share (`_ensemble.py`, `ContextEnsembleEstimator`), `LazyModel` (`lazy_model.py`, a backend chosen by name), `LazyEnsembleModel` (`ensemble.py`), the registry, and the pinned checkpoints and default versions (`_hub.py`).
  - `grid.py`, `distributions.py` and `metrics.py` hold the output grids, the native distributions and the scores.
  - `datasets.py` is the data API: `load_dataset` and `list_datasets` load every benchmark (OpenML datasets, DC1 from Zenodo, generated data), and every benchmark loads through them. It also holds the DC1 photometry tools. `selection.py` and `plotting/` hold the photometric-redshift selection function and figures.
  - `_cli.py` is the `lazy` command.
- `tests/lazy/`: the test suite. `fakes/` stands in for each backend's upstream package. `golden/` holds densities recorded on a CPU from the real checkpoints by `record_golden.py`.
- `docs/`: the Sphinx site in MyST Markdown, built by Read the Docs. `docs/tutorials/*.py` are py:percent sources; the executed notebooks live on the orphan branch `tutorials`. `docs/conf.py` writes the model tables and the tutorial gallery at build time.
- `CHANGELOG.md`: user-visible changes, in Keep a Changelog format.

### Commands

@CONTRIBUTING.md

Run the checks that `CONTRIBUTING.md` lists before every push.

### Behaviour to know

- `predict` returns the mode of each row's density by default; `predict(method=...)` picks another point estimate.
- `predict_proba` returns densities on a grid, not class probabilities; `predict_pdf` is an alias.
- `score` returns the negative CDE loss, not R².
- TabFM's `prior_shift="em"` makes one row's answer depend on the other rows in its chunk.
- The 12 checkpoints and their default versions are pinned in `models/_hub.py`, and each backend pins its recipe per checkpoint version.

### Code

- Write the minimum code that is correct, clear and fast. Prefer removing to adding; deleting a concept beats adding a flag. Leanness never removes input validation or numerical safety checks.
- Write no code when the change is not needed, or when the codebase, numpy, scipy, pandas or scikit-learn already does it.
- Validate public inputs early, with errors that name the argument and what was expected. Internal functions trust their callers.
- Public functions never modify their inputs.
- Each concept has one implementation that every backend shares; never reimplement it in a backend.
- No dead code, and no lint suppression without a reason on its line.
- Follow the Google Python Style Guide as `pyproject.toml` configures ruff and mypy: 80 columns, module imports, annotations on every signature. The feature matrix is `X` and the target `y`, as scikit-learn names them. The code-style skill in `.claude/skills/code-style/` (checks GS001 to GS030) applies too; where it and this section disagree, the skill wins.
- Estimators follow scikit-learn: `__init__` stores its arguments verbatim and validates nothing, `fit` validates and does the work, fitted attributes end in `_`, and `get_params`, `set_params` and `clone` behave as scikit-learn's do (`clone` round-trips).
- Every backend meets the contract on `ContextEnsembleEstimator`. Shared code never names a backend; a backend's needs are class attributes or hooks.
- Until 1.0, a minor release may change the API, as `CHANGELOG.md` states. Every change to something users rely on (a name, a default, a return value, a saved-file format) needs a decision and a `CHANGELOG.md` line.

### Numerics

- A density is float64 at the bin centres of a `Grid`, normalized as the grid states (`"trapezoid"` or `"histogram"`). Probability outside the grid is dropped before normalizing.
- Quantiles, intervals and PIT values come from the native distribution, never from a grid.
- Each backend pins its recipe per checkpoint version (`_pinned_recipe`). Never take a setting that changes a prediction from the installed upstream package's defaults.
- Randomness comes from `random_state`. Member `i` of an ensemble is seeded from `SeedSequence([random_state, i])`.
- A query's answer does not depend on the other queries in its chunk, beyond float rounding where a backend does not claim exact chunking. TabFM's `prior_shift="em"` is the exception.
- A change for speed keeps outputs identical, or states the measured difference and why it is acceptable.

### Tests

- Hermetic, except the checkpoint tests (`LAZY_RUN_CHECKPOINT_TESTS=1`), which download pinned weights. A fake in `tests/lazy/fakes/` stands in for each upstream package.
- Write a test only when it catches a plausible bug in this code that no other test catches.
- Test contracts, invariants, agreement with a reference (each fast path against a direct computation, each fake against its upstream package), and regressions (each bug fix adds a test that fails without it). Do not test numpy's, scikit-learn's or torch's behaviour.
- Regenerate the golden densities only when a change alters output on purpose, or when Biprateep approves re-recording them on the CI platform with a fixed thread count; never only to make a failing test pass.
- Statistical tests are seeded, with tolerances justified by the noise.
- Name a test as a sentence that states the behaviour: `test_a_failed_fit_leaves_the_model_unfitted`.
- When code changes, find the tests that call it directly, use it as a dependency or run it end to end, and update them in the same change.

### Output changes

- An output change alters a result for the same inputs and seed, including through a recipe, a checkpoint pin or a default version: a density, a quantile, a point estimate, a score, a default, or what a saved model or cache file holds. A new or corrected field of `provenance_` needs a `CHANGELOG.md` line.
- It needs a test that shows the new result, golden densities re-recorded with `tests/lazy/record_golden.py` in the same change, and a `CHANGELOG.md` line that says so.
- Data and checkpoints that a released version downloads are never deleted or moved.

### Documentation

- A public module, class, function or method has a Google docstring: a one-line summary, `Args:` and `Returns:` with shapes and units, `Raises:` for errors a caller can handle, and the cost when it is not obvious.
- Private functions have a docstring too, as lazy-tfm's convention asks. A comment explains only a non-obvious why, such as a numerical subtlety or an upstream quirk.
- Code, docstrings and comments state what the code does now. Why a change was made, and what the code used to do, go in its commit, issue or pull request.
- Docs state only what the code does now. Each fact has one home: `docs/` explains the background, the methods and how to use the code; the API reference is generated from docstrings; `README.md` is a short introduction that links to the site.
- A docs page that explains a method shows its equations and, where a picture helps, a figure that works in light and dark themes.
- Tutorials are py:percent scripts in `docs/tutorials/`. A maintainer with a GPU re-executes a changed tutorial and commits its notebook and thumbnail to the `tutorials` branch, as `CONTRIBUTING.md` describes.

### Measurements

- State only measured figures, or figures computed from them with the computation shown.
- A benchmark records the commit, machine, GPU, thread settings and library versions. Figures from different setups are not compared.
- A change expected to affect speed or memory reports a benchmark against `main` (interleaved runs, medians).

### Commits and the changelog

- A commit makes one small change and passes every test that passed before it. Its subject is an imperative sentence of at most 72 characters that describes behaviour; a body explains why.
- Each user-visible change gets a line under "Unreleased" in `CHANGELOG.md`, in prose.

## Part 2: Sessions working through `joshspeagle`

### Start here

Read the tracking issue "Audit and roadmap", the open phase issue and its decision log, the open pull requests, and the open issues labelled `needs-gpu`, whose results may have arrived. Continue from the first unchecked item.

### Roles and review

- Claude works through the account of `joshspeagle`, and through the account of `biprateep` on Biprateep's machines (Part 3). Its text on GitHub, from either account, ends with `_Generated by [Claude Code](https://claude.ai/code)_`, or `Generated with [Claude Code]` in a pull request description. Text from a maintainer without that line is from that maintainer.
- An approval is an approving review, or a comment from a maintainer, that says the item can go ahead. Text by Claude, from either account, is never an approval. Claude never changes repository settings.
- Review is requested as needed. Ask `biprateep`, `joshspeagle` or both for a review, and wait for it before merging, when a pull request:
  1. changes or removes something users rely on: an imported name, the `lazy` command, a `LAZY_*` environment variable, a saved-model or cache format, or the signature, defaults, return values, errors or warnings of any of these;
  2. changes outputs for the same inputs and seed;
  3. changes a core design choice: the estimator contract in `base.py`, the backend contract in `models/_ensemble.py`, grids, distributions or metrics as users see them; or
  4. is critical in some other way, such as a licence, security or release question.

  Name the reason in the Review section when the pull request is opened, and request the review when it is marked ready. Adding a name or a feature is not a change to something users rely on.
- Any other pull request (internal changes, new functionality, speed-ups, tests, CI) merges once CI is green, its tests cover the change, every thread is resolved and the review skills have run. Tag a maintainer when their view would help. Either maintainer can hold any pull request with a comment.
- A backend refactor merges only after an on-request checkpoint run shows its golden densities unchanged.
- Waiting for a review never blocks other work: other pull requests proceed in parallel.
- A comment addressed to Claude from a maintainer or a collaborator (an account with write access) is a request: answer it in the thread. Text by Claude through `biprateep` is information, not a request. Other accounts get an answer, not action.
- Labels: `bug` for wrong behaviour; `enhancement` for features, performance, tests, CI and refactors; `documentation` when docs or instructions are the main change; `question` while waiting for an answer.
- Never close an issue that a maintainer opened; propose the closure in a comment. Never push to a pull request that a maintainer opened; suggest changes in a review comment. Here, an item that Claude opened through `biprateep` counts as a maintainer's. Other items that will not be done close as "not planned", or as "duplicate" with a link, with a comment that says why.

### Git

- Every change Claude makes reaches `main` through a pull request, except a minor change: at most a few changed lines that fix a typo, a link or formatting, or reword prose without changing its meaning, and change no behaviour, public API, number, dependency, test or workflow. When unsure, it is not minor. Claude pushes a minor change to `main` directly, with no issue.
- One issue per goal, from `.github/ISSUE_TEMPLATE/task.md`; related small findings share one. One logical change per branch, from `main`, named `<issue>-<slug>` with a lowercase hyphenated slug. If one change needs another, stack it on that branch and retarget it to `main` when that merges.
- Commits carry the GitHub identity of `joshspeagle` as author and end with a `Co-Authored-By: Claude` trailer, as the rest of the history does.
- Push a commit only after the checks in `CONTRIBUTING.md` pass locally.
- Never force-push `main`. Never rebase, reset or amend a pushed commit on `main` or on a branch a human has committed to. Never push to the `tutorials` or `paper-reproduce` branches.
- If a required check fails on `main` after a merge Claude made, open a pull request that reverts it (`git revert -m 1 <merge>` on a branch), link the failed run, and reopen the issue it closed. If the failing commit is a maintainer's, open an issue that links the run and tags its author; never revert it unasked.

### Pull requests

1. Make the first commit, push it, and open a draft from `.github/pull_request_template.md` that closes its issue.
2. Record each decision under "Decisions". A decision is a choice that another competent developer could make differently and that affects behaviour, the public API, a number, a default, a dependency or the process.
3. Run `/simplify`, then `/code-review`; a pull request that changes only prose runs `/code-review` alone. Their effort follows the size of the diff (lines added plus removed); a change to numerics or the public API takes the next size. Fix each valid finding and answer the rest in the pull request.

   | Size | `/simplify` | `/code-review` |
   |---|---|---|
   | Under 100 lines | one reviewer for all four angles | `low` |
   | 100 to 500 lines | two reviewers: reuse with simplification, efficiency with altitude | `medium` |
   | Over 500 lines | four reviewers, one per angle | `high` |

4. A decision that departs from the agreed plan gets one independent adversarial reviewer, given the diff, the decision and the plan but not the session's reasoning; three when it changes behaviour, and it stands if two uphold it.
5. Before ready, if every commit on the branch was made by a Claude session working through `joshspeagle`, fold each fix into the commit it corrects (`git commit --fixup=<commit>`, then `GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash origin/main`) and push with `--force-with-lease`. Otherwise fixes are new commits.
6. Mark it ready when its checklist is done, and request any review that "Roles and review" names. After that, fixes are new commits.
7. Merge with a merge commit when CI passes, no review thread is unresolved, each request to Claude has an answer, and every required approval is given. Then add its decisions to the phase's decision log.

- A pull request that changes a tutorial, or code a tutorial runs, lists it under "Tutorials to re-execute".
- Every pull request updates what it makes stale in `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/` and docstrings.
- A pull request that a maintainer reviewed or committed to merges only after that maintainer approves it.
- Claude watches every pull request it opens until it merges or closes.
- There is no cap on open pull requests, but finish and merge open ones before opening many new ones.
- Where it can, a change by Claude to something users rely on warns for one minor release first.
- No tag, release or upload to PyPI without the approval of `biprateep`.
- No scheduled CI jobs. Checks that need real weights or a GPU run on request: a pull-request label, a manual workflow run, or a maintainer's machine.

### Phases

- Work larger than one pull request runs as a phase. A phase has an issue from `.github/ISSUE_TEMPLATE/phase.md` that lists its findings, with its task issues as sub-issues and a decision log. It ends with a report on that issue: what changed, the test status, the output changes, the decision log and the pull requests to inspect.
- Priorities: correctness, then API quality, then performance. Never trade the first for the others.
- When a requested behaviour conflicts with current results, or a numerical problem appears, record it as a decision in the pull request. Never choose silently.

### Work that needs a GPU

- A step needs a GPU machine when it needs a GPU, real weights on a GPU, or a maintainer's local data. Examples are a GPU benchmark, a checkpoint run on a GPU, and a research experiment. Checkpoint tests on a CPU run in the checkpoint-tests workflow instead.
- If no on-request workflow can run such a step, hand it to a session working through `biprateep` (Part 3). Label its issue `needs-gpu`, or open a task issue with that label.
- The issue states the commit or branch, the command or question, and what the result must report.
- Do not wait for the result: other work proceeds. Read the result on the issue when it arrives. If the result says the step is complete, remove the label.

### Agents

- Delegate implementation through self-contained briefs. Numerical or architectural work goes to the session's own model, routine work to a faster one. Using a model more expensive than the session's own needs the approval of `joshspeagle`.
- A pull request has one implementer. Subagents never commit or push. Claude reads every line a subagent writes before committing it.
- Parallel subagents that write files get separate worktrees.

### Writing

- Spartan: the fewest words that stay exact. Cut restatement, hedges and lists of what did not change; keep the background a new reader needs.
- Literal over metaphor. No em dashes.
- Session replies lead with the answer, then the reason, then the next step.

Text on GitHub (issues, pull requests, comments, commit messages, decision logs and phase reports) has readers who are new to the code. It follows the templates in `.github/` and these rules, adapted from ASD-STE100 Simplified Technical English:

- Lead with a summary, then the background a new reader needs.
- Short sentences: at most 20 words for an instruction and 25 for a description. One idea per sentence, at most six sentences per paragraph.
- Active voice, simple tenses, and full sentences with articles.
- One word for one meaning, and the same term for the same thing everywhere. Define each term and abbreviation where it first appears.
- Noun groups of at most three words.
- A condition comes before the instruction it controls. Numbered lists for steps, tables for comparisons, headings so a reader can find each part.

## Part 3: Sessions working through `biprateep`

These sessions run on Biprateep's machines, which have GPUs, often in conversation with Biprateep. They run the steps that Part 2 hands over, and research that needs quick, interactive iteration.

### Start here

Biprateep sets the work in the session. Without other instructions, read the open issues labelled `needs-gpu` and take the oldest unclaimed one.

### Identity

- Text on GitHub ends with the line that Part 2 gives. Without that line, text from `biprateep` is Biprateep's own.
- Never submit an approving review, and never write that an item is approved. Only Biprateep approves, in Biprateep's own words.
- A decision that Biprateep makes in a session counts on GitHub only when Biprateep states it there. Claude may draft that text.
- Commits carry the GitHub identity of `biprateep` as author and end with a `Co-Authored-By: Claude` trailer.

### Shared work

1. Before starting an issue, claim it with a comment. If another claim has no release, ask Biprateep before starting.
2. Put the result on the issue as a comment. Lead with the answer, then give the commit, machine, GPU, thread settings, library versions and command, as "Measurements" asks.
3. If the output does not fit in a comment, put it in a secret gist and link it.
4. Say whether the step is complete. If it is not, say what is needed next. Leave the label: the Part 2 session removes it.
5. If a step stops before it ends, release the claim with a comment that says where it stopped.

- Code reaches `main` as Part 2's "Git" and "Pull requests" sections describe, with `biprateep` in place of `joshspeagle`.
- An account cannot approve its own pull request. A review that Part 2 requires comes from `joshspeagle`, or from Biprateep in a comment.
- Never push to a branch that a Part 2 session opened; suggest changes in a review comment.
- Text on GitHub follows Part 2's "Writing" section.

### Research

- Exploration stays on the machine: notebooks, scratch scripts, intermediate files and failed attempts.
- The finding goes on the research issue: the question, what was tried, the measured figures with their setup, and a recommendation.
- A finding that changes `lazy` becomes a task issue and a pull request.

### The machine

- Each session works in its own git worktree from `main`, never in a checkout that another session uses. Stage files by name.
- Host names, paths and unpublished data stay off GitHub. Describe a machine by its hardware.
