# Reproducing the paper

This directory will hold the production scripts behind the paper: the exact
runs, in the exact order, that produce every number and figure in it.

It is empty on purpose. The research direction is not finalised, and freezing
half-finished experiment drivers here would give a false promise of
reproducibility.

Until then, the exploratory code — the benchmark harness, the Cal-PIT
recalibration, the FlexCode/FlexZBoost baselines, the biased-selection study
and the frozen splits — lives in a separate private archive alongside the
research log that records what was tried and what it showed. It deliberately
does not follow this package's API: it is a working record, not a library.
None of it is part of this repository's history, which starts clean.

## What lands here, and when

When the results are frozen, the scripts that produced them are ported to the
public API and committed here, alongside:

- the exact `uv.lock` used for the published runs;
- the pinned checkpoint revisions (see `lazy.models.CHECKPOINTS`);
- a `Makefile` or shell driver that runs them end to end;
- the commit hash and DOI of the archived release.

At that point this file is replaced by instructions that actually work.
