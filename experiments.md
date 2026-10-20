# Experiments

The experiments behind the two scaling curves (representative and biased),
and where each stands. `results/ledger.md`, rebuilt from the files on every
host by `scripts/ledger.sh`, is the record of their state; this page is the
plan.

Every experiment is one method at one training size of one regime:

- **Search.** Five-fold cross-validation on the training subset of seed 0,
  choosing the method's settings and features by the CDE loss on the common
  grid (4,000 bins of 0.0005 on [0, 2]); the test set is never used.
- **Production.** The chosen setting refitted on the subsets of seeds 0 to 4
  and scored on the test set (390,990 galaxies representative, 19,383
  biased).

Sizes: representative 1,000, 5,000, 10,000, 15,000, 20,000, 30,000 and
43,486; biased 1,000, 5,000, 10,000, 15,000, 20,000, 30,000 and 35,011.
Six methods, two regimes and seven sizes make 84 experiments of a search and
five production runs each.

## Classical baselines

| Method | Search round | Trials per search | Host |
| --- | --- | --- | --- |
| CMNN | 1 | 300 | done |
| GPz | 2 | 100 (production: best of 5 restarts) | tara |
| FlexZBoost, representative | 2 | 200 | dhruba |
| FlexZBoost, biased 1k-15k | 2 | 100 | tara |
| FlexZBoost, biased 20k-35k | 2 | 100 | dhruba |

## Foundation models (round 2)

Round 1 completed only 6 to 23 trials per search, and its LimiX bags were a
fraction of the training set, so the bags of a production fit were larger
than the bags its search validated (21.7k rows against 17.4k at the full
representative size, past the 20,000 rows LimiX-2 was pretrained on). Round
2 fixes both:

1. LimiX-2 bags are a number of rows (10,000, 15,000 or 20,000), the same in
   a search and in production.
2. Every search reaches at least 40 completed trials, starting from the five
   best settings of round 1 at that size, the round-1 best at every other
   size, and the round-2 best of the next smaller size.
3. A production run is redone only for the seeds whose record was made from
   another setting than the search's best (the ledger counts them as stale).

| Method | Representative | Biased |
| --- | --- | --- |
| TabPFN-3.5 | dhruba | dhruba |
| Kumo Tabular | dhruba | tara |
| LimiX-2 | tara | tara |

LimiX-2 runs on tara because its production fits at 30,000 rows and more
were killed for lack of memory on dhruba while the FlexZBoost search held it.
