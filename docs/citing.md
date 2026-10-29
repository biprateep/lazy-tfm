# Citing

If LAZY is useful in your work, we ask that you cite the paper that describes
it and the software itself. The paper is Dey & Speagle, "Tabular Foundation
Models Dramatically Outperform Bespoke Photometric Redshift Estimators",
presented at the 9th Workshop on Machine Learning and the Physical Sciences
(ML4PS 2026); we will add its arXiv link and BibTeX entry here once the
preprint is posted. The software is archived on Zenodo, where the concept DOI
[10.5281/zenodo.23751881](https://doi.org/10.5281/zenodo.23751881) refers to
every version of lazy-tfm and resolves to the latest release, while each
release also has its own DOI on its Zenodo record for when you need to cite
the exact version you used. The BibTeX entry below is the one Zenodo provides
for the concept DOI, and the "Cite this repository" button on GitHub gives
the same metadata from the repository's `CITATION.cff`.

```bibtex
@misc{https://doi.org/10.5281/zenodo.23751881,
  doi = {10.5281/ZENODO.23751881},
  url = {https://zenodo.org/doi/10.5281/zenodo.23751881},
  author = {Dey, Biprateep and Speagle, Joshua Shen},
  keywords = {photometric redshift, astronomy, foundation model, tabular, LSST},
  title = {LAZY: Lazy but Accurate z for Yinz},
  publisher = {Zenodo},
  year = {2026},
  copyright = {MIT License}
}
```

We also ask that you cite the works describing the tabular foundation model
you used as a backend, since LAZY builds on their pretrained weights. The
`provenance_` attribute of a fitted model records exactly which weights, and
which versions of the code, produced your results, which tells you which of
these works to cite.

## Licenses

The package is MIT-licensed. The pretrained checkpoints carry their own
licenses, several of which are non-commercial, and
{data}`lazy.CHECKPOINTS <lazy.models.CHECKPOINTS>` quotes each one. Therefore,
please check the license of the checkpoint you are using before applying it
to your own work.
