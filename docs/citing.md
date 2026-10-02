# Citing

A paper describing LAZY is in preparation. Until it appears, please cite the
repository (the "Cite this repository" button on GitHub uses its
`CITATION.cff`) together with the checkpoints used, since the weights of a
foundation model are part of the method. The `provenance_` attribute of a
fitted model records exactly which weights, and which versions of the code,
produced the results.

## Licenses

The package is MIT-licensed. The pretrained checkpoints carry their own
licenses, several of which are non-commercial, and
{data}`lazy.CHECKPOINTS <lazy.models.CHECKPOINTS>` quotes each one. The HSC
selection grid used by {mod}`lazy.selection` is redistributed by DESC's
[rail_astro_tools](https://github.com/LSSTDESC/rail_astro_tools) (MIT) and
derives from HSC PDR2 (Aihara et al. 2019). It is downloaded from there at a
pinned commit rather than bundled with the package.

The project was started from the
[LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template).
