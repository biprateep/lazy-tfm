# Citing

A paper is in preparation. Until it appears, cite the repository (GitHub's
"Cite this repository" button uses its `CITATION.cff`) and the checkpoints you
used: a foundation model's weights are part of the method. The `provenance_` of
a fitted model records exactly which weights, and which versions of the code,
produced your numbers.

## Licences

The package is MIT-licensed. The pretrained checkpoints carry their own
licences, several of them non-commercial; {data}`lazy.CHECKPOINTS` quotes each
one. The HSC selection grid used by {mod}`lazy.selection` is redistributed by
DESC's [rail_astro_tools](https://github.com/LSSTDESC/rail_astro_tools) (MIT)
and derives from HSC PDR2 (Aihara et al. 2019); it is downloaded from there at
a pinned commit rather than bundled.

The project was started from the
[LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template).
