# CoralMTL

CoralMTL is a multi-task model that reads coral growth-form composition and
habitat structural complexity out of one shared representation.

> Boheng Li, Guanyan Keelung Chen, Jiaqi Wang, Joshua Madin,
> Nina M. D. Schiettekatte, Tung-Yung Fan, Crystal J. McRae, Katsunori Mizuno.
> *Coral growth-form composition and habitat structural complexity from one
> shared representation: multi-task learning on a single RGB reef orthomosaic.*

From a single 512 x 512 RGB orthomosaic patch, one shared UNet++ encoder feeds
three decoders and yields three spatially aligned outputs:

- an eight-class growth-form segmentation map — seven live-coral growth forms
  (branching, corymbose, digitate, massive, encrusting, foliose, submassive)
  plus one non-coral background class;
- rugosity (RU), derived analytically from the predicted surface normals as
  the mean of `1 / n_z`;
- fractal dimension (FD), obtained differentiably from the ordinary
  least-squares slope `S` of a four-point log height-variation curve at pixel
  scales {64, 128, 256, 512}, as `D = 3 - S`.

Co-registered digital elevation models (DEMs) supply the structural training
labels. No DEM is required at inference.

Two geometric mechanisms couple the branches. Geometric Discontinuity
Supervision (GDS) supervises an edge head on the segmentation decoder with the
Sobel response of the DEM-derived vertical normal component, so the boundary
signal costs no manual annotation. Geometry-Guided Attention (GGA) modulates
the complexity decoder with a spatial attention map computed from the geometry
features, with the gradient cut at the attention input so the complexity branch
cannot rewrite the geometry decoder.

## Contents

| Path | Contents |
| --- | --- |
| `coralmtl/model.py` | the shared encoder and the three decoders, with the GDS edge head, the GGA gate, and MultiScaleHead |
| `coralmtl/losses.py` | cross-entropy and edge losses for segmentation, cosine normal loss and pixel-wise rugosity loss for geometry, per-scale curve regression and differentiable FD slope loss for complexity |
| `coralmtl/constants.py` | the four pixel scales, the 0.5 mm pixel pitch, the label column names, seeding, and patch-level R-squared |
| `coralmtl/data.py` | the dataset: patch pairing, resizing, augmentation, DEM-derived normals, and label scaling |
| `coralmtl/engine.py` | one training or validation epoch, with the weighted loss sum and the patch-level aggregation of RU and FD |
| `train.py` | the training entry point: paths, hyperparameters, augmentation, optimiser and schedule, and checkpoint selection |
| `data/` | empty folders for the patches the training script reads; the layout and the label columns are described in `data/README.md` |

`train.py` reads from `data/` and writes checkpoints to `output/`, both
resolved relative to the repository root. The expected directory layout and
the label columns are described in `data/README.md`.

## Requirements

See `requirements.txt`; versions are not pinned, so install `torch` to match
your CUDA setup. `coralmtl/model.py`, `losses.py`, and `constants.py` need
only `torch`, `torchvision`, `segmentation-models-pytorch`, and `numpy`; the
dataset and the training script additionally need `albumentations`,
`opencv-python`, `pandas`, `tifffile`, and `tqdm`.

## Citation

```bibtex
@misc{li_coralmtl,
  author = {Li, Boheng and Chen, Guanyan Keelung and Wang, Jiaqi and
            Madin, Joshua and Schiettekatte, Nina M. D. and Fan, Tung-Yung and
            McRae, Crystal J. and Mizuno, Katsunori},
  title  = {Coral growth-form composition and habitat structural complexity
            from one shared representation: multi-task learning on a single
            RGB reef orthomosaic}
}
```

## License

All rights reserved. The code is published here to document the method; it is
not released under an open-source licence. For academic use beyond reading,
please contact the corresponding author.
