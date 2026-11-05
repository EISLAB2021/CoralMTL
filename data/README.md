# Data

`train.py` reads from this folder and writes checkpoints to `../output/`.
Put your own patches in place as follows.

```
data/
├── train/
│   ├── images/        RGB patches: .png, .jpg, .tif or .tiff
│   ├── mask/          label maps: <same stem>.png
│   ├── dems/          elevation rasters: <same stem>.tif or .tiff
│   └── train_fd.csv   patch-level labels
└── val/
    ├── images/
    ├── mask/
    ├── dems/
    └── val_fd.csv
```

The three rasters of one patch are paired by file-name stem, so
`images/P001.png`, `mask/P001.png` and `dems/P001.tif` belong together. A
missing mask or DEM is silently replaced by zeros, so check your file names if
training appears to learn nothing.

Each image is resized to 512 x 512 on load. Mask pixels hold a class index in
0-7; the value 255 marks pixels excluded from the segmentation loss. DEM
rasters are single-band elevation in metres, which is what makes the surface
normals consistent with the 0.5 mm pixel pitch set in
`coralmtl/constants.py` — a DEM in other units will give wrong normals, and
therefore wrong rugosity.

The CSV is indexed by `Filename` (matched on its stem) and supplies the
patch-level structural labels:

| Column | Contents |
| --- | --- |
| `Filename` | the patch file name |
| `FD_Variation` | fractal dimension of the patch |
| `Log_H_64px`, `Log_H_128px`, `Log_H_256px`, `Log_H_512px` | log10 of the mean height variation at the four pixel scales |
| `R_Squared` | R-squared of the ordinary least-squares fit through those four points; patches below 0.9 are dropped from the per-scale regression term but still used for segmentation |
