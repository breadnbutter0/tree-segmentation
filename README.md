# tree-segmentation
SAM tree segmentation using BIGTIFF file


```bash
conda create -n treeseg python=3.9 
conda activate treeseg
```

```bash
conda install segment-anything gdal pyproj scipy tqdm pandas matplotlib
```

```bash
python segmentation.py bigtiff.tif
```