# tree-segmentation
SAM tree segmentation using BIGTIFF file


```bash
conda create -n treeseg python=3.9 
conda activate treeseg
```

```bash
pip install torch torchvision segment-anything opencv-python
```


```bash
conda install gdal pyproj scipy tqdm pandas matplotlib
```

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

```bash
python segmentation.py bigtiff.tif
```

