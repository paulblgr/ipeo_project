'''Merges small patches into a single raster file
Example: 
python merge_rasters.py -i /media/data/gaston/glaciers_mapping/test/gt -o /media/data/gaston/glaciers_mapping/test/gt_merged.tif
'''
import argparse
import click 
import rasterio as rio
from rasterio.merge import merge
from glob import glob
import os
from icecream import ic
from os.path import join, expanduser, exists, dirname, basename
from os import listdir
import numpy as np

from rasterio import logging
from tqdm import tqdm


log = logging.getLogger()
log.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='Path to the folder containing all the patches', type=str) 
parser.add_argument('--output', '-o', type=str, help='Path to the output rater') 
args = parser.parse_args()


def merge_rasters():
    out_file = args.output
    files = glob(join(args.input, '*.tif'))
    rasters = [rio.open(i, 'r') for i in files]
    merged_raster, out_transform = merge(rasters)
    meta = rasters[0].meta
    
    meta.update(transform=out_transform, height=merged_raster.shape[1], width=merged_raster.shape[2], compress="LZW")
    with rio.open(out_file, 'w', **meta) as f:
        for i in range(meta['count']):
            f.write(merged_raster[i], indexes=i+1)
if __name__ == "__main__":
    merge_rasters()