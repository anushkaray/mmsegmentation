import geopandas as gpd
import shapefile
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
import argparse


ade20k_color_to_pred_class = {
    (180, 120, 120) : 'building',
    (4, 200, 3): 'tree',
    (4, 250, 7) : 'grass',
    (235, 255, 7) : 'sidewalk',
    (120, 120, 70) : 'earth',
    (61, 230, 250) : 'water',
    # (0, 41, 255) : 'clutter'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Count pixels for each semantic class and for each census tract')
    parser.add_argument('--ct-shapefile', required=True, help='File path to the census tract shapefile')
    parser.add_argument('--mask-folder',required=True, help='Folder path to mask images')
    parser.add_argument('--segmentation-folder',required=True, help='Folder path to segmentation images')
    parser.add_argument('--csv-save-path',required=True, help='File path to save output CSV')
    
    args = parser.parse_args()
    return args


def get_euclidean_distance(rgb1, rgb2):
    return ((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2) ** (1/2)


def get_closest(rgb, threshold):
    distances = []
    for key in ade20k_color_to_pred_class:
        distances.append((get_euclidean_distance(rgb, key), key))
    distances.sort(key=lambda y: y[0])
    closest_dist, closest_rgb = distances[0]
    if closest_dist < threshold:
        return closest_rgb
    return None


def find_pixels(ct_name, ct_pixel_count, full_mask_path, full_seg_path, seg_base_name, csv_path):
    segmentation = Image.open(full_seg_path)
    loaded_seg = segmentation.load()
    mask = Image.open(full_mask_path)
    all_rgbs = set()

    segmentation_np = np.array(segmentation)
    mask_np = np.array(mask)
    
    mask_np = mask_np == 255
    pixels_in_ct = segmentation_np * mask_np

    width, height, channels = pixels_in_ct.shape
    
    for x in range(width):
        for y in range(height):
            rgb = tuple(pixels_in_ct[x, y])
            all_rgbs.add(rgb)

            closest_rgb = get_closest(rgb, threshold=45)
            if closest_rgb is not None and closest_rgb in ade20k_color_to_pred_class:
                ft_name = ade20k_color_to_pred_class[closest_rgb]
            else:
                ft_name = 'clutter'

            ct_pixel_count[ct_name][ft_name] += 1
            ct_pixel_count[ct_name]['total'] += 1
            
    ct_pixel_count_df = pd.DataFrame(ct_pixel_count).transpose()
    ct_pixel_count_df.to_csv(csv_path)

    
def count_features_in_all_segmentations(ct_name_list, segmentation_folder, mask_folder, csv_path):
    seg_paths = os.listdir(segmentation_folder)
    mask_paths = os.listdir(mask_folder)
    ct_pixel_count = dict()

    for ct_name in tqdm(ct_name_list):
        ct_pixel_count[ct_name] = {'building' : 0,
                                   'tree': 0,
                                   'grass' : 0,
                                   'sidewalk' : 0,
                                   'earth' : 0,
                                   'water' : 0,
                                   'clutter' : 0,
                                   'total' : 0}
        
        for seg_name in tqdm(seg_paths):
            seg_base_name, count = seg_name[:-4].rsplit('_', 1)

            mask_name = f"{seg_base_name}_{ct_name}_{count}.png"
            full_mask_path = f"{mask_folder}/{mask_name}"
            full_seg_path = f"{segmentation_folder}/{seg_name}"

            if mask_name in mask_paths:
                find_pixels(ct_name, ct_pixel_count, full_mask_path, full_seg_path, seg_base_name, csv_path)


if __name__ == '__main__':
    args = parse_args()
    census_tract_shapefile = args.ct_shapefile
    census_tract_df = gpd.read_file(census_tract_shapefile)
    ct_name_list = census_tract_df["NAME20"].values.tolist()
    
    count_features_in_all_segmentations(ct_name_list, args.segmentation_folder, args.mask_folder, args.csv_save_path)
