import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='remove black masks')
    parser.add_argument('--mask-path', required=True, help='File path to the mask folder')
    parser.add_argument('--output-path',required=True, help='Output folder')
    args = parser.parse_args()
    return args


def remove_all_black_masks(mask_path, output_path):
    mask_paths = os.listdir(mask_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    removed = 0
    for cropped_mask_path in tqdm(mask_paths):
        cropped_mask = Image.open(f"{mask_path}/{cropped_mask_path}")
        cropped_mask_np = np.array(cropped_mask)
        cropped_mask_np_sum = np.sum(cropped_mask_np)

        if cropped_mask_np_sum != 0:
            cropped_mask_image = Image.fromarray(cropped_mask_np)
            cropped_mask_image = cropped_mask_image.convert("RGB")
            cropped_mask_image.save(f"{output_path}/{cropped_mask_path}")
        else:
            removed += 1
    print("Number of Removed Masks: ", removed)



if __name__ == '__main__':
    args = parse_args()
    mask_path = args.mask_path
    output_path = args.output_path
    remove_all_black_masks(mask_path, output_path)
