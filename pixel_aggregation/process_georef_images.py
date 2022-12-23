import geopandas as gpd
import shapefile
import os
from shapely.geometry import Polygon
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.measure import grid_points_in_poly
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Count pixels for each semantic class and for each census tract')
    parser.add_argument('--image-folder', required=True, help='Folder path to RGB images')
    parser.add_argument('--mask-folder', required=True, help='Folder path to mask images')
    parser.add_argument('--ct-shapefile', required=True, help='File path to the census tract shapefile')
    parser.add_argument('--split-image-folder', required=True, help='Folder path to RGB images')
    parser.add_argument('--split-mask-folder', required=True, help='Folder path to mask images')
    parser.add_argument('--num_splits', type=int, default=15, help='Number of splits to divide each image')
    
    args = parser.parse_args()
    return args


def image_pixels_to_world_coords(x, y, XCellSize, YCellSize, WorldX, WorldY):
    x1 = XCellSize * x + WorldX
    y1 = YCellSize * y + WorldY
    return x1, y1


def world_coords_to_image_pixels(x1, y1, XCellSize, YCellSize, WorldX, WorldY):
    image_x = int(np.round((x1 - WorldX) / XCellSize))
    image_y = int(np.round((y1 - WorldY) / YCellSize))
    return image_x, image_y



def mask_image_with_shapefile(image_folder, ct_shapefile, mask_folder):
    df = gpd.read_file(ct_shapefile)
    df = df.to_crs(epsg=3857)
   
    file_paths = os.listdir(image_folder)
    
    for ind in tqdm(df.index):
        geometry = df["geometry"][ind]

        already_seen_file_names = set()
        ct_already_seen_world_coords = set()

        for file_name in file_paths:
            file_base_name = file_name[:-4]

            if file_base_name in already_seen_file_names:
                continue
            already_seen_file_names.add(file_base_name)

            image_file = f"{file_base_name}.jpg"
            coord_file = f"{file_base_name}.jgw"

            with open(f"{image_folder}/{coord_file}",'r') as f:
                XCellSize = float(f.readline()) # A
                dont_care = f.readline() # should be 0, D
                dont_care = f.readline() # should be 0, B
                YCellSize = float(f.readline()) # E
                WorldX = float(f.readline()) # C
                WorldY = float(f.readline()) # F

            image = Image.open(f"{image_folder}/{image_file}")
            width, height = image.size
            
            image_corner_coords = [[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]

            image_corner_world_coords = []
            for x, y in image_corner_coords:
                world_x, world_y = image_pixels_to_world_coords(x, y, XCellSize, YCellSize, WorldX, WorldY)
                image_corner_world_coords.append([world_x, world_y])

            image_polygon = Polygon(image_corner_world_coords)
            image_df = gpd.GeoDataFrame(crs = 'epsg:3857', geometry=[image_polygon])

            image_polygon = image_df['geometry'][0]

            if geometry.intersects(image_polygon):
                ct_name = df["NAME20"][ind]

                image_mask = create_mask(geometry, image_polygon, image, f"{image_folder}/{coord_file}", ct_already_seen_world_coords, ct_name)
                image_mask = image_mask.transpose()
                mask_image = Image.fromarray(image_mask)
                mask_image = mask_image.convert("RGB")
                mask_image.save(f"{mask_folder}/{file_base_name}_{ct_name}.png")


def convert_image_to_world(x, y, coord_path):
    with open(coord_path,'r') as f:
        XCellSize = float(f.readline()) # A
        dont_care = f.readline() # should be 0, D
        dont_care = f.readline() # should be 0, B
        YCellSize = float(f.readline()) # E
        WorldX = float(f.readline()) # C
        WorldY = float(f.readline()) # F

    world_x, world_y = image_pixels_to_world_coords(x, y, XCellSize, YCellSize, WorldX, WorldY)

    return world_x, world_y
    


def create_mask(geometry, image_polygon, image, coord_path, ct_already_seen_world_coords, ct_name):
    width, height = image.size
    mask_array = np.zeros((width, height))

    all_polygon_x_points = []
    all_polygon_y_points = []
    intersection_polygon = geometry.intersection(image_polygon)

    if intersection_polygon.geom_type == "MultiPolygon":
        polygons = list(intersection_polygon)
        for polygon in polygons:
            polygon_x_points, polygon_y_points = polygon.exterior.coords.xy
            polygon_x_points = list(polygon_x_points)
            polygon_y_points = list(polygon_y_points)
            all_polygon_x_points.extend(polygon_x_points)
            all_polygon_y_points.extend(polygon_y_points)
    else:
        all_polygon_x_points, all_polygon_y_points = intersection_polygon.exterior.coords.xy


    with open(coord_path,'r') as f:
        XCellSize = float(f.readline()) # A
        dont_care = f.readline() # should be 0, D
        dont_care = f.readline() # should be 0, B
        YCellSize = float(f.readline()) # E
        WorldX = float(f.readline()) # C
        WorldY = float(f.readline()) # F

    intersection_polygon_image_coords = []
    for x_world_coord, y_world_coord in zip(all_polygon_x_points, all_polygon_y_points):
        image_coords = world_coords_to_image_pixels(x_world_coord, y_world_coord, XCellSize, YCellSize, WorldX, WorldY)
        intersection_polygon_image_coords.append(image_coords)
    
    intersection_polygon_image_coords = np.asarray(intersection_polygon_image_coords)
    mask_array = grid_points_in_poly((width, height), intersection_polygon_image_coords)

    for mask_x in range(width):
        for mask_y in range(height):
            if mask_array[mask_x, mask_y]:
                world_x, world_y = image_pixels_to_world_coords(mask_x, mask_y, XCellSize, YCellSize, WorldX, WorldY)
                if (world_x, world_y) in ct_already_seen_world_coords:
                    mask_array[mask_x, mask_y] = False

                    ct_already_seen_world_coords.add((world_x, world_y))

    return mask_array


def split_images_and_masks(image_folder, mask_folder, split_image_folder, split_mask_folder, num_splits=12):
    image_paths = os.listdir(image_folder)
    mask_paths = os.listdir(mask_folder)
    for mask_path in tqdm(mask_paths):
        file_base_name = mask_path[:-4]
        mask = Image.open(f"{mask_folder}/{file_base_name}.png")

        width, height = mask.size
        split_width = width // num_splits
        split_height = height // num_splits

        mask = np.array(mask)
        shape = mask.shape
        
        counter = 0
        for x_split in range(num_splits-1):
            for y_split in range(num_splits-1):
                x_right_split = (x_split+1) * split_width
                if x_split == num_splits-2:
                    x_right_split = width

                y_right_split = (y_split+1) * split_height
                if y_split == num_splits-1:
                    y_right_split = height
                cropped_mask = mask[y_split * split_height : y_right_split, x_split * split_width : x_right_split]

                cropped_mask = Image.fromarray(cropped_mask)
                cropped_mask.save(f"{split_mask_folder}/{file_base_name}_{counter}.png")

                counter += 1

    for image_path in tqdm(image_paths):
        file_base_name = image_path[:-4]

        image = Image.open(f"{image_folder}/{file_base_name}.jpg")

        width, height = image.size
        split_width = width // num_splits
        split_height = height // num_splits

        image = np.array(image)
        img_shape = image.shape

        counter = 0
        for x_split in range(num_splits-1):
            for y_split in range(num_splits-1):
                x_right_split = (x_split+1) * split_width
                if x_split == num_splits-2:
                    x_right_split = width

                y_right_split = (y_split+1) * split_height
                if y_split == num_splits-1:
                    y_right_split = height

                cropped_image = image[y_split * split_height : y_right_split, x_split * split_width : x_right_split]

                cropped_image = Image.fromarray(cropped_image)
                cropped_image.save(f"{split_image_folder}/{file_base_name}_{counter}.jpg")

                counter += 1


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.split_image_folder):
        os.mkdir(args.split_image_folder)
    if not os.path.exists(args.split_mask_folder):
        os.mkdir(args.split_mask_folder)

    mask_image_with_shapefile(args.image_folder, args.ct_shapefile, args.mask_folder)

    split_images_and_masks(args.image_folder, args.mask_folder, args.split_image_folder, args.split_mask_folder, args.num_splits)
