import csv
from tqdm import tqdm
import pandas as pd
import math
import copy
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve building features (e.g. building height, building age, use class, presence of air conditioning) from Boston Buildings Inventory')
    parser.add_argument('--building-inventory-path', required=True, help='File path to Boston Buildings Inventory CSV')
    parser.add_argument('--save-folder',required=True, help='Folder path to save CSV files with processed features')
    
    args = parser.parse_args()
    return args


def process_census_tract_name(row_census_tract):
    row_census_tract_str = str(int(row_census_tract))
    row_census_tract_fixed = str(int(row_census_tract_str[5:9])) + "." + row_census_tract_str[-2:]
    return row_census_tract_fixed


def aggregate_building_use_class_by_census_tract(building_inventory_df, census_tract_column_name, building_use_class_column_name):
    # Find unique census tracts and building use classes
    unique_census_tracts = set()
    unique_building_use_classes = set()
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        
        if (not math.isnan(row_census_tract)) and (row_census_tract not in unique_census_tracts):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            unique_census_tracts.add(row_census_tract_fixed)
        
        row_building_use_class = row[building_use_class_column_name]
        if (type(row_building_use_class) == str) and (row_building_use_class not in unique_building_use_classes):
            unique_building_use_classes.add(row_building_use_class)

    # Initialize dictionary
    use_class_dict = {}
    for use_class in unique_building_use_classes:
        use_class_dict[use_class] = 0
    use_class_dict["num_buildings"] = 0
    
    use_class_by_census_tract = {}
    for census_tract in unique_census_tracts:
        use_class_by_census_tract[census_tract] = copy.deepcopy(use_class_dict)

    # Aggregate use classes per census tract
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        row_building_use_class = row[building_use_class_column_name]

        if (not math.isnan(row_census_tract)) and (type(row_building_use_class) == str):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            use_class_by_census_tract[row_census_tract_fixed][row_building_use_class] += 1
            use_class_by_census_tract[row_census_tract_fixed]["num_buildings"] += 1

    # Convert counts to percentages
    for census_tract in use_class_by_census_tract:
        for use_class in unique_building_use_classes:
            use_class_by_census_tract[census_tract][use_class] = use_class_by_census_tract[census_tract][use_class] / use_class_by_census_tract[census_tract]["num_buildings"]

    return use_class_by_census_tract


def aggregate_air_conditioning_by_census_tract(building_inventory_df, census_tract_column_name, air_conditioning_column_name):
    # Find unique census tracts
    unique_census_tracts = set()
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        
        if (not math.isnan(row_census_tract)) and (row_census_tract not in unique_census_tracts):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            unique_census_tracts.add(row_census_tract_fixed)
    
    # Initialize dictionary
    air_conditioning_by_census_tract = {}
    for census_tract in unique_census_tracts:
        air_conditioning_by_census_tract[census_tract] = copy.deepcopy({"has_air_conditioning": 0, "num_buildings": 0})

    # Aggregate air conditioning per census tract
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        row_building_ac = row[air_conditioning_column_name]
        
        if not math.isnan(row_census_tract):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)

            air_conditioning_by_census_tract[row_census_tract_fixed]["num_buildings"] += 1
            if (row_building_ac is not None) and (row_building_ac != "None"):
                air_conditioning_by_census_tract[row_census_tract_fixed]["has_air_conditioning"] += 1

    # Convert counts to percentages
    for census_tract in air_conditioning_by_census_tract:
        air_conditioning_by_census_tract[census_tract]["has_air_conditioning"] = air_conditioning_by_census_tract[census_tract]["has_air_conditioning"] / air_conditioning_by_census_tract[census_tract]["num_buildings"]

    return air_conditioning_by_census_tract


def aggregate_building_height_by_census_tract(building_inventory_df, census_tract_column_name, building_height_column_name):
    # Find unique census tracts
    unique_census_tracts = set()
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        
        if (not math.isnan(row_census_tract)) and (row_census_tract not in unique_census_tracts):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            unique_census_tracts.add(row_census_tract_fixed)
    
    # Initialize dictionary
    bucket_building_height_dict = {"[1.0, 2.0)": 0, "[2.0, 6.0)": 0, "[6.0, 10.0)": 0, "[10.0, inf)": 0, "num_buildings": 0}
    building_height_by_census_tract = {}
    for census_tract in unique_census_tracts:
        building_height_by_census_tract[census_tract] = copy.deepcopy(bucket_building_height_dict)
    
    # Aggregate building height per census tract
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        row_building_height = row[building_height_column_name]
        
        if not math.isnan(row_census_tract):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)

            building_height_by_census_tract[row_census_tract_fixed]["num_buildings"] += 1
            if (not math.isnan(row_building_height)) and (row_building_height is not None) and (row_building_height != "None"):
                if 1.0 <= row_building_height < 2.0:
                    building_height_by_census_tract[row_census_tract_fixed]["[1.0, 2.0)"] += 1
                elif 2.0 <= row_building_height < 6.0:
                    building_height_by_census_tract[row_census_tract_fixed]["[2.0, 6.0)"] += 1
                elif 6.0 <= row_building_height < 10.0:
                    building_height_by_census_tract[row_census_tract_fixed]["[6.0, 10.0)"] += 1
                elif 10.0 <= row_building_height:
                    building_height_by_census_tract[row_census_tract_fixed]["[10.0, inf)"] += 1

    # Convert counts to percentages
    for census_tract in building_height_by_census_tract:
        building_height_by_census_tract[census_tract]["[1.0, 2.0)"] = building_height_by_census_tract[census_tract]["[1.0, 2.0)"] / building_height_by_census_tract[census_tract]["num_buildings"]
        building_height_by_census_tract[census_tract]["[2.0, 6.0)"] = building_height_by_census_tract[census_tract]["[2.0, 6.0)"] / building_height_by_census_tract[census_tract]["num_buildings"]
        building_height_by_census_tract[census_tract]["[6.0, 10.0)"] = building_height_by_census_tract[census_tract]["[6.0, 10.0)"] / building_height_by_census_tract[census_tract]["num_buildings"]
        building_height_by_census_tract[census_tract]["[10.0, inf)"] = building_height_by_census_tract[census_tract]["[10.0, inf)"] / building_height_by_census_tract[census_tract]["num_buildings"]

    return building_height_by_census_tract


def aggregate_building_age_by_census_tract(building_inventory_df, census_tract_column_name, building_age_column_name):
    # Find unique census tracts
    unique_census_tracts = set()
    unique_building_age_classes = set()
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        
        if (not math.isnan(row_census_tract)) and (row_census_tract not in unique_census_tracts):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            unique_census_tracts.add(row_census_tract_fixed)

        row_building_age_class = row[building_age_column_name]
        if (type(row_building_age_class) == str) and (row_building_age_class not in unique_building_age_classes):
            unique_building_age_classes.add(row_building_age_class)
    
    # Initialize dictionary
    building_age_dict = {}
    for building_age in unique_building_age_classes:
        building_age_dict[building_age] = 0
    building_age_dict["num_buildings"] = 0
    
    building_age_by_census_tract = {}
    for census_tract in unique_census_tracts:
        building_age_by_census_tract[census_tract] = copy.deepcopy(building_age_dict)
    
    # Aggregate building age per census tract
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        row_building_age = row[building_age_column_name]
        
        if (not math.isnan(row_census_tract)) and (type(row_building_age) == str):
            row_census_tract_fixed = process_census_tract_name(row_census_tract)
            
            building_age_by_census_tract[row_census_tract_fixed][row_building_age] += 1
            building_age_by_census_tract[row_census_tract_fixed]["num_buildings"] += 1

    # Convert counts to percentages
    for census_tract in building_age_by_census_tract:
        for building_age in unique_building_age_classes:
            building_age_by_census_tract[census_tract][building_age] = building_age_by_census_tract[census_tract][building_age] / building_age_by_census_tract[census_tract]["num_buildings"]
    
    return building_age_by_census_tract


if __name__ == '__main__':
    args = parse_args()
    building_inventory_df = pd.read_csv(args.building_inventory_path)
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    # Process building height
    building_height_by_census_tract = aggregate_building_height_by_census_tract(building_inventory_df, "census_tract_number", "num_floors")
    df = pd.DataFrame.from_dict(building_height_by_census_tract)
    df = df.transpose()
    df.to_csv(f"{args.save_folder}/building_height_by_census_tract.csv")

    # Process building age
    building_age_by_census_tract = aggregate_building_age_by_census_tract(building_inventory_df, "census_tract_number", "year_built_class")
    df = pd.DataFrame.from_dict(building_age_by_census_tract)
    df = df.transpose()
    df.to_csv(f"{args.save_folder}/building_age_by_census_tract.csv")
    
    # Process building use class
    use_class_by_census_tract = aggregate_building_use_class_by_census_tract(building_inventory_df, "census_tract_number", "use_class")
    df = pd.DataFrame.from_dict(use_class_by_census_tract)
    df = df.transpose()
    df.to_csv(f"{args.save_folder}/building_use_class_by_census_tract.csv")

    # Process air conditioning
    air_conditioning_by_census_tract = aggregate_air_conditioning_by_census_tract(building_inventory_df, "census_tract_number", "ac_system_type")
    df = pd.DataFrame.from_dict(air_conditioning_by_census_tract)
    df = df.transpose()
    df.to_csv(f"{args.save_folder}/building_air_conditioning_by_census_tract.csv")
