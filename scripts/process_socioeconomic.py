import geopandas as gpd
from shapely.geometry import Polygon
import csv
from tqdm import tqdm
import pandas as pd
import math
import copy
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process Socioeconomic Data for Boston')
    parser.add_argument('--boston-building-path', required=True, help='File path to the Boston Buildings Inventory dataset')
    parser.add_argument('--vulnerabilities', required=True, type=str, nargs='+', help = "Socioeconomic columns to process from the Boston buildings inventory")
    parser.add_argument('--output-path',required=True, help='Output CSV path')
    args = parser.parse_args()
    return args

def unique_census_tracts(building_inventory_df, census_tract_column_name):
    unique_census_tracts = set()
    unique_building_use_classes = set()
    for index, row in building_inventory_df.iterrows():
        row_census_tract = row[census_tract_column_name]
        if (not math.isnan(row_census_tract)) and (row_census_tract not in unique_census_tracts):
            unique_census_tracts.add(row_census_tract)
    return unique_census_tracts

def process_socioecon(unique_ct, vulnerable_pops, boston_df):
    output = {}
    non_percent = set()

    for vulnerable in vulnerable_pops:
        if 'pop' in vulnerable:
            non_percent.add(vulnerable)
    for ct in unique_ct:
        int_ct = int(ct)
        inner_dict = {}
        for vulnerable in vulnerable_pops:
            inner_dict[vulnerable] = 0
        for pop in vulnerable_pops:
            row = boston_df.loc[boston_df["census_tract_number"] == int_ct]
            if pop in non_percent:
                socio = row.iloc[0].loc[pop]
            else:
                socio = (row.iloc[0].loc[pop]) / 100
            
            inner_dict[pop] = socio
        row_census_tract_str = str(int(ct))
        row_census_tract_fixed = str(int(row_census_tract_str[5:9])) + "." + row_census_tract_str[-2:]
        output[row_census_tract_fixed] = inner_dict
    return output



if __name__ == '__main__':
    args = parse_args()
    boston_building_inventory_path = args.boston_building_path
    vulnerable_pops = args.vulnerabilities
    output_file =args.output_path
    boston_building_inventory_df = pd.read_csv(boston_building_inventory_path)
    
    unique_census_tracts = unique_census_tracts(boston_building_inventory_df, "census_tract_number")
    output_dict = process_socioecon(unique_census_tracts, vulnerable_pops, boston_building_inventory_df)
    
    df = pd.DataFrame.from_dict(output_dict)
    df = df.transpose()
    df.to_csv(output_file)
    