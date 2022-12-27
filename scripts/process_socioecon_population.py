import geopandas as gpd
import csv
from tqdm import tqdm
import pandas as pd
import argparse
import shapefile
import fiona


def parse_args():
    parser = argparse.ArgumentParser(description='Process Socioeconomic Data for Boston (Including Population Density)')
    parser.add_argument('--socioecon-path', required=True, help='File path to the Socioeconomic Data')
    parser.add_argument('--shapefile', required=True, help='File path to the Boston Census Tracts shapefile')
    parser.add_argument('--features', required=True, type=str, nargs='+', help = "Socioeconomic columns to find population density for")
    parser.add_argument('--output-path',required=True, help='Output CSV path')
    args = parser.parse_args()
    return args

def process_socioeconomic_population(socioecon_file, shape_file, features, output_file):
    output = {}
    ct_df = gpd.read_file(shape_file)
    ct_df = ct_df.to_crs(epsg=4326)

    pop_df = pd.read_csv(socioecon_file)
    pop_df.rename(columns = {'Unnamed: 0':'CensusTract'}, inplace = True)
    
    error_count = 0
    for ind in ct_df.index:
        ct_name = ct_df["NAME20"][ind]
        ct_name_str = str(ct_name)
        area_sq_mile = ct_df["ALAND20"][ind] / 2589988
        inner_dict = {"Area (mi^2) ": area_sq_mile}
        for feature in features:
            ct_name_float = float(ct_name)
            try:
                row = pop_df.loc[pop_df["CensusTract"] == ct_name_float]
                feature_val = row.iloc[0].loc[feature]
                feature_pop_density = feature + " Density"
                inner_dict[feature_pop_density] = feature_val / area_sq_mile
                inner_dict[feature] = feature_val
            except:
                break
        if len(inner_dict) > 1:
            output[ct_name_str] = inner_dict
    
    
    
    df = pd.DataFrame.from_dict(output)
    df = df.transpose()
    df.to_csv(output_file)
    
    
if __name__ == '__main__':
    args = parse_args()
    socioecon_file = args.socioecon_path
    boston_ct_file = args.shapefile
    features = args.features
    output_file = args.output_path
    process_socioeconomic_population(socioecon_file, boston_ct_file, features, output_file)
    


