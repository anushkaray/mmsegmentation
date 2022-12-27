import csv
import pandas as pd
import geopandas as gpd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Compute population divided by area per census tract')
    parser.add_argument('--ct-data-path', required=True, help='File path to census tract data')
    parser.add_argument('--ct-data-pop-name', required=True, help='Column in census tract data for population to process')
    parser.add_argument('--ct-data-ct-name', default="TRACT", help='Column in census tract data for census tract name')
    parser.add_argument('--ct-shapefile-path', required=True, help='File path to census tract shapefile')
    parser.add_argument('--ct-shapefile-area-name', required=True, help='Column in census tract shapefile for area')
    parser.add_argument('--ct-shapefile-area-units', default="square_miles", help='Select either square_miles, square_kilometers, or square_meters')
    parser.add_argument('--ct-shapefile-ct-name', default="NAME20", help='Column in census tract shapefile for census tract name')
    parser.add_argument('--save-csv-path',required=True, help='File path to save CSV')
    
    args = parser.parse_args()
    return args


def find_ct_pop_per_area(
    census_tract_df,
    census_tract_df_pop_name,
    census_tract_df_ct_name,
    census_tract_shapefile,
    census_tract_shapefile_area_name,
    census_tract_shapefile_area_units,
    census_tract_shapefile_ct_name,
    save_csv_path,
):
    ct_pop = {}
    ct_area = {}
    ct_pop_per_area = {}

    for index, row in census_tract_df.iterrows():
        if index == 0: # the first row contains column descriptions
            continue
        
        row_ct = row[census_tract_df_ct_name]
        if row_ct[-2:] == "00":
            row_ct = row_ct[:-2]
        else:
            row_ct = row_ct[:-2] + "." + row_ct[-2:] # add decimal point to census tract name
        
        row_pop = row[census_tract_df_pop_name]
        ct_pop[row_ct] = row_pop

    for index, row in census_tract_shapefile.iterrows():
        row_ct = row[census_tract_shapefile_ct_name]
        row_area = row[census_tract_shapefile_area_name]
        ct_area[row_ct] = row_area
        
        if census_tract_shapefile_area_units == "square_miles":
            ct_area[row_ct] = row_area/2589988 # divide by 2589988 as per https://www.census.gov/quickfacts/fact/note/US/LND110210
        elif census_tract_shapefile_area_units == "square_kilometers":
            ct_area[row_ct] = row_area/1000000

    all_ct_names = (ct_pop.keys() | ct_area.keys())
    for ct_name in all_ct_names:
        if (ct_name in ct_area) and (ct_name in ct_pop):

            ct_pop_per_area[ct_name] = {"population": ct_pop[ct_name],
                                        f"area_{census_tract_shapefile_area_units}": ct_area[ct_name],
                                        "population_per_area": int(ct_pop[ct_name]) / ct_area[ct_name]}

    final_df = pd.DataFrame.from_dict(ct_pop_per_area)
    final_df = final_df.transpose()
    final_df.to_csv(save_csv_path)


if __name__ == '__main__':
    args = parse_args()

    census_tract_df = pd.read_csv(args.ct_data_path)
    census_tract_shapefile = gpd.read_file(args.ct_shapefile_path)
    census_tract_shapefile.to_crs('epsg:4326', inplace=True)

    find_ct_pop_per_area(census_tract_df,
                         args.ct_data_pop_name,
                         args.ct_data_ct_name,
                         census_tract_shapefile,
                         args.ct_shapefile_area_name,
                         args.ct_shapefile_area_units,
                         args.ct_shapefile_ct_name,
                         args.save_csv_path)
