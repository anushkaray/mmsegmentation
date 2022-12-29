import csv
from tqdm import tqdm
import pandas as pd
import math
import copy
import geopandas as gpd
import fiona
import argparse

#Defining Features Variables

#X01_AGE_AND_SEX
total_population = 'B01001e1'
male_total_pop_est = 'B01001e2'
female_total_pop_est = 'B01001e26'


#X02_RACE
white_alone_est = 'B02001e2'
black_afr_amer_alone_est = 'B02001e3'
american_indian_alaska_est = 'B02001e4'
asian_alone_est = 'B02001e5'
native_hawaii_pac_island_est = 'B02001e6'
some_other_race_est = 'B02001e7'
two_or_more_race_est = 'B02001e8'

#X03_HISPANIC_OR_LATINO_ORIGIN
total_hispanic_latino_est = 'B03001e3'


#X05_FOREIGN_BORN_CITIZENSHIP
foreign_born_not_us_citizen = 'B05002e21'
children_under_6_years = 'B05009e2'


#X16_LANGUAGE_SPOKEN_AT_HOME
spanish_limited_english = 'C16001e5'
french_hatitian_cajun_limited = 'C16001e8'
german_limited_english = 'C16001e11'
russian_polish_slavic_limited = 'C16001e14'
other_indo_europ_limited_eng = 'C16001e17'
korean_limited_english = 'C16001e20'
chinese_limited_english = 'C16001e23'
vietnamese_limited_english = 'C16001e26'
tagalog_limited_english = 'C16001e29'
other_aapi_limited_english = 'C16001e32'
arabic_limited_english = 'C16001e35'
other_limited_english = 'C16001e38'

#X17_POVERTY
total_poverty_status = 'B17001e2'

#X18_DISABILITY
under_18_one_type_disability = 'C18108e3'
under_18_two_or_more = 'C18108e4'
_18_to_64_one_type = 'C18108e7'
_18_to_64_two_or_more = 'C18108e8'
over_65_one_type = 'C18108e11'
over_65_two_or_more = 'C18108e12'


#This dictionary is needed to label the CSV columns
code_to_feature = {
                      total_population: 'total_population',
                      male_total_pop_est : 'male_total_pop_est',
                      female_total_pop_est: 'female_total_pop_est',
                      white_alone_est : 'white_alone_est',
                      black_afr_amer_alone_est : 'black_afr_amer_alone_est',
                      american_indian_alaska_est : 'american_indian_alaska_est',
                      asian_alone_est : 'asian_alone_est',
                      native_hawaii_pac_island_est : 'native_hawaii_pac_island_est',
                      some_other_race_est : 'some_other_race_est',
                      two_or_more_race_est : 'two_or_more_race_est',
                      total_hispanic_latino_est : 'total_hispanic_latino_est',
                      foreign_born_not_us_citizen : 'foreign_born_not_us_citizen',
                      children_under_6_years : 'children_under_6_years',
                      spanish_limited_english :'spanish_limited_english',
                      french_hatitian_cajun_limited : 'french_hatitian_cajun_limited',
                      german_limited_english : 'german_limited_english',
                      russian_polish_slavic_limited : 'russian_polish_slavic_limited', 
                      other_indo_europ_limited_eng : 'other_indo_europ_limited_eng',
                      korean_limited_english : 'korean_limited_english',
                      chinese_limited_english : 'chinese_limited_english',
                      vietnamese_limited_english : 'vietnamese_limited_english',
                      tagalog_limited_english : 'tagalog_limited_english',
                      other_aapi_limited_english : 'other_aapi_limited_english',
                      arabic_limited_english : 'arabic_limited_english',
                      other_limited_english : 'other_limited_english',
                      total_poverty_status : 'total_poverty_status',
                      under_18_one_type_disability : 'under_18_one_type_disability',
                      under_18_two_or_more : 'under_18_two_or_more',
                      _18_to_64_one_type : '_18_to_64_one_type',
                      _18_to_64_two_or_more : '_18_to_64_two_or_more',
                      over_65_one_type : 'over_65_one_type',
                      over_65_two_or_more : 'over_65_two_or_more'
                  }
                   
#The features to process for each layer/category 
to_process_per_layer = {'X01_AGE_AND_SEX' : [total_population,
                                             male_total_pop_est, 
                                             female_total_pop_est],
                        'X02_RACE' : [white_alone_est, 
                                      black_afr_amer_alone_est,
                                      american_indian_alaska_est,
                                      asian_alone_est,     
                                      native_hawaii_pac_island_est, 
                                      some_other_race_est,
                                      two_or_more_race_est],    
                       'X03_HISPANIC_OR_LATINO_ORIGIN': [total_hispanic_latino_est],
                       'X05_FOREIGN_BORN_CITIZENSHIP' : [foreign_born_not_us_citizen,
                                                        children_under_6_years],
                        'X16_LANGUAGE_SPOKEN_AT_HOME' : [spanish_limited_english,
                                                        french_hatitian_cajun_limited,
                                                        german_limited_english,
                                                        russian_polish_slavic_limited,
                                                        other_indo_europ_limited_eng,
                                                        korean_limited_english,
                                                        chinese_limited_english,
                                                        vietnamese_limited_english,
                                                        tagalog_limited_english,
                                                        other_aapi_limited_english,
                                                        arabic_limited_english,
                                                        other_limited_english],
                        'X17_POVERTY' : [total_poverty_status],
                        'X18_DISABILITY' : [under_18_one_type_disability,
                                           under_18_two_or_more,
                                           _18_to_64_one_type,
                                           _18_to_64_two_or_more,
                                           over_65_one_type,
                                           over_65_two_or_more]
                       }

#Grouping Together Features
poc = to_process_per_layer['X02_RACE'][1:] + to_process_per_layer['X03_HISPANIC_OR_LATINO_ORIGIN']
disability = to_process_per_layer['X18_DISABILITY']
limited_english = to_process_per_layer['X16_LANGUAGE_SPOKEN_AT_HOME']

def parse_args():
    parser = argparse.ArgumentParser(description='Process Socioeconomic Data for Boston from ACS Source')
    parser.add_argument('--socioecon-path', required=True, help='File path to the ACS Socioeconomic Data GDB File for Massachusetts')
    parser.add_argument('--shapefile', required=True, help='File path to the Boston Census Tracts shapefile')
    parser.add_argument('--output-path', required=True, help='Output CSV file save path')
    args = parser.parse_args()
    return args

def combine_groups(df, poc, disability, limited_english, code_to_feature):
    total_poc = 0
    for group in poc:
        group_str = code_to_feature[group]
        total_poc += df[group_str]
    df['POC'] = total_poc
    
    total_disability = 0
    for group in disability:
        group_str = code_to_feature[group]
        total_disability += df[group_str]
    
    df['Disabled Population'] = total_disability
    
    total_lim_english = 0
    for group in limited_english:
        group_str = code_to_feature[group]
        total_lim_english += df[group_str]
    
    df['Limited English Proficiency'] = total_lim_english
    
    
    return df

def find_pop_density(shapefile, result_df):
    ct_df = gpd.read_file(shapefile)
    ct_df = ct_df.to_crs(epsg=4326)
    output = result_df.to_dict('index')
    new_output = dict()
    for ind in ct_df.index:
        ct_name = ct_df["NAME20"][ind]
        if "." not in ct_name:
            ct_name = ct_name + '.00'
        area = ct_df["ALAND20"][ind]
        area_sq_mile = area / 2589988
        if ct_name in output:
            inner_keys = list(output[ct_name].keys())
            new_output[ct_name] = {}
            new_output[ct_name]["Area (sq miles)"] = area_sq_mile
            
            for inner_key in inner_keys:
                if inner_key != "total_population":
                    pop_den = inner_key + " Population Per Area"
                    new_output[ct_name][pop_den] = output[ct_name][inner_key] / area_sq_mile
                    new_output[ct_name][inner_key] = output[ct_name][inner_key]
                else:
                    new_output[ct_name][inner_key] = output[ct_name][inner_key]
    df = pd.DataFrame.from_dict(new_output)
    df = df.transpose()
    return df
    

def process_acs_data(gdb_path, to_process_per_layer, code_to_feature, poc, disability, limited_english, shapefile_path):
    ct_df = gpd.read_file(shapefile_path)
    ct_df = ct_df.to_crs(epsg=4326)
    layers = fiona.listlayers(gdb_path)
    counter = 0
    output = {}
    for layer in tqdm(layers):
        if layer in to_process_per_layer:
            gdf = gpd.read_file(gdb_path, layer=layer)
            
            for index, row in gdf.iterrows():
                if "US25025" in row["GEOID"]: # massachusetts , suffolk county
                    inner_dict = dict()
                    census_tract = str(int(row["GEOID"][-6:]))
                    census_tract = census_tract[:-2] + "." + census_tract[-2:]
                    features = to_process_per_layer[layer]
                    
                    for feature in features:
                        feat = row[feature]
                        feat_string = code_to_feature[feature]
                        if census_tract in output:
                            output[census_tract][feat_string] = feat
                        else:
                            inner_dict[feat_string] = feat
                    
                    if census_tract not in output:
                        output[census_tract] = inner_dict
    df = pd.DataFrame.from_dict(output)
    df = df.transpose()
    df = combine_groups(df, poc, disability, limited_english, code_to_feature) 
    return df


def get_condensed(df, output_path_condense, poc, disability, limited_english):
    values_to_include = ['total_population','POC','Disabled Population', 'Limited English Proficiency', 'total_population', 'male_total_pop_est','female_total_pop_est', 'foreign_born_not_us_citizen', 'children_under_6_years', 'total_poverty_status']
        
    output_df = df[values_to_include].copy()
    return output_df
    
    
if __name__ == '__main__':
    args = parse_args()
    boston_census_tract_shapefile = args.shapefile
    gdb_path = args.socioecon_path
    output_path_condense = args.output_path
    big_df = process_acs_data(gdb_path, to_process_per_layer, code_to_feature, poc, disability, limited_english, boston_census_tract_shapefile)
    condensed_df = get_condensed(big_df, output_path_condense, poc, disability, limited_english)
    condensed_pop_df = find_pop_density(boston_census_tract_shapefile, condensed_df)
    condensed_pop_df.to_csv(output_path_condense)



