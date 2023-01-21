import csv
from tqdm import tqdm
import pandas as pd
import math
import copy

# Census tract mapping from 2010 to 2020
map_2010_to_2020_census_tracts = {
    1304.02: [9819.0],
    8.02: [8.04, 8.05],
    8.03: [8.06, 8.07],
    5.04: [5.05, 5.06],
    1.0: [1.01, 1.02],
    6.02: [6.03, 6.04],
    102.03: [102.05, 102.06],
    203.03: [203.04, 203.05],
    303.0: [303.01, 303.02],
    606.0: [606.01, 606.02, 606.03, 606.04],
    612.0: [612.01, 612.02, 612.03, 612.04],
    701.01: [701.02, 701.03, 701.04],
    702.0: [702.01, 702.02],
    703.0: [703.01, 703.02],
    705.0: [705.01, 705.02],
    708.0: [708.01, 708.02],
    709.0: [709.01, 709.02],
    811.0: [811.01, 811.02],
    813.0: [813.01, 813.02],
    1101.03: [1101.04, 1101.05, 1101.06],
    1301.0: [1301.01],
}

# Set of 2020 census tracts
_2020_ct = {
    1.01, 1.02, 2.01, 2.02, 3.01, 3.02, 4.01, 4.02, 5.02, 5.03, 5.05, 5.06,
    6.01, 6.03, 6.04, 7.01, 7.03, 7.04, 8.04, 8.05, 8.06, 8.07, 101.03, 101.04,
    102.04, 102.05, 102.06, 103.0, 104.03, 104.04, 104.05, 104.08, 105.0, 106.0,
    107.01, 107.02, 108.01, 108.02, 201.01, 202.0, 203.01, 203.02, 203.04, 203.05,
    301.0, 302.0, 303.01, 303.02, 304.0, 305.0, 401.0, 402.0, 403.0, 404.01, 406.0,
    408.01, 501.01, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 509.01, 510.0, 511.01,
    512.0, 601.01, 602.0, 603.01, 604.0, 605.01, 606.01, 606.02, 606.03, 606.04, 607.0,
    608.0, 610.0, 611.01, 612.01, 612.02, 612.03, 612.04, 701.02, 701.03, 701.04, 702.01,
    702.02, 703.01, 703.02, 704.02, 705.01, 705.02, 706.0, 707.0, 708.01, 708.02, 709.01,
    709.02, 711.01, 712.01, 801.0, 803.0, 804.01, 805.0, 806.01, 808.01, 809.0, 810.01,
    811.01, 811.02, 812.0, 813.01, 813.02, 814.0, 815.0, 817.0, 818.0, 819.0, 820.0, 821.0,
    901.0, 902.0, 903.0, 904.0, 906.0, 907.0, 909.01, 910.01, 911.0, 912.0, 913.0, 914.0,
    915.0, 916.0, 917.0, 918.0, 919.0, 920.0, 921.01, 922.0, 923.0, 924.0, 1001.0, 1002.0,
    1003.0, 1004.0, 1005.0, 1006.01, 1006.03, 1007.0, 1008.0, 1009.0, 1010.01, 1010.02,
    1011.01, 1011.02, 1101.04, 1101.05, 1101.06, 1102.01, 1103.01, 1104.01, 1104.03, 1105.01,
    1105.02, 1106.01, 1106.07, 1201.03, 1201.04, 1201.05, 1202.01, 1203.01, 1204.0, 1205.0,
    1206.0, 1207.0, 1301.01, 1302.0, 1303.0, 1304.02, 1304.04, 1304.06, 1401.02, 1401.05,
    1401.06, 1401.07, 1402.01, 1402.02, 1403.0, 1404.0, 9801.01, 9803.0, 9807.0, 9809.0,
    9810.0, 9811.0, 9812.01, 9812.02, 9813.0, 9815.01, 9815.02, 9816.0, 9817.0, 9818.0, 9819.0,
}

def convert_data_2010_to_2020(_2010_csv, _2020_csv_save_path, ct_col_name):
    _2010df = pd.read_csv(_2010_csv)
    _2010df_cols = _2010df.columns
    
    # Create new DataFrame to store data with 2020 census tracts
    _2020df = pd.DataFrame(columns = _2010df_cols)

    for index, row in _2010df.iterrows():
        ct_name = row[ct_col_name]
        
        # Census tract 1304.02 in 2010 is unique because it becomes 1304.02 and 9819 in 2020
        if ct_name == 1304.02:
            _2020df = _2020df.append(_2010df.iloc[index])
            _2020_name = map_2010_to_2020_census_tracts[ct_name][0]
            row_dict = row.to_dict()
            new_dict = row_dict.copy()
            new_dict[ct_col_name] = _2020_name
            row_df = pd.DataFrame(new_dict, index=[len(_2020df.index)])
            _2020df = pd.concat([_2020df, row_df])
        
        # Census tract in 2010 maps to different number in 2020
        elif ct_name in map_2010_to_2020_census_tracts:
            _2020_names = map_2010_to_2020_census_tracts[ct_name]
            row_dict = row.to_dict()
            
            for new_name in _2020_names:
                new_dict = row_dict.copy()
                new_dict[ct_col_name] = new_name
                row_df = pd.DataFrame(new_dict, index=[len(_2020df.index)])
                _2020df = pd.concat([_2020df, row_df])
        
        # Census tract in 2010 maps to same number in 2020
        elif ct_name in _2020_ct:
            _2020df = _2020df.append(_2010df.iloc[index])
    
    # Save CSV with data mapped to 2020 census tracts
    _2020df.to_csv(_2020_csv_save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Map Boston data from 2010 to 2020 census tracts')
    parser.add_argument('--csv-path', required=True, help='File path to CSV with data for 2010 census tracts (rows)')
    parser.add_argument('--save-path', required=True, help='File path to save CSV with data for 2020 census tracts')
    parser.add_argument('--ct-col-name', required=True, default='Unnamed: 0', help='Column name for census tracts')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    convert_data_2010_to_2020(args.csv_path, args.save_path, args.ct_col_name)
