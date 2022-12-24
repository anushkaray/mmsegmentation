import csv
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert all columsn to percent')
    parser.add_argument('--input-path',required=True, help='Input CSV path')
    parser.add_argument('--output-path',required=True, help='Output CSV path')
    args = parser.parse_args()
    return args



def convert_to_percent(csv_file, output_csv):
    new_output = {}
    pixel_df = pd.read_csv(csv_file)
    pixel_df.rename(columns = {'Unnamed: 0':'CensusTract'}, inplace = True)
    pixel_df.iloc[:,1:-1] = pixel_df.iloc[:,1:-1].div(pixel_df.total, axis=0)
    pixel_df.to_csv(output_csv)
    

    
if __name__ == '__main__':
    args = parse_args()
    file_path = args.input_path
    output = args.output_path
    convert_to_percent(file_path, output)



