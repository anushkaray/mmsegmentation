import pandas as pd
import csv
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate features using weights')
    parser.add_argument('--features-path', required=True, help='File path to CSV with features (columns) per census tract (rows)')
    parser.add_argument('--weights-path', required=True, help='File path to JSON with keys as feature names to use and values as weights per feature')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top census tracts to output')
    parser.add_argument('--save-path',required=True, help='File path to CSV with top K census tracts based on weighted sum of features')
    parser.add_argument('--normalize-features', action='store_true', help='Normalize features to be between 0 and 1')
    
    args = parser.parse_args()
    return args


def normalize_features_df(features_df, weights):
    output_df = features_df.copy()

    for feature_name in weights.keys():
        max_value = features_df[feature_name].max()
        min_value = features_df[feature_name].min()
        output_df[feature_name] = (features_df[feature_name] - min_value) / (max_value - min_value)

    return output_df


def assign_weights_to_features(features_df, weights, weighted_sum_col):
    feature_names = weights.keys()

    for i, feature_name in enumerate(feature_names):
        if i == 0:
            features_df[weighted_sum_col] = features_df[feature_name] * weights[feature_name]
        else:
            features_df[weighted_sum_col] += features_df[feature_name] * weights[feature_name]

    return features_df


if __name__ == '__main__':
    args = parse_args()
    
    features_df = pd.read_csv(args.features_path)
    with open(args.weights_path) as f:
        weights = json.load(f)
    
    if args.normalize_features:
        features_df = normalize_features_df(features_df, weights)
    
    weighted_sum_col = "Weighted Sum"
    features_df = assign_weights_to_features(features_df, weights, weighted_sum_col)

    top_census_tracts_df = features_df.nlargest(args.top_k, weighted_sum_col)
    top_census_tracts_df.to_csv(args.save_path)
