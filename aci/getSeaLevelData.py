#!/usr/bin/env python3

import os
import shutil
import pandas as pd
import argparse

# Load the DataFrame A
# Assuming A is a CSV file, replace 'path_to_dataframe.csv' with the actual path to your CSV
try:
    df = pd.read_csv('../data/required_data/psmsl_data.csv')
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
    exit(1)

# Directory containing the .rlrdata files
source_dir = '../data/required_data/rlr_monthly/data'
if not os.path.exists(source_dir):
    print(f"Source directory {source_dir} does not exist.")
    exit(1)

# Function to copy and rename files based on country abbreviation
def copy_and_rename_files_by_country(abbreviation):
    # Directory to copy files to
    target_dir = f'../data/sealevel_data_{abbreviation}'

    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory {target_dir}")

    # Filter DataFrame by country abbreviation
    filtered_df = df[df['Country'] == abbreviation]
    if filtered_df.empty:
        print(f"No entries found for country abbreviation {abbreviation}")
        return

    # Iterate over each ID and copy the associated file, then rename it
    for file_id in filtered_df['ID']:
        source_file = os.path.join(source_dir, f'{file_id}.rlrdata')
        if os.path.exists(source_file):
            destination_file = os.path.join(target_dir, f'{file_id}.rlrdata')
            shutil.copy(source_file, destination_file)
            print(f'Copied {source_file} to {destination_file}')

            # Rename the copied file to .txt
            new_filename = os.path.join(target_dir, f'{file_id}.txt')
            os.rename(destination_file, new_filename)
            print(f'Renamed {destination_file} to {new_filename}')
        else:
            print(f'File {source_file} does not exist')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy .rlrdata files based on country abbreviation and rename them to .txt.')
    parser.add_argument('country_abbreviation', type=str, help='The country abbreviation to filter by')
    args = parser.parse_args()

    # Print the country abbreviation for confirmation
    print(f"Country abbreviation provided: {args.country_abbreviation}")

    # Copy and rename the files
    copy_and_rename_files_by_country(args.country_abbreviation)
