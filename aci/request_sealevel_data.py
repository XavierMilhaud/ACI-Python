import os
import shutil
import pandas as pd
import requests
import zipfile
import argparse
import sys

from aci.datasets import load_psmsl_data

# Constants
URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip"
DESTINATION_DIR = "data/required_data"
ZIP_FILE_PATH = os.path.join(DESTINATION_DIR, "rlr_monthly.zip")
EXTRACT_PATH = os.path.join(DESTINATION_DIR, "rlr_monthly")
#CSV_FILE_PATH = os.path.join(DESTINATION_DIR, 'psmsl_data.csv')
SOURCE_DIR = os.path.join(EXTRACT_PATH, 'data')


def download_and_extract_data():
    """
    Downloads and extracts the data from the given URL if the data
    doesn't already exist.
    """
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(DESTINATION_DIR, exist_ok=True)
        response = requests.get(URL)
        with open(ZIP_FILE_PATH, 'wb') as file:
            file.write(response.content)

        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(DESTINATION_DIR)

        os.remove(ZIP_FILE_PATH)
        print("Data downloaded and extracted successfully.")
    else:
        print("The directory rlr_monthly already exists. No action needed.")


def load_dataframe():
    """
    Loads the data from a CSV file into a DataFrame.

    Returns:
    --------
    pd.DataFrame:
        The loaded DataFrame.
    """
    try:
        df = load_psmsl_data()
        print("DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print("CSV file not found. Please check the file path.")
        sys.exit(1)


def copy_and_rename_files_by_country(abbreviation, df):
    """
    Copies and renames files based on the country abbreviation.

    Parameters:
    -----------
    abbreviation : str
        The country abbreviation to filter by.
    df : pd.DataFrame
        The DataFrame containing the file information.
    """
    target_dir = f'data/sealevel_data_{abbreviation}'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory {target_dir}")

    filtered_df = df[df['Country'] == abbreviation]
    if filtered_df.empty:
        print(f"No entries found for country abbreviation {abbreviation}")
        return

    for file_id in filtered_df['ID']:
        source_file = os.path.join(SOURCE_DIR, f'{file_id}.rlrdata')
        if os.path.exists(source_file):
            destination_file = os.path.join(target_dir, f'{file_id}.rlrdata')
            shutil.copy(source_file, destination_file)

            new_filename = os.path.join(target_dir, f'{file_id}.txt')
            os.rename(destination_file, new_filename)
        else:
            print(f'File {source_file} does not exist')


def main(country_abbreviation):
    """
    Main function to download data, load DataFrame, and process files.

    Parameters:
    -----------
    country_abbreviation : str
        The country abbreviation to filter by.
    """
    download_and_extract_data()
    df = load_dataframe()
    copy_and_rename_files_by_country(country_abbreviation, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Copy .rlrdata files based on country abbreviation and '
            'rename them to .txt.'
        )
    )
    parser.add_argument(
        'country_abbreviation',
        type=str,
        help='The country abbreviation to filter by'
    )
    args = parser.parse_args()

    main(args.country_abbreviation)
