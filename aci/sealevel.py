#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getSeaLevelData as gd
from datetime import datetime, timedelta

class SeaLevelComponent:
    """
    A class used to represent the Sea Level Component.

    Attributes
    ----------
    directory : str
        directory path where sea level data files are stored
    study_period : tuple
        tuple of start and end dates for the study period
    reference_period : tuple
        tuple of start and end dates for the reference period

    Methods
    -------
    load_data():
        Loads sea level data from files in the directory.
    correct_date_format(data):
        Corrects the date format from a specific float representation to YYYY-MM-DD.
    clean_data(data):
        Cleans the data by replacing sentinel values with NaN.
    compute_monthly_stats(data, reference_period, stats):
        Computes monthly statistics (mean or std deviation) for the reference period.
    standardize_data(data, monthly_means, monthly_std_devs, study_period):
        Standardizes the data using the reference period statistics.
    process():
        Full processing of the sea level data: loading, correcting dates, cleaning, and standardizing.
    plot_rolling_mean(data, window=60):
        Plots the rolling mean of the data.
    convert_to_xarray(data):
        Converts the data to an xarray.
    resample_data(data):
        Resamples the data to a specified frequency.
    save_to_netcdf(data, filename):
        Saves the data to a NetCDF file.
    """

    def __init__(self, country_abrev, study_period, reference_period):
        """
        Constructs all the necessary attributes for the SeaLevelComponent object.

        Parameters
        ----------
            country_abrev : str
                Abbreviation of the country for which the sea level data is relevant.
            study_period : tuple
                Tuple containing the start and end date of the study period (YYYY-MM-DD).
            reference_period : tuple
                Tuple containing the start and end date of the reference period (YYYY-MM-DD).
        """
        gd.copy_and_rename_files_by_country(country_abrev)
        self.directory = "../data/sealevel_data_" + country_abrev
        self.study_period = study_period
        self.reference_period = reference_period

    def load_data(self):
        """
        Loads sea level data from files in the directory.

        Returns
        -------
        DataFrame
            A DataFrame containing the concatenated data from all files.
        """
        dataframes = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.directory, filename)
                temp_data = pd.read_csv(file_path, sep=";", names=["Date", f"Measurement_{filename[:-4]}", "2", "3"], skipinitialspace=True)
                temp_data = temp_data[["Date", f"Measurement_{filename[:-4]}"]]
                temp_data["Date"] = temp_data["Date"].astype(float)
                temp_data.set_index("Date", inplace=True)
                dataframes.append(temp_data)

        combined_data = pd.concat(dataframes, axis=1)
        #print(f"Loaded data from {len(dataframes)} files.")
        #print("Data range:", combined_data.index.min(), "-", combined_data.index.max())
        return combined_data

    def correct_date_format(self, data):
        """
        Corrects the date format from a specific float representation to YYYY-MM-DD.

        Parameters
        ----------
        data : DataFrame
            The DataFrame with the original date format.

        Returns
        -------
        DataFrame
            The DataFrame with corrected date format.
        """
        month_mapping = {
            "0417": "01", "125": "02", "2083": "03", "2917": "04", "375": "05", 
            "4583": "06", "5417": "07", "625": "08", "7083": "09", "7917": "10", 
            "875": "11", "9583": "12"
        }

        corrected_dates = []
        for date in data.index:
            date_str = str(date)
            try:
                year = int(float(date_str))
                month_fraction = date_str.split('.')[1][:4]
                if month_fraction in month_mapping:
                    month = month_mapping[month_fraction]
                    corrected_date = f"{year}-{month}-01"
                    corrected_dates.append(corrected_date)
                else:
                    corrected_dates.append(np.nan)
            except (ValueError, IndexError) as e:
                corrected_dates.append(np.nan)

        data['Corrected_Date'] = pd.to_datetime(corrected_dates, errors='coerce')
        #print("Corrected dates:")
        #print(data['Corrected_Date'].head(10))

        data.dropna(subset=['Corrected_Date'], inplace=True)
        data.set_index('Corrected_Date', inplace=True)
        data.sort_index(inplace=True)  # Ensure the dates are sorted
        return data

    def clean_data(self, data):
        """
        Cleans the data by replacing sentinel values with NaN.

        Parameters
        ----------
        data : DataFrame
            The DataFrame to be cleaned.

        Returns
        -------
        DataFrame
            The cleaned DataFrame.
        """
        data.replace(-99999.0, np.nan, inplace=True)
        return data

    def compute_monthly_stats(self, data, reference_period, stats):
        """
        Computes monthly statistics (mean or std deviation) for the reference period.

        Parameters
        ----------
        data : DataFrame
            The DataFrame containing the sea level data.
        reference_period : tuple
            Tuple containing the start and end date of the reference period (YYYY-MM-DD).
        stats : str
            The type of statistics to compute ("means" or "std").

        Returns
        -------
        Series
            A Series containing the monthly statistics.
        """
        reference_period_mask = (data.index >= reference_period[0]) & (data.index < reference_period[1])
        data_ref = data[reference_period_mask]
        mean_ref = data_ref.mean(axis=1)

        monthly_means = mean_ref.groupby(mean_ref.index.month).mean()
        monthly_std_devs = mean_ref.groupby(mean_ref.index.month).std()

        if stats == "means":
            return monthly_means
        elif stats == "std":
            return monthly_std_devs
        else:
            raise Exception("stats in wrong format: should be 'means' or 'std'")

    def standardize_data(self, data, monthly_means, monthly_std_devs, study_period):
        """
        Standardizes the data using the reference period statistics.

        Parameters
        ----------
        data : DataFrame
            The DataFrame containing the sea level data.
        monthly_means : Series
            A Series containing the monthly means for the reference period.
        monthly_std_devs : Series
            A Series containing the monthly standard deviations for the reference period.
        study_period : tuple
            Tuple containing the start and end date of the study period (YYYY-MM-DD).

        Returns
        -------
        DataFrame
            The standardized DataFrame.
        """
        study_period_mask = (data.index >= study_period[0]) & (data.index < study_period[1])
        data_study = data[study_period_mask]
        standardized_df = data_study.copy()

        if data_study.index.isna().any():
            print("NaN values detected in index. Dropping NaN values.")
            data_study = data_study.dropna()

        if data_study.empty:
            print(f"No valid data in the study period after dropping NaN values. Study period: {study_period}")
            raise ValueError("No valid data in the study period after dropping NaN values.")

        min_year = int(data_study.index.year.min())
        max_year = int(data_study.index.year.max()) + 1

        #print(f"Min year: {min_year}, Max year: {max_year}")

        for year in range(min_year, max_year):
            for month in range(1, 13):
                month_data = data_study.loc[(data_study.index.month == month) & (data_study.index.year == year)]
                if not month_data.empty:
                    z_score = (month_data - monthly_means.loc[month]) / monthly_std_devs.loc[month]
                    standardized_df.loc[(standardized_df.index.month == month) & (standardized_df.index.year == year)] = z_score.values[0]

        return standardized_df

    def process(self):
        """
        Full processing of the sea level data: loading, correcting dates, cleaning, and standardizing.

        Returns
        -------
        DataFrame
            The fully processed and standardized DataFrame.
        """
        sea_level_data = self.load_data()
        #print("Original Data:")
        #print(sea_level_data.head())

        sea_level_data = self.correct_date_format(sea_level_data)
        #print("Data after date correction:")
        #print(sea_level_data.head())

        sea_level_data = self.clean_data(sea_level_data)
        #print("Data after cleaning:")
        #print(sea_level_data.head())

        monthly_means = self.compute_monthly_stats(sea_level_data, self.reference_period, "means")
        monthly_std_devs = self.compute_monthly_stats(sea_level_data, self.reference_period, "std")

        standardized_data = self.standardize_data(sea_level_data, monthly_means, monthly_std_devs, self.study_period)
        return standardized_data

    def plot_rolling_mean(self, data, window=60):
        """
        Plots the rolling mean of the data.

        Parameters
        ----------
        data : DataFrame
            The DataFrame containing the sea level data.
        window : int, optional
            The window size for calculating the rolling mean (default is 60).
        """
        data.rolling(window, min_periods=30, center=True).mean().plot()

    def convert_to_xarray(self, data):
        """
        Converts the data to an xarray.

        Parameters
        ----------
        data : DataFrame
            The DataFrame to be converted.

        Returns
        -------
        xarray.DataArray
            The converted xarray.
        """
        return data.to_xarray()

    def resample_data(self, data):
        """
        Resamples the data to a specified frequency.

        Parameters
        ----------
        data : DataFrame
            The DataFrame to be resampled.

        Returns
        -------
        DataFrame
            The resampled DataFrame.
        """
        return data.resample('3M').mean()

    def save_to_netcdf(self, data, filename):
        """
        Saves the data to a NetCDF file.

        Parameters
        ----------
        data : xarray.DataArray
            The data to be saved.
        filename : str
            The name of the file to save the data.
        """
        data.to_netcdf(filename)

if __name__ == "__main__":
    sea_level_component = SeaLevelComponent("USA", ('1960-01-01', '1969-12-31'), ('1960-01-01', '1964-12-31'))

    try:
        standardized_data = sea_level_component.process()
        print("\n" + "SEA LEVEL DATA" + "\n")
        print(standardized_data)
        print("\n")
        sea_level_component.plot_rolling_mean(standardized_data.mean(axis=1))
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")

