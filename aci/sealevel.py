import os
import pandas as pd
import numpy as np
import getSeaLevelData as gd
from component import Component


class SeaLevelComponent(Component):
    """
    A class used to represent the Sea Level Component.

    Attributes
    ----------
    directory : str
        Directory path where sea level data files are stored.
    study_period : tuple
        Tuple of start and end dates for the study period.
    reference_period : tuple
        Tuple of start and end dates for the reference period.

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
        Constructs all the necessary attributes for the Sea Level Component object.

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
        self.directory = f"../data/sealevel_data_{country_abrev}"
        self.study_period = study_period
        self.reference_period = reference_period

    def load_data(self):
        """
        Loads sea level data from files in the directory.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the concatenated data from all files.

        Complexity
        ----------
        O(N * M)
        where N is the number of files and M is the number of rows per file.
        """
        dataframes = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.directory, filename)
                temp_data = pd.read_csv(
                    file_path,
                    sep=";",
                    names=["Date", f"Measurement_{filename[:-4]}", "2", "3"],
                    skipinitialspace=True
                )
                temp_data = temp_data[["Date", f"Measurement_{filename[:-4]}"]]
                temp_data["Date"] = temp_data["Date"].astype(float)
                temp_data.set_index("Date", inplace=True)
                dataframes.append(temp_data)

        combined_data = pd.concat(dataframes, axis=1)
        return combined_data

    def correct_date_format(self, data):
        """
        Corrects the date format from a specific float representation to YYYY-MM-DD.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame with the original date format.

        Returns
        -------
        pd.DataFrame
            The DataFrame with corrected date format.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataFrame.
        """
        month_mapping = {
            "0417": "01", "125": "02", "2083": "03", "2917": "04", "375": "05",
            "4583": "06", "5417": "07", "625": "08", "7083": "09", "7917": "10",
            "875": "11", "9583": "12"
        }

        def convert_date(date):
            date_str = str(date)
            try:
                year = int(float(date_str))
                month_fraction = date_str.split('.')[1][:4]
                month = month_mapping.get(month_fraction, None)
                if month:
                    return pd.Timestamp(f"{year}-{month}-01")
                return pd.NaT
            except (ValueError, IndexError):
                return pd.NaT

        corrected_dates = data.index.to_series().apply(convert_date)
        data = data.assign(Corrected_Date=corrected_dates)
        data = data.dropna(subset=['Corrected_Date'])
        data = data.set_index('Corrected_Date')
        data.sort_index(inplace=True)
        return data

    def clean_data(self, data):
        """
        Cleans the data by replacing sentinel values with NaN.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be cleaned.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataFrame.
        """
        return data.replace(-99999.0, np.nan)

    def compute_monthly_stats(self, data, reference_period, stats):
        """
        Computes monthly statistics (mean or std deviation) for the reference period.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the sea level data.
        reference_period : tuple
            Tuple containing the start and end date of the reference period (YYYY-MM-DD).
        stats : str
            The type of statistics to compute ("means" or "std").

        Returns
        -------
        pd.Series
            A Series containing the monthly statistics.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the reference period DataFrame.
        """
        reference_period_mask = (data.index >= reference_period[0]) & (data.index < reference_period[1])
        data_ref = data.loc[reference_period_mask]
        mean_ref = data_ref.mean(axis=1)

        monthly_means = mean_ref.groupby(mean_ref.index.month).mean()
        monthly_std_devs = mean_ref.groupby(mean_ref.index.month).std()

        if stats == "means":
            return monthly_means
        elif stats == "std":
            return monthly_std_devs
        else:
            raise ValueError("stats must be 'means' or 'std'")

    def standardize_data(self, data, monthly_means, monthly_std_devs, study_period):
        """
        Standardizes the data using the reference period statistics.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the sea level data.
        monthly_means : pd.Series
            A Series containing the monthly means for the reference period.
        monthly_std_devs : pd.Series
            A Series containing the monthly standard deviations for the reference period.
        study_period : tuple
            Tuple containing the start and end date of the study period (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            The standardized DataFrame.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the study period DataFrame.
        """
        study_period_mask = (data.index >= study_period[0]) & (data.index < study_period[1])
        data_study = data.loc[study_period_mask]

        def calculate_z_score(row, monthly_means, monthly_std_devs):
            month = row.name.month
            if month in monthly_means and month in monthly_std_devs:
                return (row - monthly_means[month]) / monthly_std_devs[month]
            return np.nan

        standardized_df = data_study.apply(calculate_z_score, args=(monthly_means, monthly_std_devs), axis=1)
        return standardized_df.dropna(how='all')

    def process(self):
        """
        Full processing of the sea level data: loading, correcting dates, cleaning, and standardizing.

        Returns
        -------
        pd.DataFrame
            The fully processed and standardized DataFrame.

        Complexity
        ----------
        O(N * M)
        where N is the number of rows and M is the number of columns in the final standardized DataFrame.
        """
        sea_level_data = self.load_data()
        sea_level_data = self.correct_date_format(sea_level_data)
        sea_level_data = self.clean_data(sea_level_data)
        monthly_means = self.compute_monthly_stats(sea_level_data, self.reference_period, "means")
        monthly_std_devs = self.compute_monthly_stats(sea_level_data, self.reference_period, "std")
        standardized_data = self.standardize_data(sea_level_data, monthly_means, monthly_std_devs, self.study_period)
        return standardized_data

    def plot_rolling_mean(self, data, window=60):
        """
        Plots the rolling mean of the data.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the sea level data.
        window : int, optional
            The window size for calculating the rolling mean (default is 60).

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataFrame.
        """
        data.rolling(window, min_periods=30, center=True).mean().plot()

    def convert_to_xarray(self, data):
        """
        Converts the data to an xarray.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be converted.

        Returns
        -------
        xr.DataArray
            The converted xarray.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataFrame.
        """
        return data.to_xarray()

    def resample_data(self, data):
        """
        Resamples the data to a specified frequency.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be resampled.

        Returns
        -------
        pd.DataFrame
            The resampled DataFrame.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataFrame.
        """
        return data.resample('3M').mean()

    def save_to_netcdf(self, data, filename):
        """
        Saves the data to a NetCDF file.

        Parameters
        ----------
        data : xr.DataArray
            The data to be saved.
        filename : str
            The name of the file to save the data.

        Complexity
        ----------
        O(N)
        where N is the number of rows in the DataArray.
        """
        data.to_netcdf(filename)
