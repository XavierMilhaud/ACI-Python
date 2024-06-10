import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#import Era5var
from datetime import datetime



class PrecipitationComponent:
    """
    A class to handle precipitation data and perform related calculations.

    Attributes:
        precipitation_path (str): The file path of the precipitation data.
        mask_path (str): The file path of the mask data.
    """

    def __init__(self, precipitation_path, mask_path):
        """
        Initializes the PrecipitationComponent with precipitation and mask data.

        Args:
            precipitation_path (str): The file path of the precipitation data.
            mask_path (str): The file path of the mask data.
        """
        self.precipitation = xr.open_dataset(precipitation_path)
        self.mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})

    def apply_mask(self, var_name):
        """
        Applies a mask to the precipitation data.

        Args:
            var_name (str): The variable name in the precipitation data to be masked.

        Returns:
            xr.Dataset: The masked precipitation data.
        """
        temp = self.precipitation.copy()
        temp['mask'] = self.mask.country
        threshold = 0.8
        country_mask = temp['mask'] >= threshold
        temp[var_name] = xr.where(country_mask, temp[var_name], float('nan'))
        return temp.drop_vars('mask')

    def calculate_rolling_sum(self, var_name, window_size):
        """
        Calculates the rolling sum of precipitation over a specified window size.

        Args:
            var_name (str): The variable name in the precipitation data to calculate the rolling sum.
            window_size (int): The size of the rolling window in days.

        Returns:
            xr.DataArray: The rolling sum of precipitation.
        """
        preci = self.apply_mask(var_name)
        var = preci[var_name]
        rolling_sum = var.rolling(time=window_size).sum(dim='time')
        return rolling_sum

    def calculate_monthly_max(self, var_name, window_size):
        """
        Calculates the maximum monthly precipitation over a specified window size.

        Args:
            var_name (str): The variable name in the precipitation data to calculate the monthly maximum.
            window_size (int): The size of the rolling window in days.

        Returns:
            xr.DataArray: The maximum monthly precipitation.
        """
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        monthly_max = rolling_sum.resample(time='M').max()
        return monthly_max

    def calculate_monthly_max_anomaly(self, var_name, window_size, reference_period, area=None):
        """
        Calculates the anomaly of maximum monthly precipitation relative to a reference period.

        Args:
            var_name (str): The variable name in the precipitation data.
            window_size (int): The size of the rolling window in days.
            reference_period (tuple): A tuple containing the start and end dates of the reference period (e.g., ('1961-01-01', '1989-12-31')).

        Returns:
            xr.DataArray: The anomaly of maximum monthly precipitation.
        """
        monthly_max = self.calculate_monthly_max(var_name, window_size)
        reference_period_data = monthly_max.sel(time=slice(reference_period[0], reference_period[1]))
        rx5day_mean = reference_period_data.groupby("time.month").mean().sel(month=monthly_max.time.dt.month)
        rx5day_std = reference_period_data.groupby("time.month").std().sel(month=monthly_max.time.dt.month)
        rx5day_anomaly = ((monthly_max - rx5day_mean) / rx5day_std).drop("month")

        area = False if area is None else area
        if (not area):
            return rx5day_anomaly
        else:
            return rx5day_anomaly.mean(dim=['latitude', 'longitude'])
        
    
    def calculate_seasonly_max(self, var_name, window_size):
        """
        Calculates the maximum seasonly precipitation over a specified window size.

        Args:
            var_name (str): The variable name in the precipitation data to calculate the monthly maximum.
            window_size (int): The size of the rolling window in days.

        Returns:
            xr.DataArray: The maximum seasonly precipitation.
        """

        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        seasonly_max = rolling_sum.resample(time='QS-DEC').max()

        return seasonly_max
    
    def calculate_seasonly_max_anomaly(self, var_name, window_size, reference_period, area=None):
        """
        Calculates the anomaly of maximum seasonly precipitation relative to a reference period.

        Args:
            var_name (str): The variable name in the precipitation data.
            window_size (int): The size of the rolling window in days.
            reference_period (tuple): A tuple containing the start and end dates of the reference period (e.g., ('1961-01-01', '1989-12-31')).

        Returns:
            xr.DataArray: The anomaly of maximum seasonly precipitation.
        """

        
        seasonly_max = self.calculate_seasonly_max(var_name, window_size)
        seasonly_max_reference = seasonly_max.sel(time=slice(reference_period[0], reference_period[1]))

        time_index_season = seasonly_max.time.dt.month

        # Group the reference period data by year and month and calculate mean and standard deviation
        rx5day_mean_season = seasonly_max_reference.groupby("time.month").mean().sel(month=time_index_season)
        rx5day_std_season = seasonly_max_reference.groupby("time.month").std().sel(month=time_index_season)

        rx5day_season = ((seasonly_max - rx5day_mean_season)/rx5day_std_season).drop("month")
            


