import xarray as xr
import numpy as np

class Component:
    """
    Base class for components that handle various climate data and perform related calculations.

    Attributes:
    - array (xarray.Dataset): The dataset containing the primary data.
    - mask (xarray.Dataset): The dataset containing the mask data.
    - file_name (str): The file name of the dataset.

    """

    def __init__(self, array, mask, file_name):
        """
        Initializes the Component with primary data and mask data.

        Args:
            array (xarray.Dataset): The dataset containing the primary data.
            mask (xarray.Dataset): The dataset containing the mask data.
            file_name (str): The file name of the dataset.
        """
        self.array = array
        self.mask = mask
        self.file_name = file_name
    

    def apply_mask(self, var_name, threshold=0.8):
        """
        Apply a mask to the dataset.

        Parameters:
        - var_name (str): Variable name in the dataset to which the mask is applied.
        - threshold (float): Threshold value for the mask. Default is 0.8.

        Returns:
        - xarray.Dataset: Dataset with the mask applied to the specified variable.
        """
        if self.array is None or self.mask is None:
            raise ValueError("Data not loaded. Please ensure precipitation and mask data are loaded.")
        
        f_temp = self.array.copy()
        f_temp['mask'] = self.mask.country
        
        # Create a mask based on the threshold
        country_mask = f_temp['mask'] >= threshold
        
        # Apply the mask to the precipitation data
        f_temp[var_name] = xr.where(country_mask, f_temp[var_name], float('nan'))
        
        return f_temp.drop_vars('mask')


    def standardize_metric(self, metric, reference_period, area=None):
        """
        Standardizes a given metric based on a reference period.

        Args:
            metric (xarray.DataArray): The metric to be standardized.
            reference_period (tuple): A tuple containing the start and end dates of the reference period.
            area (bool): If True, calculate the area-averaged standardized metric. Default is None.

        Returns:
            xarray.DataArray: The standardized metric.
        """
        reference = metric.sel(time=slice(reference_period[0], reference_period[1]))
        time_index = metric.time.dt.month
        mean = reference.groupby("time.month").mean().sel(month=time_index)
        std = reference.groupby("time.month").std().sel(month=time_index)
        standardized = ((metric - mean) / std).drop("month")

        if area:
            return standardized.mean(dim=['latitude', 'longitude'])
        else:
            return standardized

    def calculate_rolling_sum(self, var_name, window_size):
        """
        Calculates the rolling sum of a variable over a specified window size.

        Args:
            var_name (str): The variable name in the data to calculate the rolling sum.
            window_size (int): The size of the rolling window.

        Returns:
            xarray.DataArray: The rolling sum of the variable.
        """
        data = self.apply_mask(var_name)
        var = data[var_name]
        rolling_sum = var.rolling(time=window_size).sum(dim='time')
        return rolling_sum
