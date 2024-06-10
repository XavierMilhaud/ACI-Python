import xarray as xr
import numpy as np
#import Era5var
from datetime import datetime





class DroughtComponent:
    """
    Class to process drought data and calculate standardized anomalies of consecutive dry days.

    Attributes:
    - precipitation (xarray.Dataset): Dataset containing precipitation data.
    - mask (xarray.Dataset): Dataset containing mask data.
    - drought_days (xarray.DataArray): Maximum number of consecutive dry days in each year.
    """

    def __init__(self, precipitation_path, mask_path):
        """
        Initialize the DroughtComponent object.

        Parameters:
        - precipitation (xarray.Dataset): path to nc file containing precipitation data.
        - mask (xarray.Dataset): path to nc file containing mask data.
        """
        self.precipitation = xr.open_dataset(precipitation_path)
        self.mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})
        

    def apply_mask(self, temp: xr.Dataset, var_name: str, mask: xr.Dataset) -> xr.Dataset:
        """
        Apply a mask to the precipitation data.

        Parameters:
        - temp (xarray.Dataset): Dataset containing precipitation data.
        - var_name (str): Variable name to be masked.
        - mask (xarray.Dataset): Dataset containing mask data.

        Returns:
        - xarray.Dataset: Precipitation data with mask applied.
        """
        f_temp = temp.copy()
        f_temp['mask'] = mask.country
        threshold = 0.8
        country_mask = f_temp['mask'] >= threshold
        f_temp[var_name] = xr.where(country_mask, f_temp[var_name], float('nan'))
        return f_temp.drop_vars('mask')

    def calculate_max_consecutive_dry_days(self) -> xr.DataArray:
        """
        Calculate the maximum number of consecutive dry days in each year.

        Returns:
        - xarray.DataArray: Maximum number of consecutive dry days.
        """

        preci =  self.apply_mask(self.precipitation, "tp", self.mask)
        precipitation_per_day = preci['tp'].resample(time='d').sum()
        precipitation_per_day = precipitation_per_day.to_dataset()
        precipitation_per_day['days_below_thresholds'] = xr.where(precipitation_per_day.tp < 0.001, 1, 0)
        day_sum = precipitation_per_day['days_below_thresholds'].resample(time='d').sum()
        days_above_thresholds = xr.where(day_sum < 0.001, 1, 0)

        precipitation_per_day['das'] = precipitation_per_day['days_below_thresholds'].cumsum(dim='time') - precipitation_per_day['days_below_thresholds'].cumsum(dim='time').where(precipitation_per_day['days_below_thresholds'] == 0).ffill(dim='time').fillna(0)
        precipitation_per_day = self.apply_mask(precipitation_per_day,"das", self.mask)
        days = days_above_thresholds.cumsum(dim='time') - days_above_thresholds.cumsum(dim='time').where(days_above_thresholds == 0).ffill(dim='time').fillna(0)
        #max_days_drought_per_year = days.groupby('time.year').max()
        return days

    def standardize_max_consecutive_dry_days(self, reference_period, area= None) -> xr.DataArray:
        """
        Standardize the maximum number of consecutive dry days.

        Returns:
        - xarray.DataArray: Standardized maximum number of consecutive dry days.
        """

        max_days_drought_per_month = self.calculate_max_consecutive_dry_days().resample(time='m').max()
        max_days_drought_per_month_reference = max_days_drought_per_month.sel(time=slice(reference_period[0], reference_period[1]))
        time_index = max_days_drought_per_month.time.dt.month
        cdd_mean = max_days_drought_per_month_reference.groupby("time.month").mean().sel(month=time_index)
        cdd_std = max_days_drought_per_month_reference.groupby("time.month").std().sel(month=time_index)
        cdd = ((max_days_drought_per_month - cdd_mean) / cdd_std).drop("month")

        area = False if area is None else area
        if (not area):
            return cdd
        else:
            return cdd.mean(dim=['latitude', 'longitude'])
        

