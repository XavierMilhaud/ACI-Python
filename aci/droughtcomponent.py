import xarray as xr
import numpy as np
from component import Component

class DroughtComponent(Component):
    """
    Class to process drought data and calculate standardized anomalies of consecutive dry days.

    Attributes:
    - precipitation (xarray.Dataset): Dataset containing precipitation data.
    - mask (xarray.Dataset): Dataset containing mask data.
    """

    def __init__(self, precipitation_path, mask_path):
        """
        Initialize the DroughtComponent object.

        Parameters:
        - precipitation_path (str): Path to the dataset containing precipitation data.
        - mask_path (str): Path to the dataset containing mask data.
        """
        precipitation = xr.open_dataset(precipitation_path)
        mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})
        super().__init__(precipitation, mask, precipitation_path)

    def max_consecutive_dry_days(self):
        """
        Calculate the maximum number of consecutive dry days in each year.

        Returns:
        - xarray.DataArray: Maximum number of consecutive dry days.
        """
    
        preci = self.apply_mask("tp")
        precipitation_per_day = preci['tp'].resample(time='d').sum()
        precipitation_per_day = precipitation_per_day.to_dataset()
        precipitation_per_day['days_below_thresholds'] = xr.where(precipitation_per_day.tp < 0.001, 1, 0)
        day_sum = precipitation_per_day['days_below_thresholds'].resample(time='d').sum()
        days_above_thresholds = xr.where(day_sum < 0.001, 1, 0)
        precipitation_per_day['das'] = (
            precipitation_per_day['days_below_thresholds'].cumsum(dim='time') 
            - precipitation_per_day['days_below_thresholds'].cumsum(dim='time')
            .where(precipitation_per_day['days_below_thresholds'] == 0)
            .ffill(dim='time')
            .fillna(0)
        )

        days = (
            days_above_thresholds.cumsum(dim='time') 
            - days_above_thresholds.cumsum(dim='time')
            .where(days_above_thresholds == 0)
            .ffill(dim='time')
            .fillna(0)
        )
        
        return days

    def std_max_consecutive_dry_days(self, reference_period, area=None):
        """
        Standardize the maximum number of consecutive dry days.

        Parameters:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.
        - area (bool): If True, calculate the area-averaged standardized metric. Default is None.

        Returns:
        - xarray.DataArray: Standardized maximum number of consecutive dry days.
        """
        max_days_drought_per_month = self.max_consecutive_dry_days().resample(time='m').max()
        return self.standardize_metric(max_days_drought_per_month, reference_period, area)
