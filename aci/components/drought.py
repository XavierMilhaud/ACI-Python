import xarray as xr
import os
from components.component import Component
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DroughtComponent(Component):
    """
    Class to process drought data and calculate standardized anomalies
    of consecutive dry days (CDD).

    Attributes
    ----------
    precipitation : xarray.Dataset
        Dataset containing precipitation data.
    mask : xarray.Dataset or None
        Dataset containing mask data, if provided.
    """

    def __init__(self, precipitation_source, mask_path=None):
        """
        Initialize the DroughtComponent object.

        Parameters
        ----------
        precipitation_source : str
            Path to a directory containing NetCDF files or a single NetCDF file.
        mask_path : str, optional
            Path to the dataset containing mask data. Default is None.

        Complexity
        ----------
        O(P) for loading and initializing precipitation and mask data,
        where P is the size of the precipitation dataset.
        """
        # Determine if the source is a directory or a single file
        if os.path.isdir(precipitation_source):
            # Load multiple NetCDF files using open_mfdataset
            precipitation = xr.open_mfdataset(
                os.path.join(precipitation_source, "*.nc"), combine='by_coords'
            )
        else:
            # Load a single NetCDF file
            precipitation = xr.open_dataset(precipitation_source)

        # Load mask data if provided
        mask = None
        if mask_path:
            mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})

        super().__init__(precipitation, mask, precipitation_source)

    def max_consecutive_dry_days(self):
        """
        Calculate the maximum number of consecutive dry days in each year.

        Returns
        -------
        xarray.DataArray
            Maximum number of consecutive dry days.

        Complexity
        ----------
        O(N) for calculating cumulative sums and transformations,
        where N is the number of time steps in the dataset.
        """
        preci = self.apply_mask("tp") if self.mask is not None else self.array
        precipitation_per_day = preci['tp'].resample(time='d').sum()

        # Rechunk data after resampling for optimal performance
        
        days_below_thresholds = xr.where(precipitation_per_day < 0.001, 1, 0)
        days_above_thresholds = xr.where(days_below_thresholds == 0, 1, 0)
        cumsum_above = days_above_thresholds.cumsum(dim='time')
        days = cumsum_above - cumsum_above.where(days_above_thresholds == 0).ffill(dim='time').fillna(0)
        result = days.resample(time='Y').max()

        
        return result

    def drought_interpolate(self, max_days_drought_per_year):
        """
        Perform linear interpolation of the maximum number of consecutive dry days (CDD)
        to obtain monthly values from annual values.

        Parameters
        ----------
        max_days_drought_per_year : xarray.DataArray
            The maximum number of consecutive dry days per year.

        Returns
        -------
        xarray.DataArray
            Interpolated monthly CDD values.

        Complexity
        ----------
        O(Y * M) where Y is the number of years and M is the number of months,
        as interpolation is done for each month of each year.
        """
        monthly_values = []
        years = pd.to_datetime(max_days_drought_per_year.time.values).year

        for i in range(len(years) - 1):
            cdd_k = max_days_drought_per_year.isel(time=i)
            cdd_k_plus_1 = max_days_drought_per_year.isel(time=i + 1)

            for month in range(1, 13):
                weight1 = (12 - month) / 12
                weight2 = month / 12
                interpolated_value = weight1 * cdd_k + weight2 * cdd_k_plus_1
                monthly_time = np.datetime64(f"{years[i]}-{month:02d}-01")
                interpolated_value = interpolated_value.expand_dims("time")
                interpolated_value["time"] = [monthly_time]
                monthly_values.append(interpolated_value)

        # Handle the last year by repeating the values of the last available year
        cdd_last = max_days_drought_per_year.isel(time=-1)
        for month in range(1, 13):
            monthly_time = np.datetime64(f"{years[-1]}-{month:02d}-01")
            repeated_value = cdd_last.copy()
            repeated_value = repeated_value.expand_dims("time")
            repeated_value["time"] = [monthly_time]
            monthly_values.append(repeated_value)

        monthly_values = xr.concat(monthly_values, dim="time")

    

        return monthly_values

    def calculate_component(self, reference_period, area=None):
        """
        Standardize the maximum number of consecutive dry days.

        Parameters
        ----------
        reference_period : tuple
            A tuple containing the start and end dates of the reference
            period (e.g., ('1961-01-01', '1990-12-31')).
        area : bool, optional
            If True, calculate the area-averaged standardized metric.
            Default is None.

        Returns
        -------
        xarray.DataArray
            Standardized maximum number of consecutive dry days.

        Complexity
        ----------
        O(N + R) for calculating maximum consecutive dry days and
        standardizing, where N is the number of time steps and R is the
        size of the reference period.
        """
        max_days_drought_per_year = self.max_consecutive_dry_days()


        monthly_values = self.drought_interpolate(max_days_drought_per_year)
        # Standardize the interpolated monthly values
        standardized_values = self.standardize_metric(monthly_values, reference_period, area)


        return standardized_values