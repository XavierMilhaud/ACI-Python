import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class WindComponent:
    """
    Class to process wind data and calculate wind power and wind thresholds.

    Attributes:
    - u10 (xarray.Dataset): Dataset containing wind u-component data.
    - v10 (xarray.Dataset): Dataset containing wind v-component data.
    - mask (xarray.Dataset): Dataset containing mask data.
    - wind_power (xarray.DataArray): Daily wind power calculated from u10 and v10.
    """

    def __init__(self, u10_path, v10_path, mask_path):
        """
        Initialize the WindComponent object.

        Parameters:
        - u10_path (str): Path to the dataset containing wind u-component data.
        - v10_path (str): Path to the dataset containing wind v-component data.
        - mask_path (str): Path to the dataset containing mask data.
        """
        self.u10 = xr.open_dataset(u10_path)
        self.v10 = xr.open_dataset(v10_path)
        self.mask = xr.open_dataset(mask_path).rename({'lon':'longitude', 'lat': 'latitude'})

    def apply_mask(self, temp: xr.Dataset, var_name: str, mask: xr.Dataset) -> xr.Dataset:
        """
        Apply a mask to the wind data.

        Parameters:
        - temp (xarray.Dataset): Dataset containing wind data.
        - var_name (str): Variable name to be masked.
        - mask (xarray.Dataset): Dataset containing mask data.

        Returns:
        - xarray.Dataset: Wind data with mask applied.
        """
        f_temp = temp.copy()
        f_temp['mask'] = mask.country
        threshold = 0.8
        country_mask = f_temp['mask'] >= threshold
        f_temp[var_name] = xr.where(country_mask, f_temp[var_name], float('nan'))
        return f_temp.drop_vars('mask')

    def calculate_wind_power(self, reference_period=None) -> xr.DataArray:
        """
        Calculate daily wind power from wind u and v components.

        Args:
        reference_period (tuple): if not None, the method computes daily wind power for the reference period (e.g : ('1960-01-01', '1961-12-31'))

        Returns:
        - xarray.DataArray: Daily wind power.
        """
        u10 = self.apply_mask(self.u10, "u10", self.mask)
        v10 = self.apply_mask(self.v10, "v10", self.mask)
        ws = np.sqrt(u10.u10**2 + v10.v10**2)
        rho = 1.23  # Air density constant
        dailymean_ws = ws.resample(time='D').mean()
        wind_power = 0.5 * rho * dailymean_ws**3

        if reference_period:
            return wind_power.sel(time=slice(reference_period[0], reference_period[1]))
        else:
            return wind_power

    def calculate_wind_thresholds(self, reference_period) -> xr.DataArray:
        """
        Calculate wind power thresholds based on the 90th percentile.

        Returns:
        - xarray.DataArray: Wind power thresholds.
        """
        wind_power = self.calculate_wind_power()
        wind_power_reference = wind_power.sel(time=slice(reference_period[0], reference_period[1]))
        time_index = wind_power.time.dt.dayofyear
        dset_mean = wind_power_reference.groupby("time.dayofyear").mean().sel(dayofyear=time_index)
        dset_std = wind_power_reference.groupby("time.dayofyear").std().sel(dayofyear=time_index)
        wind_power_thresholds = (dset_mean + 1.28 * dset_std).drop("dayofyear")
        return wind_power_thresholds

    def calculate_days_above_thresholds(self, reference_period) -> xr.DataArray:
        """
        Calculate the days above thresholds.

        Returns:
        - xarray.DataArray: Days above thresholds.
        """
        wind_power_thresholds = self.calculate_wind_thresholds(reference_period)
        wind_power = self.calculate_wind_power()
        diff_array = wind_power_thresholds - wind_power
        days_above_thresholds = xr.where(diff_array<0, 1, 0)
        return days_above_thresholds

    def calculate_wind_exceedance_frequency(self, reference_period) -> xr.DataArray:
        """
        Calculate the frequency of daily mean wind power above the 90th percentile.

        Returns:
        - xarray.DataArray: Wind exceedance frequency.
        """
        days_above_thresholds = self.calculate_days_above_thresholds(reference_period)
        monthly_total_days_above = days_above_thresholds.resample(time='m').sum() / days_above_thresholds.resample(time="m").count()
        return monthly_total_days_above

    def standardize_wind_exceedance_frequency(self, reference_period, area=None) -> xr.DataArray:
        """
        Standardize the wind exceedance frequency.

        Returns:
        - xarray.DataArray: Standardized wind exceedance frequency.
        """
        monthly_total_days_above = self.calculate_wind_exceedance_frequency(reference_period)
        monthly_total_days_above_reference = monthly_total_days_above.sel(time=slice(reference_period[0], reference_period[1]))
        time_index = monthly_total_days_above.time.dt.month
        wp90_mean = monthly_total_days_above_reference.groupby("time.month").mean().sel(month=time_index)
        wp90_std = monthly_total_days_above_reference.groupby("time.month").std().sel(month=time_index)
        w_std = ((monthly_total_days_above - wp90_mean) / wp90_std).drop("month")

        if area:
            return w_std.mean(dim=['latitude', 'longitude'])
        else:
            return w_std

    def standardize_seasonly_wind_exceedance_frequency(self, reference_period, area=None) -> xr.DataArray:
        """
        Standardize the seasonal wind exceedance frequency.

        Returns:
        - xarray.DataArray: Standardized seasonal wind exceedance frequency.
        """
        days_above_thresholds = self.calculate_days_above_thresholds(reference_period)
        seasonly_total_days_above = days_above_thresholds.resample(time='QS-DEC').sum() / days_above_thresholds.resample(time='QS-DEC').count()
        seasonly_total_days_above_reference = seasonly_total_days_above.sel(time=slice(reference_period[0], reference_period[1]))
        time_index_seasonly = seasonly_total_days_above.time.dt.month
        wp90_mean_season = seasonly_total_days_above_reference.groupby("time.month").mean().sel(month=time_index_seasonly)
        wp90_std_season = seasonly_total_days_above_reference.groupby("time.month").std().sel(month=time_index_seasonly)
        w_std_season = ((seasonly_total_days_above - wp90_mean_season) / wp90_std_season).drop("month")

        if area:
            return w_std_season.mean(dim=['latitude', 'longitude'])
        else:
            return w_std_season

