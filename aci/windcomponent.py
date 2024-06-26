import xarray as xr
import numpy as np
from component import Component


class WindComponent(Component):
    """
    Class to process wind data and calculate wind power and wind thresholds.

    Attributes:
    - u10 (xarray.Dataset): Dataset containing wind u-component data.
    - v10 (xarray.Dataset): Dataset containing wind v-component data.
    - mask (xarray.Dataset): Dataset containing mask data.
    """

    def __init__(self, u10_path, v10_path, mask_path):
        """
        Initialize the WindComponent object.

        Parameters:
        - u10_path (str): Path to the dataset containing wind u-component data.
        - v10_path (str): Path to the dataset containing wind v-component data.
        - mask_path (str): Path to the dataset containing mask data.
        """
        u10 = xr.open_dataset(u10_path)
        v10 = xr.open_dataset(v10_path)
        mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})
        super().__init__(u10, mask, u10_path)
        self.v10 = v10

    def wind_power(self, reference_period=None):
        """
        Calculate daily wind power from wind u and v components.

        Args:
        reference_period (tuple): if not None, the method computes daily wind power for the
        reference period (e.g : ('1960-01-01', '1961-12-31'))

        Returns:
        - xarray.DataArray: Daily wind power.
        """
        u10 = self.apply_mask("u10")
        temp = self.array
        self.array = self.v10
        v10 = self.apply_mask("v10")
        self.array = temp

        ws = np.sqrt(u10.u10**2 + v10.v10**2)
        rho = 1.23  # Air density constant
        dailymean_ws = ws.resample(time='D').mean()
        wind_power = 0.5 * rho * dailymean_ws**3

        if reference_period:
            return wind_power.sel(time=slice(reference_period[0], reference_period[1]))
        else:
            return wind_power

    def wind_thresholds(self, reference_period):
        """
        Calculate wind power thresholds based on the 90th percentile.

        Args:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.

        Returns:
        - xarray.DataArray: Wind power thresholds.
        """
        wind_power = self.wind_power()
        wind_power_reference = wind_power.sel(time=slice(reference_period[0], reference_period[1]))
        time_index = wind_power.time.dt.dayofyear

        dset_mean = wind_power_reference.groupby("time.dayofyear").mean().sel(dayofyear=time_index)
        dset_std = wind_power_reference.groupby("time.dayofyear").std().sel(dayofyear=time_index)
        wind_power_thresholds = (dset_mean + 1.28 * dset_std).drop("dayofyear")
        return wind_power_thresholds

    def days_above_thresholds(self, reference_period):
        """
        Calculate the days above thresholds.

        Args:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.

        Returns:
        - xarray.DataArray: Days above thresholds.
        """
        wind_power_thresholds = self.wind_thresholds(reference_period)
        wind_power = self.wind_power()
        diff_array = wind_power_thresholds - wind_power
        days_above_thresholds = xr.where(diff_array < 0, 1, 0)
        return days_above_thresholds

    def wind_exceedance_frequency(self, reference_period):
        """
        Calculate the frequency of daily mean wind power above the 90th percentile.

        Args:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.

        Returns:
        - xarray.DataArray: Wind exceedance frequency.
        """
        days_above_thresholds = self.days_above_thresholds(reference_period)
        monthly_total_days_above = days_above_thresholds.resample(time='m').sum() / days_above_thresholds.resample(time="m").count()
        return monthly_total_days_above

    def std_wind_exceedance_frequency(self, reference_period, area=None):
        """
        Standardize the wind exceedance frequency.

        Parameters:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.
        - area (bool): If True, calculate the area-averaged standardized metric. Default is None.

        Returns:
        - xarray.DataArray: Standardized wind exceedance frequency.
        """
        monthly_total_days_above = self.wind_exceedance_frequency(reference_period)
        return self.standardize_metric(monthly_total_days_above, reference_period, area)

    def std_seasonly_wind_exceedance_frequency(self, reference_period, area=None):
        """
        Standardize the seasonal wind exceedance frequency.

        Parameters:
        - reference_period (tuple): A tuple containing the start and end dates of the reference period.
        - area (bool): If True, calculate the area-averaged standardized metric. Default is None.

        Returns:
        - xarray.DataArray: Standardized seasonal wind exceedance frequency.
        """
        days_above_thresholds = self.days_above_thresholds(reference_period)
        seasonly_total_days_above = days_above_thresholds.resample(time='QS-DEC').sum() / days_above_thresholds.resample(time='QS-DEC').count()
        return self.standardize_metric(seasonly_total_days_above, reference_period, area)
