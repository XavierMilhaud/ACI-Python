import numpy as np
import xarray as xr
from components.component import Component


class TemperatureComponent(Component):
    """
    A class to handle temperature data and perform related calculations.

    Attributes:
        temperature_data (xarray.Dataset): Dataset containing temperature data.
        mask_data (xarray.Dataset): Dataset containing mask data.
    """

    def __init__(self, temperature_data_path, mask_data_path):
        """
        Initialize the TemperatureComponent object.

        Parameters:
        - temperature_data_path (str): Path to the dataset containing temperature data.
        - mask_data_path (str): Path to the dataset containing mask data.

        Complexity:
        O(T) for loading and initializing temperature and mask data, where T is the size of the temperature dataset.
        """
        temperature_data = xr.open_dataset(temperature_data_path)
        mask_data = xr.open_dataset(mask_data_path).rename({'lon': 'longitude', 'lat': 'latitude'})
        super().__init__(temperature_data, mask_data, temperature_data_path)
        temperature = self.apply_mask("t2m")
        self.temperature_days = temperature.isel(
            time=temperature.time.dt.hour.isin([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        )
        self.temperature_nights = temperature.isel(
            time=temperature.time.dt.hour.isin([0, 1, 2, 3, 4, 5, 22, 23])
        )

    def temp_extremum(self, extremum, period):
        """
        Compute daily min or max temperature for days or nights.

        Parameters:
        - extremum (str): 'min' or 'max' to compute minimum or maximum temperatures.
        - period (str): 'day' or 'night' to specify the time period.

        Returns:
        - xarray.DataArray: Daily min or max temperatures.

        Complexity:
        O(N), where N is the number of time steps in the selected period.
        """
        if period == "day":
            temperature = self.temperature_days
        elif period == "night":
            temperature = self.temperature_nights
        else:
            raise ValueError("period must be 'day' or 'night'")

        if extremum == "min":
            return temperature.resample(time='D').min()
        elif extremum == "max":
            return temperature.resample(time='D').max()
        else:
            raise ValueError("extremum must be 'min' or 'max'")

    def percentiles(self, n, reference_period, tempo):
        """
        Compute percentiles for day or night temperatures over a reference period.

        Parameters:
        - n (int): Percentile to compute (e.g., 90 for 90th percentile).
        - reference_period (tuple): Start and end dates of the reference period.
        - tempo (str): 'day' or 'night' to specify the time period.

        Returns:
        - xarray.DataArray: Percentiles for each day of the year.

        Complexity:
        O(N), where N is the number of time steps in the reference period.
        """
        if tempo == "day":
            rolling_window_size = 80
            temperature_reference = self.temperature_days.sel(
                time=slice(reference_period[0], reference_period[1])
            )
        elif tempo == "night":
            rolling_window_size = 40
            temperature_reference = self.temperature_nights.sel(
                time=slice(reference_period[0], reference_period[1])
            )
        else:
            raise ValueError("tempo must be 'day' or 'night'")

        percentile_reference = temperature_reference['t2m'].rolling(
            time=rolling_window_size, min_periods=1, center=True).reduce(np.percentile, q=n)
        percentile_calendar = percentile_reference.groupby('time.dayofyear').reduce(np.percentile, q=n)
        return percentile_calendar

    def t90(self, reference_period):
        """
        Compute T90 (90th percentile threshold exceedance) for days and nights.

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.

        Returns:
        - xarray.DataArray: T90 values.

        Complexity:
        O(N), where N is the number of time steps in the study period.
        """
        temperature_days_max = self.temp_extremum("max", "day")
        percentile_90_calendar_days = self.percentiles(90, reference_period, "day")

        time_index = temperature_days_max["time"].dt.dayofyear

        difference_array_90_days = (temperature_days_max -
                                    percentile_90_calendar_days.sel(dayofyear=time_index)).drop("dayofyear")

        days_90_above_thresholds = xr.where(difference_array_90_days > 0, 1, 0)
        tx90 = days_90_above_thresholds.resample(time='m').sum() / days_90_above_thresholds.resample(time="m").count()

        temperature_nights_max = self.temp_extremum("max", "night")
        percentile_90_calendar_nights = self.percentiles(90, reference_period, "night")

        time_index = temperature_nights_max["time"].dt.dayofyear

        difference_array_90_nights = (temperature_nights_max -
                                      percentile_90_calendar_nights.sel(dayofyear=time_index)).drop("dayofyear")

        nights_90_above_thresholds = xr.where(difference_array_90_nights > 0, 1, 0)
        tn90 = nights_90_above_thresholds.resample(time='m').sum() / nights_90_above_thresholds.resample(time="m").count()

        return 0.5 * (tx90 + tn90)

    def t10(self, reference_period):
        """
        Compute T10 (10th percentile threshold exceedance) for days and nights.

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.

        Returns:
        - xarray.DataArray: T10 values.

        Complexity:
        O(N), where N is the number of time steps in the study period.
        """
        temperature_days_min = self.temp_extremum("min", "day")
        percentile_10_calendar_days = self.percentiles(10, reference_period, "day")

        time_index = temperature_days_min["time"].dt.dayofyear

        difference_array_10_days = (temperature_days_min -
                                    percentile_10_calendar_days.sel(dayofyear=time_index)).drop("dayofyear")

        days_10_above_thresholds = xr.where(difference_array_10_days < 0, 1, 0)
        tx10 = days_10_above_thresholds.resample(time='m').sum() / days_10_above_thresholds.resample(time="m").count()

        temperature_nights_min = self.temp_extremum("min", "night")
        percentile_10_calendar_nights = self.percentiles(10, reference_period, "night")

        time_index = temperature_nights_min["time"].dt.dayofyear

        difference_array_10_nights = (temperature_nights_min -
                                      percentile_10_calendar_nights.sel(dayofyear=time_index)).drop("dayofyear")

        nights_10_above_thresholds = xr.where(difference_array_10_nights < 0, 1, 0)
        tn10 = nights_10_above_thresholds.resample(time='m').sum() / nights_10_above_thresholds.resample(time="m").count()

        return 0.5 * (tx10 + tn10)

    def std_t90(self, reference_period, area=None):
        """
        Standardize T90 values.

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.
        - area (bool): Whether to compute the area-averaged standard deviation.

        Returns:
        - xarray.DataArray: Standardized T90 values.

        Complexity:
        O(N), where N is the number of time steps in the study period.
        """
        t_90 = self.t90(reference_period)
        return self.standardize_metric(t_90, reference_period, area)

    def std_t10(self, reference_period, area=None):
        """
        Standardize T10 values.

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.
        - area (bool): Whether to compute the area-averaged standard deviation.

        Returns:
        - xarray.DataArray: Standardized T10 values.

        Complexity:
        O(N), where N is the number of time steps in the study period.
        """
        t10 = self.t10(reference_period)
        return self.standardize_metric(t10, reference_period, area)

    def plot_components(self, component_data0, component_data1, n):
        """
        Plot rolling mean of temperature components.

        Parameters:
        - component_data0 (xarray.DataArray): First temperature component data.
        - component_data1 (xarray.DataArray): Second temperature component data.
        - n (int): Rolling window size.

        Complexity:
        O(N), where N is the number of time steps in the component data.
        """
        component_data0["t2m"].rolling(time=n, center=True).mean().plot()
        component_data1["t2m"].rolling(time=n, center=True).mean().plot()
