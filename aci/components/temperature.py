import numpy as np
import xarray as xr
from aci.components.component import Component


class TemperatureComponent(Component):
    """
    A class to handle temperature data and perform related calculations.

    Attributes:
        temperature_data (xarray.Dataset): Dataset containing temperature data.
        mask_data (xarray.Dataset): Dataset containing mask data.
    """

    def __init__(self, temperature_data_path:str, mask_data_path:str,
    percentile:float, extremum:str, above_thresholds:bool=True):
        """
        Initialize the TemperatureComponent object.

        Parameters:
        - temperature_data_path (str): Path to the dataset containing temperature data.
        - mask_data_path (str): Path to the dataset containing mask data.
        - percentile (float): percentile chosen for the thresholds.
        - extremum (str): specifies whether to find 'min' or 'max' temperature.
        - above_thresholds (bool): if True counts the values above the percentile, if False under the thresholds.

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

        self.percentile = percentile
        self.extremum = extremum
        self.above_thresholds = above_thresholds

    def temp_extremum(self, extremum, period):
        """
        Compute daily min or max temperature for days or nights.

        Parameters:
        - extremum (str): 'min' or 'max' to compute minimum or maximum temperatures.
        - period (str): 'day' or 'night' to specify the time period.

        Returns:
        - xarray.DataArray: Daily min or max temperatures.
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

    def calculate_percentiles(self, n, reference_period, tempo):
        """
        Compute percentiles for day or night temperatures over a reference period.

        Parameters:
        - n (int): Percentile to compute (e.g., 90 for 90th percentile).
        - reference_period (tuple): Start and end dates of the reference period.
        - tempo (str): 'day' or 'night' to specify the time period.

        Returns:
        - xarray.DataArray: Percentiles for each day of the year.
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

    def calculate_halfday_component(self, reference_period, part_of_day:str):
        """
        Calculates the halfday component of the temperature

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.
        - part_of_day (str) : 'day' or 'night' to specify the time period.

        Returns:
        - xarray.DataArray: daily or nightly component for each month of the year.
        """
        temperature_halfday_extremum = self.temp_extremum(self.extremum,part_of_day)

        temperature_percentile_halfday = self.calculate_percentiles(self.percentile, reference_period, part_of_day)

        time_index = temperature_halfday_extremum["time"].dt.dayofyear

        difference_between_current_and_reference_period_percentile = (temperature_halfday_extremum -
                                      temperature_percentile_halfday.sel(dayofyear=time_index)).drop_vars("dayofyear")
        
        if self.above_thresholds:
            halfday_crossing_threshold = xr.where(difference_between_current_and_reference_period_percentile > 0, 1, 0)
        else :
            halfday_crossing_threshold = xr.where(difference_between_current_and_reference_period_percentile < 0, 1, 0)

        halfday_component = halfday_crossing_threshold.resample(time='ME').sum() / halfday_crossing_threshold.resample(time="ME").count()

        return halfday_component

    def calculate_component(self, reference_period, area=None):
        """
        Calculates the temperature component.

        Parameters:
        - reference_period (tuple): Start and end dates of the reference period.
        - area (bool): If True, calculate the area-averaged anomaly. Default is None.

        Returns:
        - xarray.DataArray: temperature component for each month of the year.
        """
        day_component = self.calculate_halfday_component(reference_period, 'day')

        night_component = self.calculate_halfday_component(reference_period, 'night')

        component = 0.5 * (day_component + night_component)

        component_standardized = self.standardize_metric(component, reference_period, area)

        return component_standardized


