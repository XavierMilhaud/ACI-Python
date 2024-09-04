import xarray as xr
from components.component import Component


class PrecipitationComponent(Component):
    """
    A class to handle precipitation data and perform related calculations.

    Attributes:
        precipitation (xarray.Dataset): The dataset containing the precipitation data.
        mask (xarray.Dataset): The dataset containing the mask data.
    """

    def __init__(self, precipitation_path, mask_path):
        """
        Initializes the PrecipitationComponent with precipitation and mask data.

        Args:
            precipitation_path (str): The file path of the precipitation data.
            mask_path (str): The file path of the mask data.

        Complexity:
        O(P) for loading and initializing precipitation and mask data, where P is the size of the precipitation dataset.
        """
        precipitation = xr.open_dataset(precipitation_path)
        mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})
        super().__init__(precipitation, mask, precipitation_path)

    def calculate_maximum_precipitation_over_window(self, var_name:str='tp', window_size:int=5, season:bool=False):
        """
        Calculates the maximum monthly precipitation over a specified window size.

        Args:
            var_name (str): The variable name in the precipitation data to calculate the monthly maximum.
            window_size (int): The size of the rolling window in days.

        Returns:
            xarray.DataArray: The maximum monthly precipitation.

        Complexity:
        O(N) for calculating rolling sum and resampling, where N is the number of time steps in the dataset.
        """
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        if season :
            period = 'QS-DEC'
        else :
            period = 'M'
        period_max = rolling_sum.resample(time=period).max()
        return period_max

    def calculate_component(self, reference_period, area=None, var_name:str='tp', window_size:int=5, season:bool=False):
        """
        Calculates the anomaly of maximum monthly precipitation relative to a reference period.

        Args:
            var_name (str): The variable name in the precipitation data.
            window_size (int): The size of the rolling window in days.
            reference_period (tuple): A tuple containing the start and end dates of
            the reference period (e.g., ('1961-01-01', '1989-12-31')).
            area (bool): If True, calculate the area-averaged anomaly. Default is None.

        Returns:
            xarray.DataArray: The anomaly of maximum monthly precipitation.

        Complexity:
        O(N + R) for calculating monthly maximum and standardizing, where N is the number
        of time steps and R is the size of the reference period.
        """
        period_max = self.calculate_maximum_precipitation_over_window(var_name, window_size, season)
        return self.standardize_metric(period_max, reference_period, area)
