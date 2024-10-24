import xarray as xr
from aci.components.component import Component


class PrecipitationComponent(Component):
    """
    A class to handle precipitation data and perform related calculations.

    Attributes:
        precipitation (xarray.Dataset): The dataset containing the precipitation data.
        mask (xarray.Dataset): The dataset containing the mask data.
    """

    def __init__(self, precipitation_data_path, mask_path=None):
        """
        Initializes the PrecipitationComponent with precipitation and mask data.

        Parameters:
        - precipitation_path (str): The file path of the precipitation data.
        - mask_path (str): The file path of the mask data.
        """
        super().__init__(precipitation_data_path, mask_path,var_name='tp')

    def calculate_maximum_precipitation_over_window(self, var_name:str='tp', window_size:int=5, season:bool=False):
        """
        Calculates the maximum monthly precipitation over a specified window size.

        Parameters :
        - var_name (str): The variable name in the precipitation data to calculate the monthly maximum.
        - window_size (int): The size of the rolling window in days.
        - season (bool): If True calculate the maximum precipitation over season and not monthly, default to False

        Returns:
            xarray.DataArray: The maximum monthly precipitation.

        """
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        if season :
            period = 'QS-DEC'
        else :
            period = 'ME'
        period_max = rolling_sum.resample(time=period).max()
        return period_max

    def calculate_component(self, reference_period, area=None, var_name:str='tp', window_size:int=5, season:bool=False):
        """
        Calculates the anomaly of maximum monthly precipitation relative to a reference period.

        Parameters :
        - var_name (str): The variable name in the precipitation data.
        - window_size (int): The size of the rolling window in days.
        - reference_period (tuple): A tuple containing the start and end dates of
        - the reference period (e.g., ('1961-01-01', '1989-12-31')).
        - area (bool): If True, calculate the area-averaged anomaly. Default is None.

        Returns:
            xarray.DataArray: The anomaly of maximum monthly precipitation.

        """
        period_max = self.calculate_maximum_precipitation_over_window(var_name, window_size, season)
        return self.standardize_metric(period_max, reference_period, area)
