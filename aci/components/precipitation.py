import xarray as xr
import os
from components.component import Component


class PrecipitationComponent(Component):
    """
    A class to handle precipitation data and perform related calculations.

    Attributes:
    ----------
    precipitation : xarray.Dataset
        Dataset containing precipitation data.
    mask : xarray.Dataset or None
        Dataset containing mask data, if provided.
    """

    def __init__(self, precipitation_source, mask_path=None):
        """
        Initialize the PrecipitationComponent object.

        Parameters:
        ----------
        precipitation_source : str
            Path to a directory containing NetCDF files or a single NetCDF file.
        mask_path : str, optional
            Path to the dataset containing mask data. Default is None.

        Complexity:
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

    def monthly_max(self, var_name, window_size):
        """
        Calculates the maximum monthly precipitation over a specified window size.

        Parameters:
        ----------
        var_name : str
            The variable name in the precipitation data to calculate
            the monthly maximum.
        window_size : int
            The size of the rolling window in days.

        Returns:
        -------
        xarray.DataArray
            The maximum monthly precipitation.

        Complexity:
        ----------
        O(N) for calculating rolling sum and resampling, where N is the
        number of time steps in the dataset.
        """
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        monthly_max = rolling_sum.resample(time='M').max()

        if not self.should_use_dask:
            monthly_max = monthly_max.compute()

        return monthly_max

    def monthly_max_anomaly(self, var_name, window_size, reference_period, area=None):
        """
        Calculates the anomaly of maximum monthly precipitation relative to a reference period.

        Parameters:
        ----------
        var_name : str
            The variable name in the precipitation data.
        window_size : int
            The size of the rolling window in days.
        reference_period : tuple
            A tuple containing the start and end dates of the reference
            period (e.g., ('1961-01-01', '1989-12-31')).
        area : bool, optional
            If True, calculate the area-averaged anomaly. Default is None.

        Returns:
        -------
        xarray.DataArray
            The anomaly of maximum monthly precipitation.

        Complexity:
        ----------
        O(N + R) for calculating monthly maximum and standardizing, where
        N is the number of time steps and R is the size of the reference
        period.
        """
        monthly_max = self.monthly_max(var_name, window_size)
        return self.standardize_metric(monthly_max, reference_period, area)

    def seasonly_max(self, var_name, window_size):
        """
        Calculates the maximum seasonal precipitation over a specified window size.

        Parameters:
        ----------
        var_name : str
            The variable name in the precipitation data to calculate the
            seasonal maximum.
        window_size : int
            The size of the rolling window in days.

        Returns:
        -------
        xarray.DataArray
            The maximum seasonal precipitation.

        Complexity:
        ----------
        O(N) for calculating rolling sum and resampling, where N is the
        number of time steps in the dataset.
        """
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        seasonly_max = rolling_sum.resample(time='QS-DEC').max()

        if not self.should_use_dask:
            seasonly_max = seasonly_max.compute()

        return seasonly_max

    def seasonly_max_anomaly(self, var_name, window_size, reference_period, area=None):
        """
        Calculates the anomaly of maximum seasonal precipitation relative to a reference period.

        Parameters:
        ----------
        var_name : str
            The variable name in the precipitation data.
        window_size : int
            The size of the rolling window in days.
        reference_period : tuple
            A tuple containing the start and end dates of the reference
            period (e.g., ('1961-01-01', '1989-12-31')).
        area : bool, optional
            If True, calculate the area-averaged anomaly. Default is None.

        Returns:
        -------
        xarray.DataArray
            The anomaly of maximum seasonal precipitation.

        Complexity:
        ----------
        O(N + R) for calculating seasonal maximum and standardizing,
        where N is the number of time steps and R is the size of the
        reference period.
        """
        seasonly_max = self.seasonly_max(var_name, window_size)
        return self.standardize_metric(seasonly_max, reference_period, area)
