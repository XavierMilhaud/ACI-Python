import numpy as np
import xarray as xr
import os
from component import Component


class TemperatureComponent(Component):
    """
    A class to handle temperature data and perform related calculations.

    Attributes:
        temperature_data (xarray.Dataset): Dataset containing temperature
        data.
        mask_data (xarray.Dataset or None): Dataset containing mask data,
        if provided.
    """

    def __init__(self, temperature_data_source, mask_data_path=None):
        """
        Initialize the TemperatureComponent object.

        Parameters:
        ----------
        temperature_data_source : str
            Path to a directory containing NetCDF files or a single NetCDF
            file for temperature data.
        mask_data_path : str, optional
            Path to the dataset containing mask data. Default is None.

        Complexity:
        ----------
        O(T) for loading and initializing temperature and mask data, where T
        is the size of the temperature dataset.
        """
        # Determine if the source is a directory or a single file
        if os.path.isdir(temperature_data_source):
            # Load multiple NetCDF files using open_mfdataset
            temperature_data = xr.open_mfdataset(
                os.path.join(
                    temperature_data_source, "*.nc"), combine='by_coords'
            )
        else:
            # Load a single NetCDF file
            temperature_data = xr.open_dataset(temperature_data_source)

        # Load mask data if provided
        mask_data = None
        if mask_data_path:
            mask_data = xr.open_dataset(
                mask_data_path).rename({'lon': 'longitude', 'lat': 'latitude'})

        super().__init__(temperature_data, mask_data, temperature_data_source)

        temperature = self.apply_mask("t2m")
        self.temperature_days = temperature.isel(
            time=temperature.time.dt.hour.isin(
                [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            )
        )
        self.temperature_nights = temperature.isel(
            time=temperature.time.dt.hour.isin([0, 1, 2, 3, 4, 5, 22, 23])
        )

    def temp_extremum(self, extremum, period):
        """
        Compute daily min or max temperature for days or nights.

        Parameters:
        ----------
        extremum : str
            'min' or 'max' to compute minimum or maximum temperatures.
        period : str
            'day' or 'night' to specify the time period.

        Returns:
        -------
        xarray.DataArray
            Daily min or max temperatures.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the selected period.
        """
        if period == "day":
            temperature = self.temperature_days
        elif period == "night":
            temperature = self.temperature_nights
        else:
            raise ValueError("period must be 'day' or 'night'")

        if extremum == "min":
            return temperature.resample(time="D").min()
        elif extremum == "max":
            return temperature.resample(time="D").max()
        else:
            raise ValueError("extremum must be 'min' or 'max'")

    def percentiles(self, n, reference_period, tempo):
        """
        Compute percentiles for day or night temperatures over a reference
        period.

        Parameters:
        ----------
        n : int
            Percentile to compute (e.g., 90 for 90th percentile).
        reference_period : tuple
            Start and end dates of the reference period.
        tempo : str
            'day' or 'night' to specify the time period.

        Returns:
        -------
        xarray.DataArray
            Percentiles for each day of the year.

        Complexity:
        ----------
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

        if self.should_use_dask:
            # Use Dask for parallel computation
            temperature_reference = temperature_reference.chunk(
                {'time': -1})

            def compute_percentile(arr, q):
                return np.percentile(arr, q, axis=-1)

            rolling = temperature_reference["t2m"].rolling(
                time=rolling_window_size, min_periods=1, center=True)
            rolling_constructed = rolling.construct('window_dim')
            rolling_constructed = rolling_constructed.chunk(
                {'time': -1})

            percentile_reference = xr.apply_ufunc(
                compute_percentile,
                rolling_constructed,
                input_core_dims=[['window_dim']],
                kwargs={'q': n},
                dask='parallelized',
                output_dtypes=[float]
            )

            percentile_calendar = xr.apply_ufunc(
                compute_percentile,
                percentile_reference.groupby("time.dayofyear"),
                input_core_dims=[['time']],
                kwargs={'q': n},
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float]
            )
        else:
            # Non-Dask computation
            rolling = temperature_reference["t2m"].rolling(
                time=rolling_window_size, min_periods=1, center=True)
            rolling_constructed = rolling.construct('window_dim')
            percentile_reference = rolling_constructed.reduce(
                lambda arr, axis: np.percentile(arr, n, axis=axis),
                dim='window_dim'
            )

            percentile_calendar = percentile_reference.groupby("time.dayofyear").reduce(
                lambda arr, axis: np.percentile(arr, n, axis=axis),
                dim='time'
            )

        return percentile_calendar

    def t90(self, reference_period):
        """
        Compute T90 (90th percentile threshold exceedance) for days
        and nights.

        Parameters:
        ----------
        reference_period : tuple
            Start and end dates of the reference period.

        Returns:
        -------
        xarray.DataArray
            T90 values.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the study period.
        """
        temperature_days_max = self.temp_extremum("max", "day")
        percentile_90_calendar_days = self.percentiles(
            90, reference_period, "day"
        )

        time_index = temperature_days_max["time"].dt.dayofyear
        difference_array_90_days = (
            temperature_days_max
            - percentile_90_calendar_days.sel(dayofyear=time_index)
        ).drop("dayofyear")

        days_90_above_thresholds = xr.where(difference_array_90_days > 0, 1, 0)
        tx90 = (
            days_90_above_thresholds.resample(time="m").sum()
            / days_90_above_thresholds.resample(time="m").count()
        )

        temperature_nights_max = self.temp_extremum("max", "night")
        percentile_90_calendar_nights = self.percentiles(
            90, reference_period, "night"
        )

        time_index = temperature_nights_max["time"].dt.dayofyear
        difference_array_90_nights = (
            temperature_nights_max
            - percentile_90_calendar_nights.sel(dayofyear=time_index)
        ).drop("dayofyear")

        nights_90_above_thresholds = xr.where(difference_array_90_nights > 0, 1, 0)
        tn90 = (
            nights_90_above_thresholds.resample(time="m").sum()
            / nights_90_above_thresholds.resample(time="m").count()
        )

        t90_values = 0.5 * (tx90 + tn90)

        if not self.should_use_dask:
            t90_values = t90_values.compute()

        return t90_values

    def t10(self, reference_period):
        """
        Compute T10 (10th percentile threshold exceedance) for days
        and nights.

        Parameters:
        ----------
        reference_period : tuple
            Start and end dates of the reference period.

        Returns:
        -------
        xarray.DataArray
            T10 values.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the study period.
        """
        temperature_days_min = self.temp_extremum("min", "day")
        percentile_10_calendar_days = self.percentiles(
            10, reference_period, "day"
        )

        time_index = temperature_days_min["time"].dt.dayofyear
        difference_array_10_days = (
            temperature_days_min
            - percentile_10_calendar_days.sel(dayofyear=time_index)
        ).drop("dayofyear")

        days_10_above_thresholds = xr.where(difference_array_10_days < 0, 1, 0)
        tx10 = (
            days_10_above_thresholds.resample(time="m").sum()
            / days_10_above_thresholds.resample(time="m").count()
        )

        temperature_nights_min = self.temp_extremum("min", "night")
        percentile_10_calendar_nights = self.percentiles(
            10, reference_period, "night"
        )

        time_index = temperature_nights_min["time"].dt.dayofyear
        difference_array_10_nights = (
            temperature_nights_min
            - percentile_10_calendar_nights.sel(dayofyear=time_index)
        ).drop("dayofyear")

        nights_10_above_thresholds = xr.where(difference_array_10_nights < 0, 1, 0)
        tn10 = (
            nights_10_above_thresholds.resample(time="m").sum()
            / nights_10_above_thresholds.resample(time="m").count()
        )

        t10_values = 0.5 * (tx10 + tn10)

        if not self.should_use_dask:
            t10_values = t10_values.compute()

        return t10_values

    def std_t90(self, reference_period, area=None):
        """
        Standardize T90 values.

        Parameters:
        ----------
        reference_period : tuple
            Start and end dates of the reference period.
        area : bool, optional
            If True, compute the area-averaged standardized metric.
            Default is None.

        Returns:
        -------
        xarray.DataArray
            Standardized T90 values.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the study period.
        """
        t90_values = self.t90(reference_period)
        std_t90_values = self.standardize_metric(
            t90_values, reference_period, area)

        if not self.should_use_dask:
            std_t90_values = std_t90_values.compute()

        return std_t90_values

    def std_t10(self, reference_period, area=None):
        """
        Standardize T10 values.

        Parameters:
        ----------
        reference_period : tuple
            Start and end dates of the reference period.
        area : bool, optional
            If True, compute the area-averaged standardized metric.
            Default is None.

        Returns:
        -------
        xarray.DataArray
            Standardized T10 values.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the study period.
        """
        t10_values = self.t10(reference_period)
        std_t10_values = self.standardize_metric(
            t10_values, reference_period, area)

        if not self.should_use_dask:
            std_t10_values = std_t10_values.compute()

        return std_t10_values

    def plot_components(self, component_data0, component_data1, n):
        """
        Plot rolling mean of temperature components.

        Parameters:
        ----------
        component_data0 : xarray.DataArray
            First temperature component data.
        component_data1 : xarray.DataArray
            Second temperature component data.
        n : int
            Rolling window size.

        Complexity:
        ----------
        O(N), where N is the number of time steps in the component data.
        """
        component_data0["t2m"].rolling(time=n, center=True).mean().plot()
        component_data1["t2m"].rolling(time=n, center=True).mean().plot()