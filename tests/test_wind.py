import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from windcomponent import WindComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TestWindComponent(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.mask_path = "test_mask.nc"
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)

        np.random.seed(0)
        wind_data = np.random.rand(len(times), len(latitudes), len(longitudes))
        self.u10_path = 'test_u10.nc'
        self.v10_path = 'test_v10.nc'

        u10_data = xr.Dataset(
            {'u10': (['time', 'latitude', 'longitude'], wind_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        v10_data = xr.Dataset(
            {'v10': (['time', 'latitude', 'longitude'], wind_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        u10_data.to_netcdf(self.u10_path)
        v10_data.to_netcdf(self.v10_path)

        # Création d'un masque avec la variable 'country'
        mask_data = np.ones((len(latitudes), len(longitudes)))
        mask = xr.Dataset(
            {'country': (['lat', 'lon'], mask_data)},
            coords={'lat': latitudes, 'lon': longitudes}
        )
        mask.to_netcdf(self.mask_path)

        self.reference_period = ('2000-01-01', '2009-12-31')

    def tearDown(self):
        """
        Clean up test data files.
        """
        os.remove(self.u10_path)
        os.remove(self.v10_path)
        os.remove(self.mask_path)

    def test_calculate_wind_power(self):
        """
        Test the calculate_wind_power method.
        """
        wind = WindComponent(self.u10_path, self.v10_path, self.mask_path)
        wind_power = wind.calculate_wind_power()

        # Verify that wind_power is a DataArray
        self.assertIsInstance(wind_power, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', wind_power.dims)
        self.assertIn('latitude', wind_power.dims)
        self.assertIn('longitude', wind_power.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(wind_power['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(wind_power['time'].max(), np.datetime64('2020-12-31'))

    def test_calculate_wind_thresholds(self):
        """
        Test the calculate_wind_thresholds method.
        """
        wind = WindComponent(self.u10_path, self.v10_path, self.mask_path)
        wind_thresholds = wind.calculate_wind_thresholds(self.reference_period)

        # Verify that wind_thresholds is a DataArray
        self.assertIsInstance(wind_thresholds, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', wind_thresholds.dims)
        self.assertIn('latitude', wind_thresholds.dims)
        self.assertIn('longitude', wind_thresholds.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(wind_thresholds['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(wind_thresholds['time'].max(), np.datetime64('2020-12-31'))

    def test_calculate_days_above_thresholds(self):
        """
        Test the calculate_days_above_thresholds method.
        """
        wind = WindComponent(self.u10_path, self.v10_path, self.mask_path)
        days_above_thresholds = wind.calculate_days_above_thresholds(self.reference_period)

        # Verify that days_above_thresholds is a DataArray
        self.assertIsInstance(days_above_thresholds, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', days_above_thresholds.dims)
        self.assertIn('latitude', days_above_thresholds.dims)
        self.assertIn('longitude', days_above_thresholds.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(days_above_thresholds['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(days_above_thresholds['time'].max(), np.datetime64('2020-12-31'))

    def test_calculate_wind_exceedance_frequency(self):
        """
        Test the calculate_wind_exceedance_frequency method.
        """
        wind = WindComponent(self.u10_path, self.v10_path, self.mask_path)
        wind_exceedance_frequency = wind.calculate_wind_exceedance_frequency(self.reference_period)

        # Verify that wind_exceedance_frequency is a DataArray
        self.assertIsInstance(wind_exceedance_frequency, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', wind_exceedance_frequency.dims)
        self.assertIn('latitude', wind_exceedance_frequency.dims)
        self.assertIn('longitude', wind_exceedance_frequency.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(wind_exceedance_frequency['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(wind_exceedance_frequency['time'].max(), np.datetime64('2020-12-31'))

    def test_standardize_wind_exceedance_frequency(self):
        """
        Test the standardize_wind_exceedance_frequency method.
        """
        wind = WindComponent(self.u10_path, self.v10_path, self.mask_path)
        standardized_frequency = wind.standardize_wind_exceedance_frequency(self.reference_period)

        # Verify that standardized_frequency is a DataArray
        self.assertIsInstance(standardized_frequency, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', standardized_frequency.dims)
        self.assertIn('latitude', standardized_frequency.dims)
        self.assertIn('longitude', standardized_frequency.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(standardized_frequency['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(standardized_frequency['time'].max(), np.datetime64('2020-12-31'))

        # Ensure that the mean standardized frequency over the reference period is approximately zero
        ref_standardized_frequency = standardized_frequency.sel(time=slice(self.reference_period[0], self.reference_period[1]))
        mean_standardized_frequency = ref_standardized_frequency.mean().item()
        self.assertFalse(np.isnan(mean_standardized_frequency), "Mean standardized frequency should not be NaN.")
        self.assertAlmostEqual(mean_standardized_frequency, 0, places=1)

        # Ensure that the standard deviation of the standardized frequency over the reference period is approximately one
        std_standardized_frequency = ref_standardized_frequency.std().item()
        self.assertFalse(np.isnan(std_standardized_frequency), "Std standardized frequency should not be NaN.")
        self.assertAlmostEqual(std_standardized_frequency, 1, places=1)

    
  

if __name__ == '__main__':
    unittest.main(verbosity=2)