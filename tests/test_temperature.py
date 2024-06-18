import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from temperaturecomponent import TemperatureComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestTemperature(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.mask_path = 'test_mask.nc'
        self.data_path = 'test_data.nc'
        self.reference_period = ('2000-01-01', '2004-12-31')

        # Generating test temperature data
        times = pd.date_range('2000-01-01', '2005-12-31', freq='H')
        latitudes = np.arange(48.0, 48.5, 0.1)
        longitudes = np.arange(1.0, 1.5, 0.1)
        np.random.seed(0)
        temperature_data = np.random.rand(len(times), len(latitudes), len(longitudes))

        data = xr.Dataset(
            {'t2m': (['time', 'latitude', 'longitude'], temperature_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)

        # Generating test mask data
        mask_data = np.ones((len(latitudes), len(longitudes)))
        mask = xr.Dataset(
            {'country': (['lat', 'lon'], mask_data)},
            coords={'lat': latitudes, 'lon': longitudes}
        )
        mask.to_netcdf(self.mask_path)
        
        # Construction des chemins d'accès aux données de test stockées

        data_dir = '../data/tests_data/tests_data_temperature'
        self.t2m_path = os.path.join(data_dir, 'test1_t2m.nc')
        self.mask_path_bis = os.path.join(data_dir, 'test1_mask.nc')
        self.reference_anomalies_t90_path = os.path.join(data_dir, 'test1_reference_anomalies_t90.nc')
        self.reference_anomalies_t10_path = os.path.join(data_dir, 'test1_reference_anomalies_t10.nc')


    def tearDown(self):
        """
        Clean up test data files.
        """
        os.remove(self.data_path)
        os.remove(self.mask_path)

    def test_std_t90_month(self):
        """
        Test the std_t90_month method.
        """
        temperature = TemperatureComponent(self.data_path, self.mask_path)
        anomalies = temperature.std_t90_month(self.reference_period)

        # Verify that anomalies is a Dataset
        self.assertIsInstance(anomalies, xr.Dataset)

        # Check the dimensions of the result
        self.assertIn('time', anomalies.dims)
        self.assertIn('latitude', anomalies.dims)
        self.assertIn('longitude', anomalies.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(anomalies['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(anomalies['time'].max(), np.datetime64('2005-12-31'))

    def test_no_temperature_variation(self):
        """
        Test with no temperature variation.
        """
        times = pd.date_range('2000-01-01', '2005-12-31', freq='H')
        latitudes = np.arange(48.0, 48.5, 0.1)
        longitudes = np.arange(1.0, 1.5, 0.1)

        # All zeros for no temperature variation
        temperature_data = np.zeros((len(times), len(latitudes), len(longitudes)))

        data = xr.Dataset(
            {'t2m': (['time', 'latitude', 'longitude'], temperature_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf('test_no_temperature_variation.nc')

        temperature = TemperatureComponent('test_no_temperature_variation.nc', self.mask_path)
        anomalies = temperature.std_t90_month(self.reference_period)

        self.assertTrue(np.all(np.isnan(anomalies['t2m'])), "Anomalies should be NaN when there is no temperature variation.")
        os.remove('test_no_temperature_variation.nc')

    def test_constant_temperature(self):
        """
        Test with constant temperature.
        """
        times = pd.date_range('2000-01-01', '2005-12-31', freq='H')
        latitudes = np.arange(48.0, 48.5, 0.1)
        longitudes = np.arange(1.0, 1.5, 0.1)

        # Constant temperature value
        temperature_data = np.full((len(times), len(latitudes), len(longitudes)), 10)

        data = xr.Dataset(
            {'t2m': (['time', 'latitude', 'longitude'], temperature_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf('test_constant_temperature.nc')

        temperature = TemperatureComponent('test_constant_temperature.nc', self.mask_path)
        anomalies = temperature.std_t90_month(self.reference_period)

        self.assertTrue(np.all(np.isnan(anomalies['t2m'])), "Anomalies should be NaN when temperature is constant.")
        os.remove('test_constant_temperature.nc')

    def test_random_temperature_variation(self):
        """
        Test with random temperature variations.
        """
        times = pd.date_range('2000-01-01', '2005-12-31', freq='H')
        latitudes = np.arange(48.0, 48.5, 0.1)
        longitudes = np.arange(1.0, 1.5, 0.1)

        # Random temperature variation
        np.random.seed(0)
        temperature_data = np.random.rand(len(times), len(latitudes), len(longitudes))

        data = xr.Dataset(
            {'t2m': (['time', 'latitude', 'longitude'], temperature_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf('test_random_temperature_variation.nc')

        temperature = TemperatureComponent('test_random_temperature_variation.nc', self.mask_path)
        anomalies = temperature.std_t90_month(self.reference_period)

        self.assertIsInstance(anomalies, xr.Dataset)
        os.remove('test_random_temperature_variation.nc')

    def test_temperature_component(self):
        temp_component = TemperatureComponent(self.t2m_path, self.mask_path_bis)
        calculated_anomalies_t90 = temp_component.std_t90_month(('1960-01-01', '1961-12-31'), area=True)
        calculated_anomalies_t10 = temp_component.std_t10_month(('1960-01-01', '1961-12-31'), area=True)

        reference_anomalies_t90 = xr.open_dataarray(self.reference_anomalies_t90_path)
        reference_anomalies_t10 = xr.open_dataarray(self.reference_anomalies_t10_path)

        # Extraire les DataArray des Dataset
        calculated_anomalies_t90 = calculated_anomalies_t90['t2m'].values.astype(np.float64)
        calculated_anomalies_t10 = calculated_anomalies_t10['t2m'].values.astype(np.float64)
        reference_anomalies_t90 = reference_anomalies_t90.values.astype(np.float64)
        reference_anomalies_t10 = reference_anomalies_t10.values.astype(np.float64)

        # Vérifier que les anomalies calculées correspondent aux anomalies de référence
        np.testing.assert_allclose(calculated_anomalies_t90, reference_anomalies_t90, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(calculated_anomalies_t10, reference_anomalies_t10, rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main(verbosity=2)
