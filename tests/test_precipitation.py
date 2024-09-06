import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings

from aci.components.precipitation import PrecipitationComponent


class TestPrecipitation(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.mask_path = "test_mask.nc"
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)

        np.random.seed(0)
        precipitation_data = np.random.rand(len(times), len(latitudes), len(longitudes))
        self.data_path = 'test_data.nc'

        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)

        # Création d'un masque avec la variable 'country'
        mask_data = np.ones((len(latitudes), len(longitudes)))
        mask = xr.Dataset(
            {'country': (['lat', 'lon'], mask_data)},
            coords={'lat': latitudes, 'lon': longitudes}
        )
        mask.to_netcdf(self.mask_path)
        self.reference_period = ('2000-01-01', '2009-12-31')

        # Ajout des nouveaux tests
        self.reference_period_bis = ('2000-01-01', '2000-12-31')
        self.data_dir = 'data/tests_data/tests_data_prec_bis'

    def tearDown(self):
        """
        Clean up test data files.
        """
        os.remove(self.data_path)
        os.remove(self.mask_path)

    def test_calculate_component(self):
        """
        Test the calculate_component method.
        """
        precipitation = PrecipitationComponent(self.data_path, self.mask_path)

        anomalies = precipitation.calculate_component(self.reference_period)

        # Verify that anomalies is a DataArray
        self.assertIsInstance(anomalies, xr.DataArray)

        # Check the dimensions of the result
        self.assertIn('time', anomalies.dims)
        self.assertIn('latitude', anomalies.dims)
        self.assertIn('longitude', anomalies.dims)

        # Check that the result contains the correct time period
        self.assertGreaterEqual(anomalies['time'].min(), np.datetime64('2000-01-01'))
        self.assertLessEqual(anomalies['time'].max(), np.datetime64('2020-12-31'))

        # Ensure that the mean anomaly over the reference period is approximately zero
        ref_anomalies = anomalies.sel(time=slice(self.reference_period[0], self.reference_period[1]))
        mean_anomaly = ref_anomalies.mean().item()
        self.assertFalse(np.isnan(mean_anomaly), "Mean anomaly should not be NaN.")
        self.assertAlmostEqual(mean_anomaly, 0, places=1)

        # Ensure that the standard deviation of anomalies over the reference period is approximately one
        std_anomaly = ref_anomalies.std().item()
        self.assertFalse(np.isnan(std_anomaly), "Std anomaly should not be NaN.")
        self.assertAlmostEqual(std_anomaly, 1, places=1)

    def test_no_precipitation(self):
        """
        Test with no precipitation.
        """
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)

        # All zeros for no precipitation
        precipitation_data = np.zeros((len(times), len(latitudes), len(longitudes)))

        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)

        precipitation = PrecipitationComponent(self.data_path, self.mask_path)

        anomalies = precipitation.calculate_component(self.reference_period)

        self.assertTrue(np.all(np.isnan(anomalies)), "Anomalies should be NaN when there is no precipitation.")

    def test_constant_precipitation(self):
        """
        Test with constant precipitation.
        """
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)

        # Constant precipitation value
        precipitation_data = np.full((len(times), len(latitudes), len(longitudes)), 10)

        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)

        precipitation = PrecipitationComponent(self.data_path, self.mask_path)
 
        anomalies = precipitation.calculate_component(self.reference_period)

        self.assertTrue(np.all(np.isnan(anomalies)), "Anomalies should be NaN when precipitation is constant.")

    def test_calculate_component_bis(self):
        test_cases = ['test1', 'test2', 'test3', 'test4']

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                data_path = os.path.join(self.data_dir, f'{test_case}_data.nc')
                mask_path = os.path.join(self.data_dir, f'{test_case}_mask.nc')
                reference_anomalies_path = os.path.join(self.data_dir, f'{test_case}_reference_anomalies.nc')

                # Lire les anomalies de référence
                reference_anomalies = xr.open_dataset(reference_anomalies_path)

                # Initialiser la composante précipitation
                precipitation = PrecipitationComponent(data_path, mask_path)

                # Calculer les anomalies
                anomalies = precipitation.calculate_component(self.reference_period_bis)

                # Comparer avec les anomalies de référence
                np.testing.assert_allclose(anomalies.values, reference_anomalies['tp'].values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
