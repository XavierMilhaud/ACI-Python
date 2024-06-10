import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from precipitationcomponent import PrecipitationComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestPrecipitation(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.mask_path = "test_mask.nc"  # Utilisation d'un chemin temporaire pour le masque
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

        # Affichage des variables du dataset de masque pour vérification
        self.mask = xr.open_dataset(self.mask_path)
        print("Mask Dataset variables:", self.mask.variables)
        print("Mask Dataset coordinates:", self.mask.coords)

        self.reference_period = ('2000-01-01', '2009-12-31')

    def tearDown(self):
        """
        Clean up test data files.
        """
        os.remove(self.data_path)
        os.remove(self.mask_path)

    def test_calculate_monthly_max_anomaly(self):
        """
        Test the calculate_monthly_max_anomaly method.
        """
        precipitation = PrecipitationComponent(self.data_path, self.mask_path)

        # Vérification de la présence de la variable 'country' dans le masque
        if 'country' not in precipitation.mask:
            self.fail("The mask dataset does not contain the 'country' variable.")

        anomalies = precipitation.calculate_monthly_max_anomaly('tp', 5, self.reference_period)

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
        
        # Vérification de la présence de la variable 'country' dans le masque
        if 'country' not in precipitation.mask:
            self.fail("The mask dataset does not contain the 'country' variable.")
        
        anomalies = precipitation.calculate_monthly_max_anomaly('tp', 5, self.reference_period)
        
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
        
        # Vérification de la présence de la variable 'country' dans le masque
        if 'country' not in precipitation.mask:
            self.fail("The mask dataset does not contain the 'country' variable.")
        
        anomalies = precipitation.calculate_monthly_max_anomaly('tp', 5, self.reference_period)
        
        self.assertTrue(np.all(np.isnan(anomalies)), "Anomalies should be NaN when precipitation is constant.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
