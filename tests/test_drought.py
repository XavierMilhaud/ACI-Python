import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from droughtcomponent import DroughtComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestDrought(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.mask_path = "test_mask.nc"
        self.data_path = 'test_data.nc'
        self.reference_period = ('2000-01-01', '2009-12-31')

        # Generating test precipitation data
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)
        np.random.seed(0)
        precipitation_data = np.random.rand(len(times), len(latitudes), len(longitudes))

        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
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

    def tearDown(self):
        """
        Clean up test data files.
        """
        os.remove(self.data_path)
        os.remove(self.mask_path)

    def test_standardize_max_consecutive_dry_days(self):
        """
        Test the standardize_max_consecutive_dry_days method.
        """
        drought = DroughtComponent(self.data_path, self.mask_path)
        anomalies = drought.standardize_max_consecutive_dry_days(self.reference_period)

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
        
        drought = DroughtComponent(self.data_path, self.mask_path)
        
        anomalies = drought.standardize_max_consecutive_dry_days(self.reference_period)
        
        self.assertTrue(np.all(np.isnan(anomalies)), "Anomalies should be NaN when there is no precipitation.")
    
    def test_constant_precipitation(self):
        """
        Test with constant precipitation.
        """
        times = pd.date_range('2000-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)
        
        # Constant precipitation value (below the threshold to simulate dry days)
        precipitation_data = np.full((len(times), len(latitudes), len(longitudes)), 0.0005)
        
        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)
        
        drought = DroughtComponent(self.data_path, self.mask_path)
        
        anomalies = drought.standardize_max_consecutive_dry_days(self.reference_period)
        cal = drought.calculate_max_consecutive_dry_days()

        self.assertTrue(np.all(np.isnan(anomalies)), "Anomalies should be NaN when precipitation is constant below the threshold")
        self.assertTrue(np.all(cal == cal[0, 0, 0]), "Max consecutive dry days should be the same when precipitation is constant and below the threshold.")

if __name__ == '__main__':
    unittest.main(verbosity=2)