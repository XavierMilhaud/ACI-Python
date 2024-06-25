import unittest
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import warnings
from ActuarialClimateIndex import ActuarialClimateIndex

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestActuarialClimateIndex(unittest.TestCase):

    def setUp(self):
        self.mask_path = 'test_mask.nc'
        self.temperature_path = 'test_temperature.nc'
        self.precipitation_path = 'test_precipitation.nc'
        self.wind_u10_path = 'test_wind_u10.nc'
        self.wind_v10_path = 'test_wind_v10.nc'
        self.reference_period = ('1960-01-01', '1965-12-31')
        self.study_period = ('1960-01-01', '1970-12-31')
        self.country_abbrev = 'FRA'

        # Generating test temperature data
        times = pd.date_range('1960-01-01', '1970-12-31', freq='H')
        latitudes = [49.0, 48.75, 48.5, 48.25, 48.0]
        longitudes = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        np.random.seed(0)
        temperature_data = np.random.rand(len(times), len(latitudes), len(longitudes))

        temperature_ds = xr.Dataset(
            {'t2m': (['time', 'latitude', 'longitude'], temperature_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        temperature_ds.to_netcdf(self.temperature_path)

        # Generating test mask data
        mask_data = np.ones((len(latitudes), len(longitudes)))
        mask_ds = xr.Dataset(
            {'country': (['lat', 'lon'], mask_data)},
            coords={'lat': latitudes, 'lon': longitudes}
        )
        mask_ds.to_netcdf(self.mask_path)

        # Generating test precipitation data
        precipitation_data = np.random.rand(len(times), len(latitudes), len(longitudes))
        precipitation_ds = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precipitation_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        precipitation_ds.to_netcdf(self.precipitation_path)

        # Generating test wind data
        wind_u10_data = np.random.rand(len(times), len(latitudes), len(longitudes))
        wind_v10_data = np.random.rand(len(times), len(latitudes), len(longitudes))
        wind_u10_ds = xr.Dataset(
            {'u10': (['time', 'latitude', 'longitude'], wind_u10_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        wind_u10_ds.to_netcdf(self.wind_u10_path)
        wind_v10_ds = xr.Dataset(
            {'v10': (['time', 'latitude', 'longitude'], wind_v10_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        wind_v10_ds.to_netcdf(self.wind_v10_path)

    def tearDown(self):
        os.remove(self.temperature_path)
        os.remove(self.mask_path)
        os.remove(self.precipitation_path)
        os.remove(self.wind_u10_path)
        os.remove(self.wind_v10_path)

    def test_aci_calculation(self):
        aci = ActuarialClimateIndex(
            self.temperature_path,
            self.precipitation_path,
            self.wind_u10_path,
            self.wind_v10_path,
            self.country_abbrev,
            self.mask_path,
            self.study_period,
            self.reference_period
        )

        aci_value = aci.ACI()

        self.assertIsInstance(aci_value, pd.DataFrame)
        self.assertIn('ACI', aci_value.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
