import unittest
import xarray as xr
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))

from windcomponent import WindComponent

class TestIntegrationWindComponent(unittest.TestCase):

    def setUp(self):
        self.reference_period = ('2000-01-01', '2000-12-31')
        self.data_dir = '../data/tests_data/tests_data_wind'

    def test_standardize_wind_exceedance_frequency(self):
        test_cases = ['test1', 'test2', 'test3', 'test4']

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                
                u10_path = os.path.join(self.data_dir, f'{test_case}_u10.nc')
                v10_path = os.path.join(self.data_dir, f'{test_case}_v10.nc')
                mask_path = os.path.join(self.data_dir, f'{test_case}_mask.nc')
                reference_anomalies_path = os.path.join(self.data_dir, f'{test_case}_reference_anomalies.nc')

                reference_anomalies = xr.open_dataset(reference_anomalies_path)
                wind_component = WindComponent(u10_path, v10_path, mask_path)

                calculated_anomalies = wind_component.standardize_wind_exceedance_frequency(self.reference_period, area=True)

                reference_variable_name = list(reference_anomalies.data_vars)[0]
                np.testing.assert_allclose(calculated_anomalies.values, reference_anomalies[reference_variable_name].values)

if __name__ == '__main__':
    unittest.main(verbosity=2)
