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

class TestIntegrationWindComponent(unittest.TestCase):

    def setUp(self):
        """
        Setup test data paths.
        """
        self.test_cases = ['test1', 'test2', 'test3', 'test4']
        self.reference_period = ('2000-01-01', '2009-12-31')

    def test_standardize_wind_exceedance_frequency(self):
        """
        Test the standardize_wind_exceedance_frequency method against precomputed reference anomalies.
        """
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                u10_data_path = f'../data/tests_data/tests_data_wind/{test_case}_u10_test_data.nc'
                v10_data_path = f'../data/tests_data/tests_data_wind/{test_case}_v10_test_data.nc'
                mask_data_path = f'../data/tests_data/tests_data_wind/{test_case}_mask_test_data.nc'
                reference_anomalies_path = f'../data/tests_data/tests_data_wind/{test_case}_reference_anomalies.nc'

                wind_component = WindComponent(u10_data_path, v10_data_path, mask_data_path)
                calculated_anomalies = wind_component.standardize_wind_exceedance_frequency(self.reference_period, area=True)
            
                calculated_anomalies.name = "calculated"

                reference_anomalies = xr.open_dataset(reference_anomalies_path)

                
                calculated_df = calculated_anomalies.to_dataframe()
                reference_df = reference_anomalies.to_dataframe()

               
                calculated_df.rename(columns={calculated_anomalies.name: 'calculated'}, inplace=True)
                reference_var_name = list(reference_anomalies.data_vars.keys())[0]
                reference_df.rename(columns={reference_var_name: 'reference'}, inplace=True)

               
                assert 'calculated' in calculated_df.columns, f"'calculated' column not found in calculated_df for {test_case}"
                assert 'reference' in reference_df.columns, f"'reference' column not found in reference_df for {test_case}"

             
                combined_df = pd.merge(calculated_df, reference_df, left_index=True, right_index=True, how='inner')

                pd.testing.assert_series_equal(combined_df['calculated'], combined_df['reference'], check_less_precise=True, check_names=False)

if __name__ == '__main__':
    unittest.main(verbosity=2)
