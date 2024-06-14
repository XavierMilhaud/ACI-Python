import unittest
import xarray as xr
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from droughtcomponent import DroughtComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestIntegrationDroughtComponent(unittest.TestCase):

    def setUp(self):
        """
        Setup test data paths.
        """
        self.test_cases = ['test1', 'test2', 'test3', 'test4']
        self.reference_period = ('2000-01-01', '2009-12-31')
        self.study_period = ('2000-01-01', '2010-12-31')

    def test_standardize_drought(self):
        """
        Test the standardize_max_consecutive_dry_days method against precomputed reference anomalies.
        """
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                print(f"Starting test case: {test_case}")
                precipitation_path = f'../data/tests_data/tests_data_drought/{test_case}_precipitation_test_data.nc'
                mask_path = f'../data/tests_data/tests_data_drought/{test_case}_mask_test_data.nc'
                reference_anomalies_path = f'../data/tests_data/tests_data_drought/{test_case}_reference_anomalies.nc'

                # Configurer le composant Drought
                print("Loading DroughtComponent")
                drought_component = DroughtComponent(precipitation_path, mask_path)

                # Calculer les anomalies
                print("Calculating anomalies")
                calculated_anomalies = drought_component.standardize_max_consecutive_dry_days(self.reference_period, area=False)

                # Lire les anomalies de référence
                print("Loading reference anomalies")
                reference_anomalies = xr.open_dataset(reference_anomalies_path)

                # Vérification des dimensions
                print(f"Dimensions in calculated anomalies: {calculated_anomalies.dims}")
                print(f"Dimensions in reference anomalies: {reference_anomalies.dims}")

                # Comparer les anomalies calculées et de référence
                print("Calculating mean of anomalies")
                calculated_mean = calculated_anomalies.mean(dim=['lat', 'lon'])  # Assurez-vous que les dimensions sont correctes
                reference_mean = reference_anomalies.mean(dim=['lat', 'lon'])  # Assurez-vous que les dimensions sont correctes

                # Convertir en DataFrame pour comparaison
                print("Converting to DataFrame")
                calculated_df = calculated_mean.to_dataframe(name='calculated_mean').reset_index()
                reference_df = reference_mean.to_dataframe(name='reference_mean').reset_index()

                print("Merging DataFrames")
                combined_df = pd.merge(calculated_df, reference_df, on='time', suffixes=('_calculated', '_reference')).dropna()

                print("Asserting series equal")
                pd.testing.assert_series_equal(
                    combined_df['calculated_mean'],
                    combined_df['reference_mean'],
                    check_less_precise=True,
                    check_names=False
                )
                print(f"Test case {test_case} passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
