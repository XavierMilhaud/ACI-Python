import unittest
import pandas as pd
import os
import warnings
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from sealevel import SeaLevelComponent

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestIntegrationSeaLevelComponent(unittest.TestCase):

    def setUp(self):
        """
        Setup test data paths.
        """
        self.test_cases = ['test1', 'test2', 'test3', 'test4']
        self.reference_period = ('2000-01-01', '2000-12-31')
        self.study_period = ('2000-01-01', '2001-12-31')

    def test_standardize_sealevel(self):
        """
        Test the standardize_sealevel method against precomputed reference anomalies.
        """
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                sea_level_data_path = f'../data/tests_data/tests_data_sealevel/{test_case}_sea_level_test_data.csv'
                reference_anomalies_path = f'../data/tests_data/tests_data_sealevel/{test_case}_reference_anomalies.csv'

                # Lire les données de niveau de la mer
                sea_level_data = pd.read_csv(sea_level_data_path, index_col='Corrected_Date', parse_dates=True)

                # Configurer le composant SeaLevel
                sea_level_component = SeaLevelComponent(
                    'FRA',
                    self.study_period,
                    self.reference_period
                )
                sea_level_component.data = sea_level_data  # Injecter les données générées

                # Calculer les anomalies
                calculated_anomalies = sea_level_component.process()

                # Lire les anomalies de référence
                reference_anomalies = pd.read_csv(reference_anomalies_path, index_col='Corrected_Date', parse_dates=True)

                combined_df = pd.DataFrame({
                    'calculated_mean': calculated_anomalies.mean(axis=1),
                    'reference_mean': reference_anomalies.mean(axis=1)
                }).dropna()

                # Comparer les anomalies aux valeurs de référence
                pd.testing.assert_series_equal(
                    combined_df['calculated_mean'],
                    combined_df['reference_mean'],
                    check_less_precise=True,
                    check_names=False
                )

if __name__ == '__main__':
    unittest.main(verbosity=2)
