import unittest
import xarray as xr
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))

import precipitationcomponent as pc

class TestIntegrationPrecipitationComponent(unittest.TestCase):

    def setUp(self):
        self.reference_period = ('1960-01-01', '1965-12-31')
        self.data_dir = '../data/tests_data/tests_data_prec'

    def test_calculate_monthly_max_anomaly(self):
        test_cases = ['test1', 'test2', 'test3', 'test4']

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                data_path = os.path.join(self.data_dir, f'{test_case}_data.nc')
                mask_path = os.path.join(self.data_dir, f'{test_case}_mask.nc')
                reference_anomalies_path = os.path.join(self.data_dir, f'{test_case}_reference_anomalies.nc')

                # Lire les anomalies de référence
                reference_anomalies = xr.open_dataset(reference_anomalies_path)

                # Initialiser la composante précipitation
                precipitation = pc.PrecipitationComponent(data_path, mask_path)

                # Calculer les anomalies
                anomalies = precipitation.calculate_monthly_max_anomaly('tp', 5, self.reference_period)

                # Comparer avec les anomalies de référence
                np.testing.assert_allclose(anomalies.values, reference_anomalies['tp'].values)

if __name__ == '__main__':
    unittest.main()

