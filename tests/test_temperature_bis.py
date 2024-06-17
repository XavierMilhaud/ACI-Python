import unittest
import xarray as xr
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from temperaturecomponent import TemperatureComponent

class TestIntegrationTemperatureComponent(unittest.TestCase):
    def setUp(self):
        data_dir = '../data/tests_data/tests_data_temperature'
        self.t2m_path = os.path.join(data_dir, 'test1_t2m.nc')
        self.mask_path = os.path.join(data_dir, 'test1_mask.nc')
        self.reference_anomalies_t90_path = os.path.join(data_dir, 'test1_reference_anomalies_t90.nc')
        self.reference_anomalies_t10_path = os.path.join(data_dir, 'test1_reference_anomalies_t10.nc')

    def test_temperature_component(self):
        temp_component = TemperatureComponent(self.t2m_path, self.mask_path)
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
    unittest.main()
