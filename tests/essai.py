import xarray as xr
import numpy as np
import pandas as pd
import os
import unittest

class PrecipitationComponent:
    def __init__(self, precipitation_path, mask_path):
        self.precipitation = xr.open_dataset(precipitation_path)
        self.mask = xr.open_dataset(mask_path).rename({'lon': 'longitude', 'lat': 'latitude'})

    def apply_mask(self, var_name):
        temp = self.precipitation.copy()
        temp['mask'] = self.mask.country
        threshold = 0.8
        country_mask = temp['mask'] >= threshold
        temp[var_name] = xr.where(country_mask, temp[var_name], float('nan'))
        return temp.drop_vars('mask')

    def calculate_rolling_sum(self, var_name, window_size):
        preci = self.apply_mask(var_name)
        var = preci[var_name]
        rolling_sum = var.rolling(time=window_size, min_periods=1).sum()
        return rolling_sum

    def calculate_monthly_max(self, var_name, window_size):
        rolling_sum = self.calculate_rolling_sum(var_name, window_size)
        monthly_max = rolling_sum.resample(time='M').max()
        return monthly_max

    def calculate_monthly_max_anomaly(self, var_name, window_size, reference_period):
        monthly_max = self.calculate_monthly_max(var_name, window_size)
        reference_period_data = monthly_max.sel(time=slice(reference_period[0], reference_period[1]))

        # Calculer la moyenne et l'écart-type pour chaque mois pendant la période de référence
        rx5day_mean = reference_period_data.groupby('time.month').mean(dim='time')
        rx5day_std = reference_period_data.groupby('time.month').std(dim='time')

        # Impressions des moyennes et écarts-types calculés pour chaque mois
        print("Monthly Means for Reference Period:")
        print(rx5day_mean)
        print("Monthly Standard Deviations for Reference Period:")
        print(rx5day_std)

        # Associer les valeurs de moyenne et d'écart-type aux données mensuelles maximales
        anomalies = (monthly_max.groupby('time.month') - rx5day_mean) / rx5day_std
        return anomalies

class TestPrecipitation(unittest.TestCase):
    def setUp(self):
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

        mask_data = np.ones((len(latitudes), len(longitudes)))
        mask = xr.Dataset(
            {'country': (['lat', 'lon'], mask_data)},
            coords={'lat': latitudes, 'lon': longitudes}
        )
        mask.to_netcdf(self.mask_path)

        self.reference_period = ('2000-01-01', '2009-12-31')

    def tearDown(self):
        os.remove(self.data_path)
        os.remove(self.mask_path)

    def test_anomalies_with_controlled_trend(self):
        times_ref = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        times_study = pd.date_range('2010-01-01', '2020-12-31', freq='D')
        latitudes = np.arange(48.80, 48.90, 0.1)
        longitudes = np.arange(2.20, 2.30, 0.1)

        mean_ref = 10
        std_ref = 2
        np.random.seed(0)
        precip_ref = np.random.normal(mean_ref, std_ref, (len(times_ref), len(latitudes), len(longitudes)))

        trend = np.linspace(0, 1, len(times_study))[:, None, None]
        precip_study = np.random.normal(mean_ref, std_ref, (len(times_study), len(latitudes), len(longitudes))) + trend

        times = np.concatenate([times_ref, times_study])
        precip_data = np.concatenate([precip_ref, precip_study], axis=0)

        data = xr.Dataset(
            {'tp': (['time', 'latitude', 'longitude'], precip_data)},
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes}
        )
        data.to_netcdf(self.data_path)

        precipitation = PrecipitationComponent(self.data_path, self.mask_path)
        anomalies = precipitation.calculate_monthly_max_anomaly('tp', 5, self.reference_period)

        self.assertIsInstance(anomalies, xr.DataArray)
        self.assertIn('time', anomalies.dims)
        self.assertIn('latitude', anomalies.dims)
        self.assertIn('longitude', anomalies.dims)

        ref_anomalies = anomalies.sel(time=slice(self.reference_period[0], self.reference_period[1]))
        mean_anomaly_ref = ref_anomalies.mean().item()
        print("Mean Anomaly Reference Period:", mean_anomaly_ref)  # Debugging
        self.assertFalse(np.isnan(mean_anomaly_ref), "Mean anomaly should not be NaN.")
        self.assertAlmostEqual(mean_anomaly_ref, 0, places=1)

        std_anomaly_ref = ref_anomalies.std().item()
        print("Std Anomaly Reference Period:", std_anomaly_ref)  # Debugging
        self.assertFalse(np.isnan(std_anomaly_ref), "Std anomaly should not be NaN.")
        self.assertAlmostEqual(std_anomaly_ref, 1, places=1)

        study_period = ('2010-01-01', '2020-12-31')
        study_anomalies = anomalies.sel(time=slice(study_period[0], study_period[1]))
        mean_anomaly_study = study_anomalies.mean().item()
        std_anomaly_study = study_anomalies.std().item()

        print("Mean Anomaly Study Period:", mean_anomaly_study)  # Debugging
        print("Std Anomaly Study Period:", std_anomaly_study)  # Debugging

        expected_mean_study = np.mean(trend)
        expected_mean_global = (0 * len(times_ref) + expected_mean_study * len(times_study)) / len(times)
        expected_std_global = 1

        self.assertFalse(np.isnan(mean_anomaly_study), "Mean anomaly over the study period should not be NaN.")
        self.assertAlmostEqual(mean_anomaly_study, expected_mean_study, places=1, msg="Mean anomaly over the study period should reflect the trend.")
        self.assertFalse(np.isnan(std_anomaly_study), "Std anomaly over the study period should not be NaN.")
        self.assertAlmostEqual(std_anomaly_study, expected_std_global, places=1, msg="Std anomaly over the study period should be consistent with the reference period.")

        global_anomalies = anomalies.sel(time=slice('2000-01-01', '2020-12-31'))
        mean_anomaly_global = global_anomalies.mean().item()
        std_anomaly_global = global_anomalies.std().item()

        print("Mean Anomaly Global Period:", mean_anomaly_global)  # Debugging
        print("Std Anomaly Global Period:", std_anomaly_global)  # Debugging

        self.assertFalse(np.isnan(mean_anomaly_global), "Mean anomaly over the entire period should not be NaN.")
        self.assertAlmostEqual(mean_anomaly_global, expected_mean_global, places=1, msg="Mean anomaly over the entire period should reflect the diluted trend.")
        self.assertFalse(np.isnan(std_anomaly_global), "Std anomaly over the entire period should not be NaN.")
        self.assertAlmostEqual(std_anomaly_global, expected_std_global, places=1, msg="Std anomaly over the entire period should be consistent with the reference period.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
