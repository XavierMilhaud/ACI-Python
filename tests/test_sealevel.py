import unittest
import os
import pandas as pd
import numpy as np
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aci')))
from sealevel import SeaLevelComponent

class TestSeaLevelComponent(unittest.TestCase):

    def setUp(self):
        """
        Set up test data and paths.
        """
        self.country_abrev = "USA"
        self.study_period = ('1960-01-01', '1969-12-31')
        self.reference_period = ('1960-01-01', '1964-12-31')
        self.sea_level_component = SeaLevelComponent(self.country_abrev, self.study_period, self.reference_period)
        self.data_path = "../data/sealevel_data_USA"
        
        # Creating test data directory and files
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        # Creating a sample sea level data file
        dates = pd.date_range('1960-01-01', '1964-12-31', freq='M')
        values = np.random.rand(len(dates)) * 100
        df = pd.DataFrame({"Date": dates, "Measurement_test": values})
        df['Date'] = df['Date'].apply(lambda x: float(f"{x.year}.{x.month:02}125"))
        df.to_csv(os.path.join(self.data_path, "test_file.txt"), sep=";", index=False, header=False)

    def tearDown(self):
        """
        Clean up test data.
        """
        for file in os.listdir(self.data_path):
            os.remove(os.path.join(self.data_path, file))
        os.rmdir(self.data_path)

    def test_load_data(self):
        """
        Test loading of sea level data.
        """
        data = self.sea_level_component.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_correct_date_format(self):
        """
        Test correction of date format.
        """
        data = self.sea_level_component.load_data()
        corrected_data = self.sea_level_component.correct_date_format(data)
        self.assertIsInstance(corrected_data.index, pd.DatetimeIndex)
        self.assertFalse(corrected_data.empty)

    def test_clean_data(self):
        """
        Test cleaning of data.
        """
        data = self.sea_level_component.load_data()
        data = self.sea_level_component.correct_date_format(data)
        data.iloc[0, 0] = -99999.0  # Introducing a sentinel value
        cleaned_data = self.sea_level_component.clean_data(data)
        self.assertTrue(np.isnan(cleaned_data.iloc[0, 0]))

    def test_compute_monthly_stats(self):
        """
        Test computation of monthly statistics.
        """
        data = self.sea_level_component.load_data()
        data = self.sea_level_component.correct_date_format(data)
        data = self.sea_level_component.clean_data(data)
        monthly_means = self.sea_level_component.compute_monthly_stats(data, self.reference_period, "means")
        monthly_std_devs = self.sea_level_component.compute_monthly_stats(data, self.reference_period, "std")
        self.assertIsInstance(monthly_means, pd.Series)
        self.assertIsInstance(monthly_std_devs, pd.Series)
        self.assertEqual(len(monthly_means), 12)
        self.assertEqual(len(monthly_std_devs), 12)

    def test_standardize_data(self):
        """
        Test standardization of data.
        """
        data = self.sea_level_component.load_data()
        data = self.sea_level_component.correct_date_format(data)
        data = self.sea_level_component.clean_data(data)
        monthly_means = self.sea_level_component.compute_monthly_stats(data, self.reference_period, "means")
        monthly_std_devs = self.sea_level_component.compute_monthly_stats(data, self.reference_period, "std")
        standardized_data = self.sea_level_component.standardize_data(data, monthly_means, monthly_std_devs, self.study_period)
        self.assertIsInstance(standardized_data, pd.DataFrame)
        self.assertFalse(standardized_data.empty)

    def test_process(self):
        """
        Test the full processing.
        """
        standardized_data = self.sea_level_component.process()
        self.assertIsInstance(standardized_data, pd.DataFrame)
        self.assertFalse(standardized_data.empty)

if __name__ == "__main__":
    unittest.main(verbosity=2)
