import pandas as pd
from pandas.tseries.offsets import MonthEnd

import aci.components.precipitation as pc
import aci.components.wind as wc
import aci.components.sealevel as sl
import aci.components.drought as dc
import aci.components.temperature as tc
import aci.utils as u


class ActuarialClimateIndex:
    """
    Class to calculate the Actuaries Climate Index (ACI) from its components.

    Attributes:
        temperature_component (TemperatureComponent): Instance of the TemperatureComponent class.
        precipitation_component (PrecipitationComponent): Instance of the PrecipitationComponent
        class.
        drought_component (DroughtComponent): Instance of the DroughtComponent class.
        wind_component (WindComponent): Instance of the WindComponent class.
        sealevel_component (SealevelComponent): Instance of the SealevelComponent class.
        study_period (tuple): Tuple containing the start and end dates of the study period.
        reference_period (tuple): Tuple containing the start and end dates of the reference period.
    """

    def __init__(self, temperature_data_path, precipitation_data_path, wind_u10_data_path,
                 wind_v10_data_path, country_abbrev, mask_data_path, study_period, reference_period):
        """
        Initialize the ACI object with its components.

        Parameters:
            temperature_data_path (str): Path to the temperature data file.
            precipitation_data_path (str): Path to the precipitation data file.
            wind_u10_data_path (str): Path to the wind u-component data file.
            wind_v10_data_path (str): Path to the wind v-component data file.
            country_abbrev (str): Country abbreviation for sea level data.
            mask_data_path (str): Path to the mask data file.
            study_period (tuple): Tuple containing the start and end dates of the study period.
            reference_period (tuple): Tuple containing the start and end dates of the
            reference period.
        """
        self.temperature_component = tc.TemperatureComponent(temperature_data_path, mask_data_path)
        self.precipitation_component = pc.PrecipitationComponent(precipitation_data_path,
                                                                 mask_data_path)
        self.drought_component = dc.DroughtComponent(precipitation_data_path, mask_data_path)
        self.wind_component = wc.WindComponent(wind_u10_data_path, wind_v10_data_path,
                                               mask_data_path)
        self.sealevel_component = sl.SeaLevelComponent(country_abbrev, study_period,
                                                       reference_period)
        self.study_period = study_period
        self.reference_period = reference_period

    def calculate_aci(self, factor=None):
        """
        Calculate the Actuaries Climate Index (calculate_aci).

        Parameters:
            factor (float): Erosion factor, equals 1 by default.

        Returns:
            pd.DataFrame: DataFrame containing the calculate_aci values.

        Complexity:
            Time: O(n) where n is the number of data points in the study period.
            Space: O(n) for storing intermediate DataFrames.
        """
        factor = 1 if factor is None else factor

        components = [self.drought_component, self.wind_component, self.precipitation_component]

        data_arrays = list(map(lambda component : component.calculate_component(self.reference_period, True), components))
        data_arrays.append(self.temperature_component.std_t90(self.reference_period, True))
        data_arrays.append(self.temperature_component.std_t10(self.reference_period, True))

        variables = ['drought','wind','precipitation','t90','t10']
        data_arrays_with_variable_names = zip(data_arrays, variables)

        dataframes = list(map(lambda data_array : u.reduce_dataarray_to_dataframe(data_array[0], data_array[1]), data_arrays_with_variable_names))

        sea_level = self.sealevel_component.process()
        dataframes.append(
            u.reduce_sealevel_over_region(sea_level)
        )
        # Merge DataFrames
        aci_composites = u.merge_dataframes(dataframes)

        # Calculate ACI
        aci_composites["ACI"] = (aci_composites["t90"]
                                 - aci_composites["t10"]
                                 + aci_composites["precipitation"]
                                 + aci_composites["drought"]
                                 + factor * aci_composites["sealevel"]
                                 + aci_composites["wind"]) / 6

        return aci_composites
