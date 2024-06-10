import pandas as pd
import xarray as xr
import precipitationcomponent as pc
import windcomponent as wc
import sealevel as sl
import droughtcomponent as dc
import temperaturecomponent as tc
from datetime import datetime
from pandas.tseries.offsets import MonthEnd






class ActuarialClimateIndex:
    """

    Class to calculate the Actuaries Climate Index (ACI) from its components.

    Attributes:
    - temperature_component (TemperatureComponent): Instance of the TemperatureComponent class.
    - precipitation_component (PrecipitationComponent): Instance of the PrecipitationComponent class.
    - drought_component (DroughtComponent): Instance of the DroughtComponent class.
    - wind_component (WindComponent): Instance of the WindComponent class.
    - sealevel_component (SealevelComponent): Instance of the SealevelComponent class.


    """

    def __init__(self, temperature_data_path, precipitation_data_path, wind_u10_data_path, 
                 wind_v10_data_path, country_abbrev, mask_data_path, study_period, reference_period):
        """
        
        Initialize the ACI object with its components.

        Parameters:
        - temperature_component (TemperatureComponent): Instance of the TemperatureComponent class.
        - precipitation_component (PrecipitationComponent): Instance of the PrecipitationComponent class.
        - drought_component (DroughtComponent): Instance of the DroughtComponent class.
        - wind_component (WindComponent): Instance of the WindComponent class.
        - sealevel_component (SealevelComponent): Instance of the SealevelComponent class.


        """
        self.temperature_component = tc.TemperatureComponent(temperature_data_path, mask_data_path)
        self.precipitation_component = pc.PrecipitationComponent(precipitation_data_path, mask_data_path)
        self.drought_component = dc.DroughtComponent(precipitation_data_path, mask_data_path)
        self.wind_component = wc.WindComponent(wind_u10_data_path, wind_v10_data_path, mask_data_path)
        self.sealevel_component = sl.SeaLevelComponent(country_abbrev, study_period, reference_period)
        self.study_period = study_period
        self.reference_period = reference_period

    def ACI(self, factor=None):
        """
        Calculate the Actuaries Climate Index (ACI).

        Args :
        factor (float) : erosion factor, equals 1 by default

        Returns:
        - float: Actuaries Climate Index (ACI).
        """

        factor = 1 if factor is None else factor


        preci_std = self.precipitation_component.calculate_monthly_max_anomaly("tp", 5, self.reference_period, True)
        p_df = preci_std.to_dataframe()
        p_df.columns = ["precipitation"]

        wind_std = self.wind_component.standardize_wind_exceedance_frequency(self.reference_period, True)
        w_df = wind_std.to_dataframe(name="windpower")
        w_df.columns = ["windpower"]

        drought_std = self.drought_component.standardize_max_consecutive_dry_days(self.reference_period, True)
        cdd_df = drought_std.to_dataframe()
        cdd_df.columns = ["drought"]

        temp90_std = self.temperature_component.std_t90_month(self.reference_period, True)
        t90_df = temp90_std.to_dataframe()
        t90_df.columns = ["T90"]

        temp10_std = self.temperature_component.std_t10_month(self.reference_period, True)
        t10_df = temp10_std.to_dataframe()
        t10_df.columns = ["T10"]

        
        sea_lev = self.sealevel_component.process()
        sea_std = sea_lev.mean(axis=1)
        sea_df = pd.DataFrame(sea_std, columns=["sea_mean"])
        sea_df["time"] = pd.to_datetime(sea_df.index, format="%Y-%m-%d") + MonthEnd()
        sea_df.set_index("time", inplace=True)


        df1 = pd.merge(w_df["windpower"], p_df["precipitation"], left_index=True, right_index=True)
        df2 = pd.merge(df1, cdd_df["drought"], left_index=True, right_index=True)
        df3 = pd.merge(df2,sea_df["sea_mean"], left_index=True, right_index=True)
        df4 = pd.merge(df3, t90_df["T90"], left_index=True, right_index=True)
        IACF_composites = pd.merge(df4,t10_df["T10"], left_index=True, right_index=True)

        IACF_composites["ACI"] = (IACF_composites["T90"] - IACF_composites["T10"] + IACF_composites["precipitation"] + IACF_composites["drought"] + factor*IACF_composites["sea_mean"] + IACF_composites["windpower"] )/6


        return IACF_composites
    
