import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
#import Era5var
from datetime import datetime




class TemperatureComponent:


    def __init__(self, temperature_data_path, mask_data_path):
        """
        Initialize the TemperatureComponent object.

        Parameters:
        - temperature_data (xarray.Dataset): Dataset containing temperature data.
        - mask_data (xarray.Dataset): Dataset containing mask data.
        """
        self.temperature_data = xr.open_dataset(temperature_data_path)
        self.mask_data = xr.open_dataset(mask_data_path).rename({'lon':'longitude', 'lat': 'latitude'})
        temperature =  self.apply_mask("t2m")
        self.temperature_days = temperature.isel(time=temperature.time.dt.hour.isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))
        self.temperature_nights = temperature.isel(time=temperature.time.dt.hour.isin([0,1,2,3,4,5,22,23]))
    


    def apply_mask(self, var_name):
        """
        Apply a mask to the temperature data.

        Parameters:
        - var_name (str): Name of the variable to apply the mask to.
        - threshold (float): Threshold value for the mask.

        Returns:
        - xarray.Dataset: Dataset with the mask applied to the specified variable.
        """
        threshold = 0.8
        temp = self.temperature_data.copy()
        temp['mask'] = self.mask_data.country
        country_mask = temp['mask'] >= threshold
        temp[var_name] = xr.where(country_mask, temp[var_name], float('nan'))
        return temp.drop_vars('mask')
    

    def temp_days_extremum(self, extremum):

    
        if (extremum == "min"):
            return self.temperature_days.resample(time='D').min()
        elif (extremum == "max"):
            return self.temperature_days.resample(time='D').max()
        else:
            raise Exception("extremum in wrong format : should be 'min' or 'max' ")
        

    def temp_nights_extremum(self, extremum):


        if (extremum == "min"):
            return self.temperature_nights.resample(time='D').min()
        elif (extremum == "max"):
            return self.temperature_nights.resample(time='D').max()
        else:
            raise Exception("extremum in wrong format : should be 'min' or 'max' ")
        
    
    def percentiles(self, n,  reference_period, tempo):

        
        if (tempo == "day"):
            rolling_window_size_days = 80
            temperature_days_reference = self.temperature_days.sel(time=slice(reference_period[0], reference_period[1]))
            percentile_days_reference = temperature_days_reference['t2m'].rolling(time=rolling_window_size_days, min_periods=1,center=True).reduce(np.percentile, q=n)
            percentile_calendar_days = percentile_days_reference.groupby('time.dayofyear').reduce(np.percentile, q=n)

            return percentile_calendar_days
        
        elif (tempo == "night"):
            rolling_window_size_nights = 40
            temperature_nights_reference = self.temperature_days.sel(time=slice(reference_period[0], reference_period[1]))
            percentile_nights_reference = temperature_nights_reference['t2m'].rolling(time=rolling_window_size_nights, min_periods=1,center=True).reduce(np.percentile, q=n)
            percentile_calendar_nights = percentile_nights_reference.groupby('time.dayofyear').reduce(np.percentile, q=n)

            return percentile_calendar_nights
        
        else : 
            raise Exception("tempo in wrong format : 'day' or 'night' ")


    def t90_month(self, reference_period):
        

        #Computing tx90

        temperature_days_max = self.temp_days_extremum("max")
        percentile_90_calendar_days = self.percentiles(90, reference_period, "day")
        time_index = temperature_days_max["time"].dt.dayofyear
        difference_array_90_days = (temperature_days_max - percentile_90_calendar_days.sel(dayofyear=time_index)).drop("dayofyear")
        #difference_array_90_nights = temperature_nights_max - percentile_90_calendar_nights
        days_90_above_thresholds = xr.where(difference_array_90_days > 0, 1, 0)
        tx90 = days_90_above_thresholds.resample(time='m').sum()/days_90_above_thresholds.resample(time="m").count()

        #Computing tn90
        

        temperature_nights_max = self.temp_nights_extremum("max")
        percentile_90_calendar_nights = self.percentiles(90, reference_period, "night")
        time_index = temperature_nights_max["time"].dt.dayofyear
        difference_array_90_nights = (temperature_nights_max - percentile_90_calendar_nights.sel(dayofyear=time_index)).drop("dayofyear")
        #difference_array_90_nights = temperature_nights_max - percentile_90_calendar_nights
        nights_90_above_thresholds = xr.where(difference_array_90_nights > 0, 1, 0)
        tn90 = nights_90_above_thresholds.resample(time='m').sum()/nights_90_above_thresholds.resample(time="m").count()

        return 0.5*(tx90 + tn90)
    

    def t10_month(self, reference_period):

        #Computing tx10

        temperature_days_min = self.temp_days_extremum("min")
        percentile_10_calendar_days = self.percentiles(10, reference_period, "day")
        time_index = temperature_days_min["time"].dt.dayofyear
        difference_array_10_days = (temperature_days_min - percentile_10_calendar_days.sel(dayofyear=time_index)).drop("dayofyear")
        #difference_array_90_nights = temperature_nights_max - percentile_90_calendar_nights
        days_10_above_thresholds = xr.where(difference_array_10_days < 0, 1, 0)
        tx10 = days_10_above_thresholds.resample(time='m').sum()/days_10_above_thresholds.resample(time="m").count()

        #Computing tn10
        

        temperature_nights_min = self.temp_nights_extremum("min")
        percentile_10_calendar_nights = self.percentiles(10, reference_period, "night")
        time_index = temperature_nights_min["time"].dt.dayofyear
        difference_array_10_nights = (temperature_nights_min - percentile_10_calendar_nights.sel(dayofyear=time_index)).drop("dayofyear")
        #difference_array_90_nights = temperature_nights_max - percentile_90_calendar_nights
        nights_10_above_thresholds = xr.where(difference_array_10_nights < 0, 1, 0)
        tn10 = nights_10_above_thresholds.resample(time='m').sum()/nights_10_above_thresholds.resample(time="m").count()

        return 0.5*(tx10 + tn10)
    

    def std_t90_month(self, reference_period, area=None):

        t_90 = self.t90_month(reference_period)

        t90_reference = t_90.sel(time=slice(reference_period[0], reference_period[1]))

        time_index = t_90.time.dt.month

        # Group the reference period data by year and month and calculate mean and standard deviation
        t90_mean = t90_reference.groupby("time.month").mean().sel(month=time_index)
        t90_std = t90_reference.groupby("time.month").std().sel(month=time_index)

        t90_z = ((t_90 - t90_mean)/t90_std).drop("month")

        area = False if area is None else area
        if (not area):
            return t90_z
        else:
            return t90_z.mean(dim=['latitude', 'longitude'])
    
    
    def std_t10_month(self, reference_period, area=None):

        t10 = self.t10_month(reference_period)

        t10_reference = t10.sel(time=slice(reference_period[0], reference_period[1]))

        time_index = t10.time.dt.month

        # Group the reference period data by year and month and calculate mean and standard deviation
        t10_mean = t10_reference.groupby("time.month").mean().sel(month=time_index)
        t10_std = t10_reference.groupby("time.month").std().sel(month=time_index)

        t10_z = ((t10 - t10_mean)/t10_std).drop("month")

        area = False if area is None else area
        if (not area):
            return t10_z
        else:
            return t10_z.mean(dim=['latitude', 'longitude'])

        return t10_z



    def plot_components(self, component_data0, component_data1, n):
        """
        Plot seasonal components of temperature.

        Parameters:
        - component_data (xarray.DataArray): DataArray containing the seasonal temperature component.
        """
        # Assuming 'component_data' contains seasonal temperature component data
        component_data0["t2m"].rolling(time=n,center=True).mean().plot()
        component_data1["t2m"].rolling(time=n,center=True).mean().plot()