import pandas as pd
import xarray as xr
from pandas.tseries.offsets import MonthEnd
from typing import List
from functools import reduce

def reduce_dataarray_to_dataframe(array : xr.DataArray, column_name:str=None) -> pd.DataFrame :
    """
        Reduce xarray DataArray into a pandas dataframe, change the variable name and adjust time index

        Parameters:
            array (xr.DataArray): array to transform
            column_name (str): new name of the variable, None by default
        Returns:
            pd.DataFrame: a dataframe with chosen variable name and fixed time index

    """
    dataframe = array.to_dataframe()
    if column_name is not None:
        dataframe.columns = [column_name]
    dataframe.index = pd.to_datetime(dataframe.index, format="%Y-%m-%d") + MonthEnd(1)
    return dataframe

def reduce_sealevel_over_region(dataframe:pd.DataFrame) -> pd.DataFrame:
    """
        Reduce the sealevel dataframe into one variable as the mean of stations measures

        Parameters:
            dataframe (pd.DataFrame): sealevel dataframe to reduce

        Returns:
            pd.DataFrame: dataframe with a single variable.
    """
    sea_std_mean = dataframe.mean(axis=1)
    sea_df = pd.DataFrame(sea_std_mean, columns=["sealevel"])
    sea_df["time"] = pd.to_datetime(sea_df.index, format="%Y-%m-%d") + MonthEnd()
    sea_df.set_index("time", inplace=True)
    return sea_df

def merge_dataframes(dataframes:List[pd.DataFrame]) -> pd.DataFrame:
    """
        Merge all the dataframes from the list given

        Parameters:
            dataframes (List[pd.DataFrame]): list of dataframes to merge, with a common index

        Returns:
            pd.DataFrame: DataFrame containing the variables of all elements of the list
    """
    merge_df = reduce(lambda left, right : pd.merge(left, right, right_index=True, left_index=True), dataframes)
    return merge_df