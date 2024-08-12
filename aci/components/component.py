import xarray as xr
import warnings
import psutil
import socket
from dask.distributed import Client, LocalCluster


class Component:
    """
    Base class for components that handle various climate data
    and perform related calculations.

    Attributes:
    - array (xarray.Dataset): The dataset containing the primary data.
    - mask (xarray.Dataset): The dataset containing the mask data.
    - file_name (str): The file name of the dataset.
    - dask_client (dask.distributed.Client or None): Dask client for
    distributed computing.
    - use_dask (bool): Whether to use Dask for parallel computing.
    """

    def __init__(self, array, mask, file_name):
        """
        Initializes the Component with primary data and mask data.

        Args:
            array (xarray.Dataset): The dataset containing the primary data.
            mask (xarray.Dataset): The dataset containing the mask data.
            file_name (str): The file name of the dataset.
        """
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=(
                "is_monotonic is deprecated and will be removed in a "
                "future version. Use is_monotonic_increasing instead."
            )
        )

        self.use_dask = self.should_use_dask()
        self.chunk_size = self.determine_chunk_size(mask)
        self.array = array.chunk(self.chunk_size) if self.use_dask else array
        self.mask = mask.chunk(self.chunk_size) if self.use_dask else mask
        self.file_name = file_name

        # Initialize Dask client if applicable
        self.dask_client = self.initialize_dask() if self.use_dask else None

    def should_use_dask(self):
        """
        Determines whether to use Dask based on the system's resources.

        Returns:
            bool: True if Dask should be used, False otherwise.
        """
        total_memory = psutil.virtual_memory().total
        num_cores = psutil.cpu_count(logical=False)
        # Use Dask if there are multiple cores and more than 16GB of RAM
        return num_cores > 2 and total_memory > 16 * 1024**3

    def determine_chunk_size(self, dataset, memory_fraction=0.5):
        """
        Determines the optimal chunk size based on available system memory.

        Args:
            dataset (xarray.Dataset): The dataset to be chunked.
            memory_fraction (float): Fraction of total memory to use for
            chunking. Default is 0.5.

        Returns:
            dict: Dictionary defining the chunk sizes for each dimension.
        """
        total_memory = psutil.virtual_memory().total
        chunk_size = int(total_memory * memory_fraction / 8)
        chunk_dict = {dim: chunk_size for dim in dataset.dims if dim != 'time'}
        if 'time' in dataset.dims:
            chunk_dict['time'] = chunk_size
        return chunk_dict

    def initialize_dask(self):
        """
        Initializes a Dask client for distributed computing if multiple
        CPU cores are available.

        Returns:
            dask.distributed.Client or None: Dask client if initialized,
            otherwise None.
        """
        num_cores = psutil.cpu_count(logical=False)
        if num_cores > 1:
            port = 8787
            while self.port_in_use(port):
                port += 1

            cluster = LocalCluster(
                n_workers=num_cores,
                threads_per_worker=1,
                dashboard_address=f":{port}"
            )
            client = Client(cluster)
            return client
        return None

    def port_in_use(self, port):
        """
        Checks if a specific port is already in use.

        Args:
            port (int): Port number to check.

        Returns:
            bool: True if the port is in use, False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def apply_mask(self, var_name, threshold=0.8):
        """
        Apply a mask to the dataset.

        Args:
            var_name (str): Variable name in the dataset to which the mask
            is applied.
            threshold (float): Threshold value for the mask. Default is 0.8.

        Returns:
            xarray.Dataset: Dataset with the mask applied to the specified
            variable.
        """
        if self.array is None or self.mask is None:
            raise ValueError(
                "Data not loaded. Please ensure the data and mask are loaded.")

        f_temp = self.array.copy()
        f_temp['mask'] = self.mask.country
        country_mask = f_temp['mask'] >= threshold
        f_temp[var_name] = xr.where(
            country_mask, f_temp[var_name], float('nan'))
        return f_temp.drop_vars('mask')

    def standardize_metric(self, metric, reference_period, area=None):
        """
        Standardizes a given metric based on a reference period.

        Args:
            metric (xarray.DataArray): The metric to be standardized.
            reference_period (tuple): A tuple containing the start and end
            dates of the reference period.
            area (bool): If True, calculate the area-averaged standardized
            metric. Default is None.

        Returns:
            xarray.DataArray: The standardized metric.
        """
        reference = metric.sel(
            time=slice(reference_period[0], reference_period[1]))
        time_index = metric.time.dt.month
        mean = reference.groupby("time.month").mean().sel(month=time_index)
        std = reference.groupby("time.month").std().sel(month=time_index)
        standardized = ((metric - mean) / std).drop("month")
        if area:
            return standardized.mean(dim=['latitude', 'longitude'])
        else:
            return standardized

    def calculate_rolling_sum(self, var_name, window_size):
        """
        Calculates the rolling sum of a variable over a specified window size.

        Args:
            var_name (str): The variable name in the data to calculate the
            rolling sum.
            window_size (int): The size of the rolling window.

        Returns:
            xarray.DataArray: The rolling sum of the variable.
        """
        data = self.apply_mask(var_name)
        var = data[var_name]
        rolling_sum = var.rolling(time=window_size).sum(dim='time')
        return rolling_sum.compute() if not self.use_dask else rolling_sum
