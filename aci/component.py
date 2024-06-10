import xarray as xr






class Component:
    def __init__(self, array, mask, file_name):
        self.array = array
        self.mask = mask
        self.file_name = file_name

    def apply_mask(self, var_name, threshold=0.8):
        if self.array is None or self.mask is None:
            raise ValueError("Data not loaded. Please ensure precipitation and mask data are loaded.")
        
        f_temp = self.array.copy()
        f_temp['mask'] = self.mask.country
        
        # Create a mask based on the threshold
        country_mask = f_temp['mask'] >= threshold
        
        # Apply the mask to the precipitation data
        f_temp[var_name] = xr.where(country_mask, f_temp[var_name], float('nan'))
        
        return f_temp.drop_vars('mask')







