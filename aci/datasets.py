from importlib import resources
import pandas as pd

def load_psmsl_data():
    data_file_path = resources.files("aci.data").joinpath("psmsl_data.csv")
    with data_file_path.open() as f:
        df = pd.read_csv(f)
    return df