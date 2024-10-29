from importlib import resources
import pandas as pd

def load_psmsl_data():
    with resources.path("aci.data", "psmsl_data.csv") as f:
        data_file_path = f
        df = pd.read_csv(data_file_path)
    return df