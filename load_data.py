import pandas as pd

def load_data(file_path):
    """
    Loads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(file_path)
