"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
"""

import requests
import pandas as pd

# download_file(url)
def download_file(url=None):
    """ Download the .csv file from the given link and read as dataframe

    Args: 
        url: str
    
    Returns:
        DataFrame
    """
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return pd.read_csv(local_filename)

