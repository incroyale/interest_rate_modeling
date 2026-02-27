import pandas as pd
from fredapi import Fred


API_KEY = "5b3b2e40a6372b517e0c74d556b0ca97"
series = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]


def fetch_treasury_data(series=series, start_date=None, end_date=None):
    """
    Get Treasury data from FRED,
    :params: start_date and end_date
    :returns: df
    """
    fred = Fred(api_key=API_KEY)
    data_frames = []
    for i in series:
        s = fred.get_series(i, observation_start=start_date, observation_end=end_date)
        s.name = i
        data_frames.append(s)
    df = pd.concat(data_frames, axis=1)
    df = df[series]
    return df


def preprocess_data(df):
    """
    Preprocess data, remove NaNs and change maturities: string -> T (float)
    :params: FRED Treasury Dataframe
    :returns: df
    """
    df.dropna(inplace=True, axis=0)
    if (df.isna().sum().sum() == 0) and len(df.columns) == len(series):
        print("All checks Complete")
    else:
        return "Error in df"
    new_headings = [1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    df.columns = new_headings
    return df