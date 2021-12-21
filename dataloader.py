import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
import datetime as dt

from chrono import chrono


_ROOT_DIR = Path(__file__).parent.as_posix() + "/"


def process_id(df, dic, *, keep=True):
    def split_id(df):
        return np.array([list(map(int, id.split("_"))) for id in df["Id"]])

    def update(i):
        df[NAMES[i]] = ids[:, i]

    NAMES = ["number_sta", "day_index", "hour"]

    ids = split_id(df)
    if not NAMES[0] in df.columns:
        update(0)
    update(1)
    if ids.shape[1] == 3:  # Pour les X_station
        update(2)
        if "date" in df.columns:  # X_station_train
            df["month"] = np.array([date.month - 1 for date in df["date"]])
            # df[NAMES[1]] -= 1
        else:  # X_station_test
            df["month"] = dic[ids[:, 1]]
    else:
        df["month"] = dic[ids[:, 1]]

    if not keep:
        del df["Id"]

    return df


def add_id(df):
    df["Id"] = df["number_sta"].astype(str) + "_" + df["day_index"].astype(str)
    return df


@chrono
def load_train():
    def load_csv_with_dates(f_name):
        return pd.read_csv(f_name,
                           parse_dates=["date"],
                           infer_datetime_format=True)

    def load(f_name):
        df = process_id(load_csv_with_dates(f_name), dic)
        # del df["date"]
        return df

    dic = pd.date_range("20160102", periods=730).month - 1

    root = _ROOT_DIR + "Train/"
    X = load(root + "X_station_train.csv")
    Y = load(root + "Y_train.csv")

    directory = root + "Baselines/"
    fore = load(directory + "Baseline_forecast_train.csv")
    obs = load(directory + "Baseline_observation_train.csv")

    return X, Y, fore, obs


@chrono
def load_test():
    def load(f_name):
        return process_id(pd.read_csv(f_name), dic)

    root = _ROOT_DIR + "Test/"
    dic = np.array(pd.read_csv(root + "Id_month_test.csv")["month"])
    X = load(root + "X_station_test.csv")

    directory = root + "Baselines/"
    fore = load(directory + "Baseline_forecast_test.csv")
    obs = load(directory + "Baseline_observation_test.csv")

    return X, fore, obs


def load_coordinates():
    coords = pd.read_csv(_ROOT_DIR + "Other/stations_coordinates.csv")
    return coords


def load_masks():
    datas = xr.open_dataset(_ROOT_DIR + "Other/NW_masks.nc")
    return datas


def add_coords(df):
    if all([col in df.columns for col in ["lat", "lon"]]):
        return df

    coords = load_coordinates()
    df = df.merge(coords, on=["number_sta"], how="left")
    return df


def split_train(X, Y, date="2017-01-01"):
    def split(df, date):
        where = df["date"] < date
        train = df[where]
        test = df[~where]
        return train, test

    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, "%Y-%m-%d")

    X_train, X_test = split(X, date)
    Y_train, Y_test = split(Y, date + dt.timedelta(days=1))
    return X_train, Y_train, X_test, Y_test


# %% Load datas

if __name__ == "__main__":
    X, Y, fc_train, obs_train = load_train()
    X_train, Y_train, X_test, Y_test = split_train(X, Y)
    # X_test, fc_test, obs_test = load_test()
