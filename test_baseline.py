import pandas as pd
import datetime
from chrono import chrono


@chrono
def test():
    path = "./Train/X_station_train.csv"
    first_date = datetime.datetime(2016, 1, 1)
    last_date = datetime.datetime(2017, 12, 31)

    # read the ground station data
    def read_gs_data(fname):
        gs_data = pd.read_csv(fname, parse_dates=[
                              "date"], infer_datetime_format=True)
        gs_data = gs_data.sort_values(by=["number_sta", "date"])
        return gs_data

    x = read_gs_data(path)
    x["number_sta"] = x["number_sta"].astype("category")

    # sort by station, then by date
    x = x.sort_values(["number_sta", "date"])

    # get the observation baseline
    Base_obs = x[{"number_sta", "date", "precip"}]
    Base_obs.set_index("date", inplace=True)

    # compute the accumulated rainfall per day with nan management
    # if any NaN on the day, then the value is NaN (24 values per day)
    Base_obs = Base_obs.groupby("number_sta").resample(
        "D").agg(pd.Series.sum, min_count=24)
    Base_obs = Base_obs.reset_index(["date", "number_sta"])
    Base_obs["number_sta"] = Base_obs["number_sta"].astype("category")

    # Select the observations the day before
    Base_obs["baseline_obs"] = Base_obs.groupby(["number_sta"])[
        "precip"].shift(1)
    Base_obs = Base_obs.sort_values(by=["number_sta", "date"])
    del Base_obs["precip"]
    Base_obs = Base_obs.rename(columns={"baseline_obs": "precip"})

    # get the day indexes (to the final Id)
    date = first_date
    dates = []
    while date <= (last_date - datetime.timedelta(days=1)):
        date += datetime.timedelta(days=1)
        dates.append(date)

    d_dates = pd.DataFrame(dates, columns=["date"])
    d_dates["day_index"] = d_dates.index

    # create the ID column (id_station + month + index value)
    y_f = pd.merge(Base_obs, d_dates, how="right", on=["date"])
    y_f = y_f[y_f["date"] != last_date]
    y_f["Id"] = y_f["number_sta"].astype(str) + "_" + \
        y_f["day_index"].astype(str)

    # final post-processing
    del y_f["day_index"]
    y_f = y_f.rename(columns={"precip": "Prediction"})

    # if you want to save your baseline in a csv file
    # output_file = "my_baseline_obs.csv"
    # to save the baseline in a csv file
    # y_f.to_csv("/kaggle/working/" + output_file,index=False)

    return y_f
