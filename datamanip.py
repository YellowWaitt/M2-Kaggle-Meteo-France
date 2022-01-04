from dataloader import add_coords
from chrono import chrono

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


class KNNFiller():

    def __init__(self, k):
        self._knn = KNNImputer(n_neighbors=k,
                               weights="uniform",
                               metric="nan_euclidean")

    def fill(self, df, par):
        def _fill(g):
            for p in par:
                filled = self._knn.fit_transform(g[["lat", "lon", p]])
                g[p] = filled[:, -1]
            return g

        if not np.iterable(par):
            par = [par]
        col = df.columns
        df = add_coords(df)
        groups = df.groupby(["day_index", "hour"], group_keys=False)
        df = groups.apply(_fill)
        return df[col]


@chrono
def check_nb_hours(df):
    def fill_hours(g):
        def fill(p):
            to_add[p] = g[p].values[0]

        if len(g) == 24:
            return g
        to_add = pd.DataFrame(columns=g.columns)
        to_add["hour"] = [h for h in range(24) if h not in g["hour"].values]
        to_add["Id"] = to_add["number_sta"].astype(str) \
            + "_" + to_add["day_index"].astype(str) \
            + "_" + to_add["hour"].astype(str)
        fill("number_sta")
        fill("day_index")
        fill("month")
        g = g.append(to_add, ignore_index=True)
        g = g.sort_values("hour")
        return g

    groups = df.groupby(["number_sta", "day_index"], group_keys=False)
    df = groups.apply(fill_hours)
    return df
