from dataloader import add_coords

import numpy as np
from sklearn.impute import KNNImputer


class KNNFiller():

    def __init__(self, k):
        self._knn = KNNImputer(n_neighbors=k,
                               weights="uniform",
                               metric="nan_euclidean",
                               copy=False)

    def fill(self, df, par):
        if not np.iterable(par):
            par = [par]
        to_fill = df
        df = add_coords(df)
        index = np.unique(df["day_index"])
        df = df.set_index("day_index")
        for p in par:
            df_tmp = df[["lat", "lon", p]]
            for i in index:
                self._knn.fit_transform(df_tmp.loc[i, :])
            df_tmp = df_tmp.reset_index()
            to_fill[p] = df_tmp[p]
