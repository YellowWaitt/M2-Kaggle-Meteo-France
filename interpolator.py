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
        to_fill = df
        df = add_coords(df)
        df = df[["day_index", "lat", "lon", par]]
        index = np.unique(df["day_index"])
        df = df.set_index("day_index").sort_index()
        for i in index:
            self._knn.fit_transform(df.loc[i])
        df = df.reset_index()
        to_fill[par] = df[par]
