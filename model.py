import numpy as np
import pandas as pd

from chrono import chrono, start, stop
from dataloader import add_id

from datamanip import KNNFiller
from sklearn.preprocessing import StandardScaler


PRED = "Prediction"
TRUTH = "Ground_truth"
MODEL = ["ff", "t", "td", "hu", "dd", "month", "precip"]
INDEX = ["number_sta", "day_index", "hour"]


class Model():

    def __init__(self, model, *,
                 use_scaler=False,
                 filler=None):
        self.model = model
        self._use_scaler = use_scaler
        self._filler = filler

    # TODO: Faire en sorte que le mois soit pas 24 fois par ligne
    def _vectorize(X):
        X = X.set_index(INDEX).sort_index()
        X = X[MODEL]
        X = X.unstack()
        X = X.reset_index()
        add_id(X)
        return X

    def _flatten_columns(df):
        df.columns = ["".join(map(str, col)) for col in df.columns]

    @chrono
    def fit(self, X, Y):
        # Scale the input
        if self._use_scaler:  # TODO: faire séparament avec un df intermédiaire
            self._X_scaler = StandardScaler()
            self._Y_scaler = StandardScaler()
            X[MODEL] = self._X_scaler.fit_transform(X[MODEL])
            Y[TRUTH] = self._Y_scaler.fit_transform(Y[TRUTH])

        X = Model._vectorize(X)
        Y = Y[["Id", TRUTH]]

        # Make sure Xi predict the correct Yi
        Y = pd.DataFrame(
                Y.values,
                columns=pd.MultiIndex.from_product([Y, [""]])
            ).sort_index(axis=1)
        df = X.merge(Y, on="Id").dropna()
        X = df[MODEL]
        Y = df[TRUTH]
        Model._flatten_columns(X)

        # Fit the sklearn model
        self.model.fit(X, Y)
        return self

    @chrono
    def predict(self, X):
        # Scale the input
        if self._use_scaler:  # TODO: utiliser un df intermédiaire
            X[MODEL] = self._X_scaler.transform(X[MODEL])

        X = Model._vectorize(X)
        ids = X[["Id"] + INDEX[: -1]]
        X = X[MODEL].dropna()
        Model._flatten_columns(X)
        Model._flatten_columns(ids)

        Y = self.model.predict(X)

        # Prepare the output
        df = pd.DataFrame(ids)
        df[PRED] = np.nan
        df.loc[X.index, PRED] = Y

        # Fill the missing values
        if self._filler is not None:
            self._filler.fill(df, PRED)

        # Scale the output
        if self._use_scaler:
            df[PRED] = self._Y_scaler.inverse_transform(df[PRED])

        return df[["Id", PRED]]


def MAPE(Y_truth, Y_pred, *, left=TRUTH, right=PRED):
    df = Y_truth.merge(Y_pred, on="Id", how="left")
    truth = df[left]
    pred = df[right]
    score = truth - pred
    score /= pred + 1
    score = np.abs(score)
    score = 100 * np.mean(score)
    return score


@chrono
def make_model(model, X_train, Y_train, X_test, Y_test, k=0, **kwargs):
    knn = None if k == 0 else KNNFiller(k)
    print()

    start("train phase")
    model = Model(model, filler=knn, **kwargs)
    model.fit(X_train, Y_train)
    pred_train = model.predict(X_train)
    train_err = MAPE(Y_train, pred_train)
    stop()
    print()

    start("test phase")
    pred_test = model.predict(X_test)
    test_err = MAPE(Y_test, pred_test)
    stop()
    print()

    return {"model": model,
            "pred train": pred_train,
            "train error": train_err,
            "pred test": pred_test,
            "test error": test_err}
