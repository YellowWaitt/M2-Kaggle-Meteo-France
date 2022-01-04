import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import functools as ft


def compute_mean_std(df):
    df2 = df.drop(columns=["number_sta", "day_index"])
    mean, std = df2.mean(), df2.std()
    return mean, std


def scale(df, mean, std):
    df2 = df.drop(columns=["number_sta", "day_index"])
    df2 = (df2 - mean) / std
    df2[["number_sta", "day_index"]] = df[["number_sta", "day_index"]]
    return df2


def rescale(df, mean, std):
    df2 = df.drop(columns=["number_sta", "day_index"])
    df2 = df2 * std + mean
    df2[["number_sta", "day_index"]] = df[["number_sta", "day_index"]]
    return df2


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df=None,
                 label_columns=None):
        # Store them here for future predictions
        self.mean, self.std = compute_mean_std(train_df)

        # Store the raw data.
        self.train_df = self.scale(train_df)
        self.val_df = self.scale(val_df)
        self.test_df = None if test_df is None else self.scale(test_df)

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.columns_in = [col for col in train_df.columns
                           if col not in ["number_sta", "day_index"]]
        self.column_indices = {name: i for i, name in
                               enumerate(self.columns_in)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return "\n".join([
            f"Total window size: {self.total_window_size}",
            f"Input indices: {self.input_indices}",
            f"Label indices: {self.label_indices}",
            f"Label column name(s): {self.label_columns}"])

    def scale(self, df):
        return scale(df, self.mean, self.std)

    def rescale(self, df):
        return rescale(df, self.mean, self.std)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn"t preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="precip", max_subplots=3, title=None):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(self.input_indices,
                     inputs[n, :, plot_col_index],
                     label="Inputs",
                     marker=".",
                     zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices,
                        labels[n, :, label_col_index],
                        edgecolors="k",
                        label="Labels",
                        c="#2ca02c", s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker="X",
                            edgecolors="k",
                            label="Predictions",
                            c="#ff7f0e", s=64)

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        plt.suptitle(title)
        plt.show()

    def make_dataset(self, datas, *, training=True):
        ds_list = []
        groups = datas.groupby("number_sta")

        if training:
            sequence_length = self.total_window_size
            stride = 1
            shuffle = True
            split = True
        else:
            sequence_length = self.input_width
            stride = self.input_width
            shuffle = False
            split = False

        for (n_sta, data) in groups:
            data = data.drop(columns=["number_sta", "day_index"])
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=sequence_length,
                sequence_stride=stride,
                shuffle=shuffle,
                batch_size=32,
            )
            ds_list.append(ds)

        ds = ft.reduce(lambda ds1, ds2: ds1.concatenate(ds2), ds_list)
        if split:
            ds = ds.map(self.split_window)
        if shuffle:
            ds.shuffle(len(ds), reshuffle_each_iteration=False)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, training=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, training=True)

    @property
    def test(self):
        if self.test_df is None:
            return None
        return self.make_dataset(self.test_df, training=False)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    @property
    def num_features(self):
        return len(self.column_indices)


def make_window(input_width, label_width, shift, df, label_columns=None,
                date_train="2017-07-01", date_val="2017-11-01"):

    slice_train = df["date"] < date_train
    slice_val = (date_train <= df["date"]) & (df["date"] < date_val)
    slice_test = date_val <= df["date"]

    df = df.select_dtypes("number")

    train_df = df[slice_train]
    val_df = df[slice_val]
    test_df = df[slice_test]

    window = WindowGenerator(input_width, label_width, shift,
                             train_df, val_df, test_df,
                             label_columns=label_columns)
    return window


def make_pred(model, window, df):
    df = df.sort_values(["number_sta", "day_index", "hour"])
    X = df.select_dtypes("number")
    X = window.scale(X)
    data = window.make_dataset(X, training=False)

    pred = model.predict(data, verbose=1, use_multiprocessing=True, workers=-1)
    pred = pd.DataFrame(pred.reshape((-1, pred.shape[-1])),
                        columns=window.columns_in)

    pred["number_sta"] = df["number_sta"].to_numpy()
    pred["day_index"] = df["day_index"].to_numpy()
    pred = window.rescale(pred)

    pred = pred[["number_sta", "day_index", "precip"]]
    where = pred["precip"] < 0
    pred.loc[where, "precip"] = 0
    groups = pred.groupby(["number_sta", "day_index"])
    pred = groups.agg("sum").reset_index()
    pred["Id"] = \
        pred["number_sta"].astype(str) + "_" + pred["day_index"].astype(str)

    return pred[["Id", "precip"]]


def MAPE(Y_truth, Y_pred, *, left="Prediction", right="precip"):
    df = Y_truth.merge(Y_pred, on="Id", how="left")
    truth = df[left]
    pred = df[right]
    score = truth - pred
    score /= pred + 1
    score = np.abs(score)
    score = 100 * np.mean(score)
    return score
