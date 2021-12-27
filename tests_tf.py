"""
https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from windowgenerator import WindowGenerator
from dataloader import load_train
from chrono import chrono


# %%

X, Y, _, _ = load_train()
# On prend qu"une seule station pour vérifier que ça fonctionne
df = X[X["number_sta"] == 22092001].drop(columns=["number_sta", "Id", "date"])
# On vire les nan pour les biens du test
df = df.dropna()

column_indices = {name: i for i, name in enumerate(df.columns)}
num_features = df.shape[1]

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# %% fit et eval

val_performance = {}
performance = {}


@chrono
def compile_and_fit(model, window, patience=2, max_epochs=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience,
                                                      mode="min")

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def evaluate(model, name, window):
    global val_performance, performance
    print("\n\n", name, "\n")

    history = compile_and_fit(model, window)

    val_performance[name] = model.evaluate(window.val)
    performance[name] = model.evaluate(window.test, verbose=0)

    window.plot(model, title=name)

    return history


def compare(ylabel=None):
    global val_performance, performance

    x = np.arange(len(performance))
    width = 0.3
    val_mae = [v[1] for v in val_performance.values()]
    test_mae = [v[1] for v in performance.values()]

    plt.ylabel(ylabel)
    plt.bar(x - 0.17, val_mae, width, label="Validation")
    plt.bar(x + 0.17, test_mae, width, label="Test")
    plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
    _ = plt.legend()
    plt.show()

    print("\n\n")
    for name, value in performance.items():
        print(f"{name:12s}: {value[1]:0.4f}")

    val_performance = {}
    performance = {}


# %% Modèles à une étapes et une sortie

single_step_window = WindowGenerator(1, 1, 1,
                                     train_df, val_df, test_df,
                                     label_columns=["precip"])
print(single_step_window)


# --------------------------------------------------------------------------- #

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


baseline = Baseline(label_index=column_indices["precip"])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance["Baseline"] = baseline.evaluate(single_step_window.val)
performance["Baseline"] = baseline.evaluate(single_step_window.test, verbose=0)

single_step_window.plot(baseline, title="Baseline")

# --------------------------------------------------------------------------- #

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

evaluate(linear, "Linear", single_step_window)

# --------------------------------------------------------------------------- #

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=1)
])

evaluate(dense, "Dense", single_step_window)

# --------------------------------------------------------------------------- #

CONV_WIDTH = 3
conv_window = WindowGenerator(CONV_WIDTH, 1, 1,
                              train_df, val_df, test_df,
                              label_columns=["precip"])
print(conv_window)

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

evaluate(multi_step_dense, "Multi step dense", conv_window)

# --------------------------------------------------------------------------- #

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=1),
])

evaluate(conv_model, "Conv", conv_window)

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(INPUT_WIDTH, LABEL_WIDTH, 1,
                                   train_df, val_df, test_df,
                                   label_columns=["precip"])
print(wide_conv_window)
wide_conv_window.plot(conv_model, title="Conv")

# --------------------------------------------------------------------------- #

wide_window = WindowGenerator(24, 24, 1,
                              train_df, val_df, test_df,
                              label_columns=["precip"])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

evaluate(lstm_model, "LSTM", wide_window)

# --------------------------------------------------------------------------- #

compare(ylabel="mean_absolute_error [precip, normalized]")

# %% Modèles à une étapes et plusieurs sorties

# `WindowGenerator` returns all features as labels if you
# don"t set the `label_columns` argument.
single_step_window = WindowGenerator(1, 1, 1, train_df, val_df, test_df)

wide_window = WindowGenerator(24, 24, 1, train_df, val_df, test_df)

# --------------------------------------------------------------------------- #

baseline = Baseline()
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance["Baseline"] = baseline.evaluate(wide_window.val)
performance["Baseline"] = baseline.evaluate(wide_window.test, verbose=0)

single_step_window.plot(baseline, title="Baseline")

# --------------------------------------------------------------------------- #

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])

evaluate(dense, "Dense", wide_window)

# --------------------------------------------------------------------------- #

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

evaluate(lstm_model, "LSTM", wide_window)


# --------------------------------------------------------------------------- #

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            num_features,
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ])
)

evaluate(residual_lstm, "Residual LSTM", wide_window)

# --------------------------------------------------------------------------- #

compare(ylabel="MAE (average over all outputs)")

# %% Modèles à plusieurs étapes


OUT_STEPS = 24
multi_window = WindowGenerator(24, OUT_STEPS, OUT_STEPS,
                               train_df, val_df, test_df)
print(multi_window)


# --------------------------------------------------------------------------- #

class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

val_performance["Last"] = last_baseline.evaluate(multi_window.val)
performance["Last"] = last_baseline.evaluate(multi_window.test, verbose=0)

multi_window.plot(last_baseline, title="Last")


# --------------------------------------------------------------------------- #

class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs


repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

val_performance["Repeat"] = repeat_baseline.evaluate(multi_window.val)
performance["Repeat"] = repeat_baseline.evaluate(multi_window.test, verbose=0)

multi_window.plot(repeat_baseline, title="Repeat")

# --------------------------------------------------------------------------- #

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

evaluate(multi_linear_model, "Linear", multi_window)

# --------------------------------------------------------------------------- #

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation="relu"),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

evaluate(multi_dense_model, "Dense", multi_window)

# --------------------------------------------------------------------------- #

CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

evaluate(multi_conv_model, "Conv", multi_window)

# --------------------------------------------------------------------------- #

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

evaluate(multi_lstm_model, "LSTM", multi_window)


# --------------------------------------------------------------------------- #

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


feedback_model = FeedBack(units=32,
                          out_steps=OUT_STEPS,
                          num_features=num_features)

evaluate(feedback_model, "AR LSTM", multi_window)

# --------------------------------------------------------------------------- #

compare(ylabel="MAE (average over all times and outputs)")
