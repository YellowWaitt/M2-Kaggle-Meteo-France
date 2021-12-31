import tensorflow as tf
import pandas as pd

from dataloader import load_test
from chrono import start, stop, chrono

from windowgenerator import make_window, make_pred


# %%

start("making RNN")

# TODO: il y a quelques améliorations faciles à faire sur les données
# que je prendrai pas le temps de faire
# Se débrouiller pour boucher les trous dans le test aussi
X_train = pd.read_csv("./ignore/train_filled.csv")
X_test, _, _ = load_test()


# %%

multi_window = make_window(24, 24, 24, X_train,
                           date_train="2017-11-01", date_val="2018",
                           label_columns=["precip"])

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(24 * multi_window.num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([24, multi_window.num_features])
])


# %%

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


history = compile_and_fit(multi_lstm_model, multi_window)
multi_lstm_model.save_weights("multi_lstm")


# %%

start("prediction")
pred = make_pred(multi_lstm_model, multi_window, X_test)
stop()

pred.to_csv("pred.csv")

stop()
