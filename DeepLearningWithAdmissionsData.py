from re import VERBOSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.util.deprecation import HiddenTfApiAttribute

admissions_df = pd.read_csv("admissions_data.csv")

# print(admissions_df.head())
# print(admissions_df.columns)

admissions_df.drop('Serial No.', inplace=True, axis=1)
# print(admissions_df.columns)

features = admissions_df.iloc[:, 0:-1]
labels = admissions_df.iloc[:, -1]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=42)


st = StandardScaler()
features_train_std = st.fit_transform(features_train)
features_test_std = st.transform(features_test)

"""features_train_std = pd.DataFrame(
    features_train_std, columns=features_train.columns)"""

"""features_test_std = pd.DataFrame(
    features_test_std, columns=features_test.columns)"""

# print(features_train_std.describe())


def design_model(features):
    modal = Sequential(name="Admissions_Network")
    input = InputLayer(input_shape=(features.shape[1],))
    modal.add(input)

    hidden_layer = Dense(16, activation='relu')
    modal.add(hidden_layer)
    modal.add(Dropout(0.1))

    hidden_layer_2 = Dense(8, activation='relu')
    modal.add(hidden_layer_2)
    modal.add(Dropout(0.2))
    modal.add(Dense(1))

    opt = keras.optimizers.Adam(learning_rate=0.1)
    modal.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return modal


# Using the model
model = design_model(features_train_std)

stop = EarlyStopping(monitor='val_loss', mode='min',
                     verbose=1,  patience=40)

history = model.fit(features_train_std, labels_train, epochs=100,
                    batch_size=3, verbose=1, validation_split=0.25, callbacks=[stop])

val_mse, val_mae = model.evaluate(
    features_test_std, labels_test.to_numpy(), verbose=0)

val_mse, val_mae = model.evaluate(
    features_test_std, labels_test.to_numpy(), verbose=0)

# view the MAE performance
print("MAE: ", val_mae)

# evauate r-squared score
y_pred = model.predict(features_test_std)

print(r2_score(labels_test, y_pred))

# plot MAE and val_MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()
