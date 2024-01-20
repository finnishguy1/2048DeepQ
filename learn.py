import tensorflow as tf
import keras
from keras import layers


#initializing the model
model = keras.Sequential()

model.add(keras.Input(shape=(4,4)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="linear"))

model.summary()

model.compile(optimizer="adam", loss=keras.losses.MSE)


