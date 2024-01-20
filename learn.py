import tensorflow as tf
import keras
from keras import layers
import random

#initializing the model
model = keras.Sequential()

model.add(keras.Input(shape=(4,4)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="linear"))

model.summary()

model.compile(optimizer="adam", loss=keras.losses.MSE)



def main():
        #defining parametres
        iterations, limit = (0, 10000)
        epsilon = 0.1



if __name__ == "__main__":
        main()
