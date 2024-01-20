import tensorflow as tf
import keras
from keras import layers
import random
from environment import Game2048

# initializing the model
model = keras.Sequential()

model.add(keras.Input(shape=(4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
# change from 4 to 5 if clear implemented
model.add(layers.Dense(4, activation="linear"))

model.summary()

model.compile(optimizer="adam", loss=keras.losses.MSE)


def main():
    # defining parametres
    iterations, limit = (0, 10000)
    epsilon = 0.1
    env = Game2048(4, 4)
    batchSize = 1

    sarsa = []

    while iterations < limit:
        state = env.state()
        Q = model(tf.convert_to_tensor(state))
        # implement epsilon greedy here
        action = tf.argmax(Q)

        env.move(action)
        reward = env.score()
        nextState = env.state()
        nextAction = tf.argmax(model(tf.convert_to_tensor(nextState)))

        sarsa.append([state, action, reward, nextState, nextAction])

        # for later to change batch size, train model then empty the sarsa
        if iterations % batchSize == 0:
            for list, i in enumerate(sarsa):
                pass

            sarsa = []

        iterations += 1


if __name__ == "__main__":
    main()
