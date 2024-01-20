import tensorflow as tf
import keras
from keras import layers
import random
from environment import Game2048
from math import pow

# initializing the model
model = keras.Sequential()

model.add(keras.Input(shape=(4, 4)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
# change from 4 to 5 if clear implemented
model.add(layers.Dense(1, activation="linear"))

# model.summary()

model.compile(optimizer="adam", loss=keras.losses.MSE)


# copy model and weights
def modelCopy(model):
    model2 = keras.models.clone_model(model)
    model2.compile(optimizer="adam", loss=keras.losses.MSE)
    model2.set_weights(model.get_weights())
    return model2


def main():
    # defining parametres
    # copy defines how often we copy the policy network into target network (model to model2)
    iterations, limit = (0, 10)
    epsilon = 0.1
    env = Game2048(4, 4)
    batchSize = 1
    copy = 10
    model2 = modelCopy(model)
    gamma = 0.9

    sarsa = []

    while iterations < limit:
        env.reset()
        playing = True
        counter = 0
        while playing:
            state = env.state()
            Q = model(tf.convert_to_tensor(state))
            # implement epsilon greedy here
            action = tf.argmax(Q)[0]
            env.move(action)
            lost = counter < 100
            if lost:
                playing = False

            reward = env.score
            nextState = env.state()
            nextAction = tf.argmax(model(tf.convert_to_tensor(nextState)))

            sarsa.append([state, action, reward, nextState, nextAction])

            # for later to change batch size, train model then empty the sarsa
            if counter % batchSize == 0 or not playing:
                target = 0
                # adds upp all the rewards in the sarsa list and then the last max Q value
                for i, list in enumerate(sarsa):
                    if i != len(sarsa) - 1:
                        target += pow(gamma, i) * list[2]
                    if i == len(sarsa) - 1:
                        # print(model2(tf.convert_to_tensor(list[3])))
                        target += pow(gamma, i) * max(
                            model2(tf.convert_to_tensor(list[3]))
                        )

                # update model
                print(max(model2(tf.convert_to_tensor(sarsa[0][0]))))
                model.fit(
                    max(model2(tf.convert_to_tensor(sarsa[0][0]))),
                    target,
                    batch_size=batchSize,
                )

                # resets sarsa
                sarsa = []

            if counter % copy == 0:
                model2 = modelCopy(model)

            counter += 1
        iterations += 1


if __name__ == "__main__":
    main()
