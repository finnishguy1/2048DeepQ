import random
from math import pow
from collections import deque

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from environment import Game2048

# initializing the model


# copy model and weights
def modelCopy(model):
    model2 = keras.models.clone_model(model)
    model2.compile(optimizer="adam", loss=keras.losses.MSE)
    model2.set_weights(model.get_weights())
    return model2


def define_model():
    model = keras.Sequential()

    model.add(keras.Input(shape=(4, 4)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))

    # model.summary()

    model.compile(optimizer="adam", loss=keras.losses.MSE)
    return model


def make_model_compatible(state):
    return tf.convert_to_tensor(state)


def epsilon_greedy(env, model, epsilon):
    # returns an action using epsilon greedy policy
    if random.random() > epsilon:
        return np.argmax(model(make_model_compatible(env.state())))
    return env.get_random_action()


def do_step(env, model, epsilon):
    state = env.state()
    old_score = env.get_score()

    action = epsilon_greedy(env, model, epsilon)
    env.move(action)

    reward = env.get_score() - old_score
    next_state = env.state()
    return (state, action, reward, next_state)


def go_forward_c_steps(env, model, epsilon, c):
    cur_sars = deque()
    while not env.lost() and c > 0:
        cur_sars.append(do_step(env, model, epsilon))
        c -= 1
    return cur_sars


def random_batch(sars, size):
    return random.sample(sars, size)


def sample_processing(batch, model, model_prime):
    x, y = [], []
    for state, action, reward, next_state in batch:
        q = model(make_model_compatible(state)).numpy()
        x.append(state)
        vals = max(model_prime(make_model_compatible(next_state)))
        q[action] = vals + reward
        y.append(q)
    return (x, y)


def play(env, model, model_prime, epsilon, c, copy, batch_size):
    sars = deque()
    steps = 0

    while not env.lost():
        sars.extend(go_forward_c_steps(env, model, epsilon, c))
        steps += c

        # now we need to train/fit the model
        batch = random_batch(sars, min(batch_size, len(sars)))
        x, y = sample_processing(batch, model, model_prime)

        model.fit(
            make_model_compatible(x),
            make_model_compatible(y),
            batch_size=batch_size,
            epochs=1,
        )

        if steps >= copy:
            steps = 0
            model_prime = modelCopy(model)


def main():
    # defining parametres
    # copy defines how often we copy the policy network into target network (model to model2)
    iterations, limit = 0, 10
    epsilon = 0.1
    env = Game2048(4, 4)
    batch_size = 1
    copy = 20
    c = 10
    model = define_model()
    model_prime = modelCopy(model)

    sarsa = []
    while iterations < limit:
        env.reset()
        play(env, model, model_prime, epsilon, c, copy, batch_size)
        iterations += 1


if __name__ == "__main__":
    main()
