
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from scipy.signal import find_peaks

import gym
import argparse
import numpy as np
from collections import deque
import random
import math
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--atoms', type=int, default=51)
parser.add_argument('--v_min', type=float, default=-5.)
parser.add_argument('--v_max', type=float, default=5.)

args = parser.parse_args()


class DataSet:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionValueModel:
    def __init__(self, state_dim, action_dim, z):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = args.atoms
        self.z = z

        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        return tf.keras.Model(input_state, outputs)

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, load_path):
        self.model = tf.keras.models.load_model(load_path)

    def train(self, x, y):

        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        if np.argmax(q) == 0:
            selected = True
        else:
            selected = False
        plot_histogram(z[0], 51, self.z, 'one', selected, q[0])
        plot_histogram(z[1], 51, self.z, 'two', not(selected), q[1])
        return np.argmax(q)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = DataSet()
        self.batch_size = args.batch_size
        self.v_max = args.v_max
        self.v_min = args.v_min
        self.atoms = args.atoms
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.gamma = args.gamma
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.z)
        self.q_target = ActionValueModel(
            self.state_dim, self.action_dim, self.z)
        self.target_update()
        self.file_writer = tf.summary.create_file_writer('runs' + "/metrics6")
        self.file_writer.set_as_default()
        self.step = 0

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self, steps):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        for i in range(z[0].shape[0]):
            tf.summary.histogram('Z1', data=z[0][i], step=self.step+i, buckets=51)
            tf.summary.histogram('Z2', data=z[1][i], step=self.step+i, buckets=51)
        for i in range(z[0].shape[0]):
            tf.summary.histogram('Z_1', data=z_[0][i], step=self.step+i, buckets=51)
            tf.summary.histogram('Z_2', data=z_[1][i], step=self.step+i, buckets=51)
            self.step += z[0].shape[0]
        # print('writing....')
        self.file_writer.flush()
        # z_concat = np.vstack(z)
        # q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        next_actions = np.argmax(np.sum(np.multiply(z, np.array(self.z)), axis=2), axis=0)
        # q = q.reshape((self.batch_size, self.action_dim), order='F')
        # next_actions = np.argmax(q, axis=1)
        m_prob = [np.zeros((self.batch_size, self.atoms))
                  for _ in range(self.action_dim)]
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(
                        self.v_min, rewards[i] + self.gamma * z[next_actions[i]][i][j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(
                        l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(
                        u)] += z_[next_actions[i]][i][j] * (bj - l)
            # plot_histogram1(z[i], 51, self.z, 'one', m_prob, q[0])
            # plot_histogram(z[1], 51, self.z, 'two', not (selected), q[1])
        loss = self.q.train(states, m_prob)
        tf.summary.scalar('loss', loss, steps)

    def train(self, max_epsiodes=500):
        for ep in range(max_epsiodes):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
                action = self.q.get_action(state, ep)
                self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, -
                                1 if done else 0, next_state, done)
                self.replay(steps)

                if self.buffer.size() > 1000:
                    # self.replay(steps)
                    pass
                if steps % 5 == 0:
                    # self.target_update()
                    pass

                state = next_state
                total_reward += reward
                steps += 1
            print('EP{} reward={}'.format(ep, total_reward))
            # if ep%100 == 0:
            #     self.q.save_model('runs/'+str(ep)+'/val_model')
            #     self.q_target.save_model('runs/'+str(ep)+'/target_model')
        # self.q.save_model('runs/'+str(max_epsiodes)+'/val_model')
        # self.q_target.save_model('runs/'+str(max_epsiodes)+'/target_model')

    def load_model(self):
        self.q.load_model('runs/500'+'/val_model')
        self.q_target.load_model('runs/500'+'/target_model')


def plot_histogram(x, bins, delta_z, action_number, selected, action_value):
    if find_peaks(x.flatten(), distance=3):
        pass
    if find_peaks(np.multiply(x.flatten(),delta_z), distance=3):
        pass
    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)
    axs[1].hist(x.flatten(), bins=bins)
    axs[1].set_title('neural net output histogram')
    axs[0].bar(np.arange(51), x.flatten())
    axs[0].set_title('raw neural net output values (1 - 51)')
    axs[2].bar(delta_z, np.multiply(x.flatten(),delta_z))
    axs[2].set_title('values after projection')
    if selected:
        fig.suptitle('values for action number '+str(action_number)+': '+str(action_value), fontweight='bold')
    else:
        fig.suptitle('values for action number ' + str(action_number)+': '+str(action_value))

def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.load_model()
    agent.train()

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='runs' + "/metrics", histogram_freq=1)


if __name__ == "__main__":
    main()
