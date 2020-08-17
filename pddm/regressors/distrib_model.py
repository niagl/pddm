from __future__ import print_function

import random
import numpy as np
from collections import deque
import math

import tensorflow as tf

from pddm.regressors.feedforward_network import distrib_network
from pddm.utils.helper_funcs import concat_distrib_datasets

class Distrib_Model:

    def __init__(self,
                 input_size,
                 state_size,
                 action_size,
                 sess ,
                 num_atoms,
                 params):

        # get size of state and action
        self.input_size = input_size
        self.state_size = state_size
        self.action_size = action_size
        self.output_size = num_atoms
        self.sess = sess

        # these is hyper parameters for the DQN
        self.params = params
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = self.params.batchsize
        self.tf_datatype = self.params.tf_datatype

        # Initialize Atoms
        self.num_atoms = num_atoms # 51 for C51
        self.v_max = 30 # Max possible score
        self.v_min = -10 # Min possible score
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Models for value distribution
        self.model = distrib_network(input_size, self.params.num_fc_layers, self.num_atoms,
                             self.params.depth_fc_layers, self.tf_datatype, scope='model')
        self.target_model = distrib_network(input_size, self.params.num_fc_layers, self.num_atoms,
                             self.params.depth_fc_layers, self.tf_datatype, scope='target_model')

    def get_value_dist(self, state, action):
        input = np.concatenate((state, action), axis=2)
        z = self.model.predict(input)
        dist_q = np.multiply(z, np.array(self.z))
        return np.sum(dist_q, axis=1)

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    # pick samples randomly from replay memory (with batch_size)
    def train(self,
              distrib_trainDataset_rand,
              distrib_trainDataset_onPol,
              nEpoch,
              distrib_valDataset_rand=None,
              distrib_valDataset_onPol=None
              ):

        ## currently doing only one epoch of training

        np.random.seed()

        if distrib_trainDataset_onPol.observations.shape[0]>0:
            distrib_trainDataset = concat_distrib_datasets(distrib_trainDataset_rand,
                                                  distrib_trainDataset_onPol)
        else:
            distrib_trainDataset = distrib_trainDataset_rand

        # num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        # replay_samples = random.sample(self.memory, num_samples)

        states = distrib_trainDataset.observations
        next_states = distrib_trainDataset.next_states
        actions = distrib_trainDataset.actions
        done = distrib_trainDataset.done
        reward = distrib_trainDataset.reward
        m_prob = np.zeros(self.num_atoms)

        model_input = np.concatenate((np.array(states), np.array(actions)), axis=2)
        z = self.model.predict(model_input)  # Return a list [32x51, 32x51, 32x51]
        z_ = self.target_model.predict(model_input)  # Return a list [32x51, 32x51, 32x51]

        num_samples = distrib_trainDataset.observations.shape[0]

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[int(l)] += (u - bj)
                m_prob[int(u)] += (bj - l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * z[i][j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[int(l)] += z_[i][j] * (u - bj)
                    m_prob[int(u)] += z_[i][j] * (bj - l)

        loss = self.model.fit(np.concatenate((states, actions), axis=2), m_prob, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
