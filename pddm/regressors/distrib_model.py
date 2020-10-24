# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import ray

#my imports
from pddm.regressors.feedforward_network import distrib_network
from pddm.utils.helper_funcs import concat_distrib_datasets
from pddm.utils.calculate_prob import calculate_m_prob


class Distrib_Model:
    """
    This class implements: init, train, get_loss, do_forward_sim
    """

    def __init__(self,
                 input_size,
                 action_size,
                 sess,
                 params):

        # get size of state and action
        self.input_size = input_size
        self.action_size = action_size
        self.sess = sess


        # these is hyper parameters for the DQN
        self.params = params
        self.gamma = 0.99
        self.output_size = self.params.atoms
        self.learning_rate = self.params.dist_lr
        self.batch_size = self.params.batchsize
        self.tf_datatype = self.params.tf_datatype
        self.print_minimal = self.params.print_minimal
        self.tau = self.params.dist_target_model_update_tau

        # Initialize Atoms
        self.num_atoms = self.params.atoms  # 51 for C51
        self.use_given_Vmax_Vmin = self.params.use_given_Vmax_Vmin
        self.v_max = self.params.Vmax  # Max possible score
        self.v_min = self.params.Vmin  # Min possible score
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        self.num_proc = 64

        ## create placeholders
        self.create_placeholders()

        ## clip actions
        first, second = tf.split(self.inputs_, [(self.input_size - self.action_size), self.action_size], 1)
        second = tf.clip_by_value(second, -1, 1)
        self.inputs_clipped = tf.concat([first, second], axis=1)

        ## define forward pass
        self.define_forward_pass()
        ray.init(num_cpus=self.num_proc)

    def create_placeholders(self):
        self.inputs_ = tf.placeholder(
            self.tf_datatype,
            shape=[None, self.input_size],
            name='nn_inputs_')

        self.m_prob_ = tf.placeholder(
            self.tf_datatype,
            shape=[None, self.num_atoms],
            name='nn_m_prob')


    def define_forward_pass(self):
        # optimizer
        self.opt = tf.train.AdamOptimizer(self.params.dist_lr)

        # forward pass through this network
        this_output = distrib_network(
            self.inputs_clipped, self.params.dist_num_fc_layers, self.output_size,
            self.params.dist_depth_fc_layers, self.tf_datatype, scope='model')

        target_output = distrib_network(
            self.inputs_clipped, self.params.dist_num_fc_layers, self.output_size,
            self.params.dist_depth_fc_layers, self.tf_datatype, scope='target_model')

        self.predicted_output = this_output
        self.target_output = target_output

        m_prob = self.m_prob_
        m_prob = tf.stop_gradient(m_prob)
        loss = tf.keras.losses.categorical_crossentropy(m_prob, this_output)

        # this network's weights
        model_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')

        # train step for this network
        gv = [(g, v) for g, v in self.opt.compute_gradients(
            loss, model_vars) if g is not None]

        self.train_steps = self.opt.apply_gradients(gv)
        self.loss = loss

        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_model')

        soft_updates = []
        for var, target_var in zip(model_vars, target_vars):
            soft_updates.append(tf.assign(target_var, (1. - self.tau) * target_var + self.tau * var))

        self.update_op = tf.group(*soft_updates)


    def get_value_dist(self, state, action):
        # state , action : dims b x n
        val_dist = self.sess.run(
            [
                self.predicted_output
            ],
            feed_dict={
                self.inputs_:  np.concatenate((np.array(state), np.array(action)), axis=1)
            })

        val_dist_ = np.sum(np.multiply(val_dist[0], np.array(self.z)), axis = 1)
        return val_dist_

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.sess.run(self.update_op)

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
        start = time.time()
        training_loss_list = []
        actual_rewards_list = []
        predicted_val_dist_list = []
        m_prob_list = []
        predicted_reward_list = []

        if distrib_trainDataset_onPol.observations.shape[0]>0:
            distrib_trainDataset = concat_distrib_datasets(distrib_trainDataset_rand,
                                                  distrib_trainDataset_onPol)
        else:
            distrib_trainDataset = distrib_trainDataset_rand

        # dims
        nData_rand = distrib_trainDataset_rand.observations.shape[0]
        nData_onPol = distrib_trainDataset_onPol.observations.shape[0]
        nData = nData_rand + nData_onPol

        print('    Total number of data points training for: ', distrib_trainDataset.observations.shape[0])

        states = distrib_trainDataset.observations
        actions = distrib_trainDataset.actions
        done = distrib_trainDataset.done
        reward = distrib_trainDataset.reward

        model_input = np.concatenate((np.array(states), np.array(actions)), axis=1)

        for epoch_iter in range(nEpoch):
            epoch_start = time.time()
            # reset tracking variables to 0
            sum_training_loss = 0
            num_training_batches = 0

            ##############################
            ####### training loss
            ##############################

            # randomly order indices (equivalent to shuffling)
            range_of_indices = np.arange(model_input.shape[0])
            all_indices = npr.choice(
                range_of_indices, size=(model_input.shape[0],), replace=False)

            for batch in range(int(math.floor(nData / self.batch_size))):
                # batch_start = time.time()
                # walk through the shuffled new data
                model_inputs_batch = model_input[
                    all_indices[batch * self.batch_size:(batch + 1) *
                                                       self.batch_size]]  # [bs x K x dim]
                rewards_batch = reward[all_indices[
                                                  batch * self.batch_size:(batch + 1) * self.
                                                      batch_size]]  # [bs x dim]

                done_batch = done[all_indices[
                                             batch * self.batch_size:(batch + 1) * self.
                                                 batch_size]]  # [bs x dim]

                # one iteration of feedforward training
                predicted_output, target_output = self.sess.run(
                    [
                        self.predicted_output, self.target_output
                    ],
                    feed_dict={
                        self.inputs_: model_inputs_batch
                    })

                num_samples = model_inputs_batch.shape[0]

                if not self.use_given_Vmax_Vmin:
                    self.v_max = np.max(rewards_batch)
                    self.v_min = np.min(rewards_batch)
                    self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
                    self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

                env_dict = dict()
                env_dict['v_max'] = self.v_max
                env_dict['v_min'] = self.v_min
                env_dict['num_atoms'] = self.num_atoms
                env_dict['delta_z'] = self.delta_z
                env_dict['gamma'] = self.gamma
                env_dict['z'] = self.z

                batch_length = int(num_samples / self.num_proc)
                index_ = [batch_length * x for x in range(self.num_proc)]
                index_.append(num_samples)

                done_id = ray.put(done_batch)
                rewards_id = ray.put(rewards_batch)
                pred_output_id = ray.put(predicted_output)
                target_output_id = ray.put(target_output)
                env_dict_id = ray.put(env_dict)

                results_ = [calculate_m_prob.remote(done_id, rewards_id, pred_output_id, target_output_id, index_[x],
                                                   index_[x + 1], env_dict_id) for x in range(self.num_proc)]

                m_prob = np.concatenate(ray.get(results_), axis=0)

                # one iteration of gradient calculation and update
                _, losses = self.sess.run(
                    [
                        self.train_steps, self.loss
                    ],
                    feed_dict={
                        self.inputs_: model_inputs_batch,
                        self.m_prob_: m_prob
                    })

                loss = np.mean(losses)

                training_loss_list.append(loss)
                sum_training_loss += loss
                num_training_batches += 1
                # print('Loss: ',loss)
                # print("Time Taken this Batch {:0.2f} s".format(time.time() - batch_start))

            mean_training_loss = sum_training_loss / num_training_batches

            if ((epoch_iter % 10 == 0) or (epoch_iter == (nEpoch - 1))):
                actual_rewards_list.append(rewards_batch)
                predicted_val_dist_list.append(predicted_output)
                predicted_reward = np.sum(np.multiply(predicted_output, np.array(self.z)), axis=1)
                predicted_reward_list.append(predicted_reward)
                m_prob_list.append(m_prob)

            if not self.print_minimal:
                if ((epoch_iter % 10) == 0 or (epoch_iter == (nEpoch - 1))):
                    print("\n=== Epoch {} ===".format(epoch_iter))
                    print("    train loss: ", mean_training_loss)
                    # print("Time taken this Epoch: {:0.2f} s".format(time.time() - epoch_start))

        if not self.print_minimal:
            print("Training duration: {:0.2f} s".format(time.time() - start))

        lists_to_save = dict(
            training_loss_list=training_loss_list,
            actual_rewards_list=actual_rewards_list,
            predicted_val_dist_list=predicted_val_dist_list,
            predicted_reward_list=predicted_reward_list,
            m_prob_list=m_prob_list,)

        return mean_training_loss, lists_to_save
