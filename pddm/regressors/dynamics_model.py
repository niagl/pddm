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
# FIXME: port to torch
import tensorflow as tf
import torch
import time
import math

#my imports
from pddm.regressors.feedforward_network import feedforward_network


class Dyn_Model:
    """
    This class implements: init, train, get_loss, do_forward_sim
    """

    def __init__(self,
                 inputSize,
                 outputSize,
                 acSize,
                 params,
                 normalization_data=None):

        # init vars
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.acSize = acSize
        self.normalization_data = normalization_data

        # params
        self.params = params
        self.ensemble_size = self.params.ensemble_size
        self.print_minimal = self.params.print_minimal
        self.batchsize = self.params.batchsize
        self.K = self.params.K
        self.torch_datatype = self.params.torch_datatype

        ## create placeholders
        # self.create_placeholders()

        ## clip actions
        # because MPPI sometimes tries to predict outcome
        # of acs outside of range -1 to 1
        # first, second = torch.split(self.inputs_, [(inputSize - self.acSize), self.acSize], 3)
        # second = torch.clamp(second, -1, 1)
        # self.inputs_clipped = torch.cat([first, second], dim=3)

        ## define forward pass
        # self.define_forward_pass()


    def define_forward_pass(self):

        #optimizer
        # self.opt = torch.optim.Adam(lr=self.params.lr)

        # self.mses = []
        # self.train_steps = []

        self.networks = []

        for i in range(self.ensemble_size):

            # forward pass through this network
            # this_output = feedforward_network(
            #     self.inputs_clipped[i], self.outputSize,
            #     self.params.num_fc_layers, self.params.depth_fc_layers)

            # TODO: convert input states from tenspr to int for inputSize
            self.networks.append(feedforward_network(
                self.inputSize, self.outputSize,
                self.params.num_fc_layers, self.params.depth_fc_layers))
            # self.curr_nn_outputs.append(this_output)

            # loss of this network's predictions
            # this_mse = tf.reduce_mean(
            #     tf.square(self.labels_ - this_output))
            # self.mses.append(this_mse)

            # this network's weights
            # this_theta = tf.get_collection(
            #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=str(i))

            # train step for this network
            # gv = [(g, v) for g, v in self.opt.compute_gradients(
            #     this_mse, this_theta) if g is not None]
            # self.train_steps.append(self.opt.apply_gradients(gv))

        # self.predicted_outputs = self.curr_nn_outputs


    def train(self,
              data_inputs_rand,
              data_outputs_rand,
              data_inputs_onPol,
              data_outputs_onPol,
              nEpoch,
              inputs_val=None,
              outputs_val=None,
              inputs_val_onPol=None,
              outputs_val_onPol=None):

        #init vars
        np.random.seed()
        start = time.time()
        training_loss_list = []
        val_loss_list_rand = []
        val_loss_list_onPol = []
        val_loss_list_xaxis = []
        rand_loss_list = []
        onPol_loss_list = []

        #combine rand+onPol into 1 dataset
        if data_inputs_onPol.shape[0]>0:
            data_inputs = np.concatenate((data_inputs_rand, data_inputs_onPol))
            data_outputs = np.concatenate((data_outputs_rand,
                                           data_outputs_onPol))
        else:
            data_inputs = data_inputs_rand.copy()
            data_outputs = data_outputs_rand.copy()

        #dims
        nData_rand = data_inputs_rand.shape[0]
        nData_onPol = data_inputs_onPol.shape[0]
        nData = nData_rand + nData_onPol

        #training loop
        for i in range(nEpoch):

            #reset tracking variables to 0
            sum_training_loss = 0
            num_training_batches = 0

            ##############################
            ####### training loss
            ##############################

            #randomly order indices (equivalent to shuffling)
            range_of_indices = np.arange(data_inputs.shape[0])
            all_indices = npr.choice(
                range_of_indices, size=(data_inputs.shape[0],), replace=False)

            for batch in range(int(math.floor(nData / self.batchsize))):

                #walk through the shuffled new data
                data_inputs_batch = data_inputs[
                    all_indices[batch * self.batchsize:(batch + 1) *
                                self.batchsize]]  #[bs x K x dim]
                data_outputs_batch = data_outputs[all_indices[
                    batch * self.batchsize:(batch + 1) * self.
                    batchsize]]  #[bs x dim]

                #one iteration of feedforward training
                this_dataX = np.tile(data_inputs_batch,
                                     (self.ensemble_size, 1, 1, 1))

                inputs_ = torch.from_numpy(this_dataX)
                labels_ = torch.from_numpy(data_outputs_batch)

                first, second = torch.split(inputs_, [(self.inputSize - self.acSize), self.acSize], 3)
                second = torch.clamp(second, -1, 1)
                self.inputs_clipped = torch.cat([first, second], dim=3)
                self.curr_nn_outputs = []
                self.mses = []
                self.train_steps = []

                self.define_forward_pass()

                for i in range(self.ensemble_size):

                    self.opt = torch.optim.Adam(self.networks[i].parameters(), lr=self.params.lr)
                    self.opt.zero_grad()
                    inputState = torch.flatten(self.inputs_clipped[i], start_dim=1)
                    curr_output = self.networks[i].forward(inputState)
                    self.curr_nn_outputs.append(curr_output)

                    this_mse = torch.mean(
                        torch.pow(labels_ - curr_output, 2))
                    self.mses.append(this_mse)

                    this_mse.backward()
                    self.opt.step()


                    # this network's weights
                    # this_theta = tf.get_collection(
                    #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=str(i))

                    # train step for this network
                    # gv = [(g, v) for g, v in self.opt.compute_gradients(
                    #     this_mse, this_theta) if g is not None]
                    # self.train_steps.append(self.opt.apply_gradients(gv))

                self.predicted_outputs = self.curr_nn_outputs
                losses = self.mses
                outputs = self.curr_nn_outputs
                true_output = labels_

                loss = torch.mean(torch.stack(losses))

                training_loss_list.append(loss)
                sum_training_loss += loss
                num_training_batches += 1

            mean_training_loss = sum_training_loss / num_training_batches

            if ((i % 10 == 0) or (i == (nEpoch - 1))):

                if inputs_val is None:
                    pass

                else:
                    ##############################
                    ####### validation loss on rand
                    ##############################

                    #loss on validation set
                    val_loss_rand = self.get_loss(inputs_val, outputs_val)
                    val_loss_list_rand.append(val_loss_rand)
                    val_loss_list_xaxis.append(len(training_loss_list))

                    ##############################
                    ####### validation loss on onPol
                    ##############################

                    #loss on on-pol validation set
                    val_loss_onPol = self.get_loss(inputs_val_onPol,
                                                   outputs_val_onPol)
                    val_loss_list_onPol.append(val_loss_onPol)

                    ##############################
                    ####### training loss on rand
                    ##############################

                    loss_rand = self.get_loss(
                        data_inputs_rand,
                        data_outputs_rand,
                        fraction_of_data=0.5,
                        shuffle_data=True)
                    rand_loss_list.append(loss_rand)

                    ##############################
                    ####### training loss on onPol
                    ##############################

                    if (nData_onPol > 0):
                        loss_onPol = self.get_loss(
                            data_inputs_onPol,
                            data_outputs_onPol,
                            fraction_of_data=0.5,
                            shuffle_data=True)
                        onPol_loss_list.append(loss_onPol)

            if not self.print_minimal:
                if ((i % 10) == 0 or (i == (nEpoch - 1))):
                    print("\n=== Epoch {} ===".format(i))
                    print("    train loss: ", mean_training_loss)
                    print("    val rand: ", val_loss_rand)
                    print("    val onPol: ", val_loss_onPol)

        if not self.print_minimal:
            print("Training duration: {:0.2f} s".format(time.time() - start))

        lists_to_save = dict(
            training_loss_list = training_loss_list,
            val_loss_list_rand = val_loss_list_rand,
            val_loss_list_onPol = val_loss_list_onPol,
            val_loss_list_xaxis = val_loss_list_xaxis,
            rand_loss_list = rand_loss_list,
            onPol_loss_list = onPol_loss_list,)

        #done
        return mean_training_loss, lists_to_save


    def get_loss(self,
                 inputs,
                 outputs,
                 fraction_of_data=1.0,
                 shuffle_data=False):

        """ get prediction error of the model on the inputs """

        #init vars
        nData = inputs.shape[0]
        avg_loss = 0
        iters_in_batch = 0

        if shuffle_data:
            range_of_indices = np.arange(inputs.shape[0])
            indices = npr.choice(
                range_of_indices, size=(inputs.shape[0],), replace=False)

        for batch in range(int(math.floor(nData / self.batchsize) * fraction_of_data)):

            # Batch the training data
            if shuffle_data:
                dataX_batch = inputs[indices[batch * self.batchsize:
                                             (batch + 1) * self.batchsize]]
                dataZ_batch = outputs[indices[batch * self.batchsize:
                                              (batch + 1) * self.batchsize]]
            else:
                dataX_batch = inputs[batch * self.batchsize:(batch + 1) *
                                     self.batchsize]
                dataZ_batch = outputs[batch * self.batchsize:(batch + 1) *
                                      self.batchsize]

            #one iteration of feedforward training
            this_dataX = np.tile(dataX_batch, (self.ensemble_size, 1, 1, 1))

            inputs_ = torch.from_numpy(this_dataX)
            labels_ = torch.from_numpy(dataZ_batch)
            first, second = torch.split(inputs_, [(self.inputSize - self.acSize), self.acSize], 3)
            second = torch.clamp(second, -1, 1)
            inputs_clipped = torch.cat([first, second], dim=3)
            curr_nn_outputs = []
            mses = []

            for i in range(self.ensemble_size):
                # self.opt = torch.optim.Adam(self.networks[i].parameters(), lr=self.params.lr)
                # self.opt.zero_grad()
                inputState = torch.flatten(inputs_clipped[i], start_dim=1)
                curr_output = self.networks[i].forward(inputState)
                curr_nn_outputs.append(curr_output)

                this_mse = torch.mean(
                    torch.pow(labels_ - curr_output, 2))
                mses.append(this_mse)

                z_predictions_multiple = curr_nn_outputs
                losses = mses

            # z_predictions_multiple, losses = self.sess.run(
            #     [self.curr_nn_outputs, self.mses],
            #     feed_dict={
            #         self.inputs_: this_dataX,
            #         self.labels_: dataZ_batch
            #     })
            loss = np.mean(losses)

            avg_loss += loss
            iters_in_batch += 1

        if iters_in_batch==0:
            return 0
        else:
            return (avg_loss / iters_in_batch)


    #############################################################
    ### perform multistep prediction
    ### of N different candidate action sequences
    ### as predicted by the ensemble of learned models
    #############################################################

    #forward-simulate multiple different action sequences at once
    def do_forward_sim(self, states_true, actions_toPerform):

        #init vars
        state_list = []
        N = actions_toPerform.shape[0]
        horizon = actions_toPerform.shape[1]  # actions_toPerform: [N, horizon, K, aDim]

        # states_true [K,N,sDim] --> curr_states_NK [N, K, sDim]
        if (not (len(states_true) == 2 and states_true[1] == 0)):
            if len(states_true.shape) > 2:
                curr_states_NK = np.swapaxes(states_true, 0, 1)

        # states_true [K, sDim] --> [1, K, sDim] --> curr_states_NK [N, K, sDim]
        else:
            # mppi/etc. sets the 2nd entry to just junk... like [state, 0]
            # telling you to copy the first one N times (one for each simultaneous sim)
            curr_states_NK = np.tile(
                np.expand_dims(states_true[0], 0), (N, 1, 1))

        #curr_states_NK: [ens, N, K, sDim]
        curr_states_NK = np.tile(curr_states_NK, (self.ensemble_size, 1, 1, 1))

        #advance all N sims, one timestep at a time
        for timestep in range(horizon):

            #curr_states_pastTimestep: [ens, N, sDim]
            curr_states_pastTimestep = curr_states_NK[:, :,-1, :]

            # actions_toPerform: [N, horizon, K, aDim]
            curr_actions_NK = actions_toPerform[:, timestep, :, :]
            # curr_actions_NK: [ens, N, K, aDim]
            curr_actions_NK = np.tile(curr_actions_NK,(self.ensemble_size, 1, 1, 1))

            #keep track of states for all N sims
            state_list.append(np.copy(curr_states_pastTimestep))

            #make [N x (state,action)] array to pass into NN
            states_preprocessed = np.nan_to_num(
                np.divide((curr_states_NK - self.normalization_data.mean_x),
                          self.normalization_data.std_x))
            actions_preprocessed = np.nan_to_num(
                np.divide((curr_actions_NK - self.normalization_data.mean_y),
                          self.normalization_data.std_y))
            inputs_list = np.concatenate((states_preprocessed, actions_preprocessed), axis=3)

            #run the N sims all at once
            inputs_ = torch.from_numpy(inputs_list)
            first, second = torch.split(inputs_, [(self.inputSize - self.acSize), self.acSize], 3)
            second = torch.clamp(second, -1, 1)
            inputs_clipped = torch.cat([first, second], dim=3)
            curr_nn_outputs = []

            for i in range(self.ensemble_size):
                inputState = torch.flatten(inputs_clipped[i], start_dim=1)
                curr_output = self.networks[i].forward(inputState)
                curr_nn_outputs.append(curr_output)
            model_outputs = curr_nn_outputs

            # model_outputs = self.sess.run([self.predicted_outputs],
            #                             feed_dict={self.inputs_: inputs_list})
            model_output = np.array(model_outputs[0].detach())  #[ens, N,sDim]

            state_differences = np.multiply(
                model_output, self.normalization_data.std_z
            ) + self.normalization_data.mean_z

            #update the state info
            curr_states_pastTimestep = curr_states_pastTimestep + state_differences

            #remove current oldest element of K list (0th entry of 1st axis)
            curr_states_NK = np.delete(curr_states_NK, 0, 2)  #[ens,N,K,sDim] --> [ens,N,K-1,sDim]

            #add this new one to end of K list
            newentry = np.expand_dims(curr_states_pastTimestep, 2)  #[ens,N,sDim] --> [ens,N,1,sDim]
            curr_states_NK = np.append(curr_states_NK, newentry, 2)  #[ens,N,K-1,sDim]+[ens,N,1,sDim] = [ens,N,K,sDim]

        #return a list of length = horizon+1... each one has N entries, where each entry is (sDim,)
        state_list.append(np.copy(curr_states_pastTimestep))
        return state_list

    #############################################################
    ### perform multistep prediction
    ### of 1 candidate action sequence
    ### as predicted by the first learned model of the ensemble
    #############################################################

    def do_forward_sim_singleModel(self, states_true, actions_toPerform):

        state_list = []
        curr_state_K = np.copy(states_true[0])  #curr_state_K: [K, s_dim]
        curr_state = curr_state_K[-1]

        for curr_control_K in actions_toPerform:  #curr_control_K: [K, a_dim]

            #save current state
            state_list.append(np.copy(curr_state))  #curr_state: [s_dim, ]

            #preprocess and combine into [s,a]
            curr_state_K_preprocessed = (
                curr_state_K -
                self.normalization_data.mean_x) / self.normalization_data.std_x
            curr_control_K_preprocessed = (
                curr_control_K -
                self.normalization_data.mean_y) / self.normalization_data.std_y
            inputs_K_preprocessed = np.expand_dims(
                np.concatenate(
                    [curr_state_K_preprocessed, curr_control_K_preprocessed],
                    1), 0)

            #run through NN to get prediction
            this_dataX = np.tile(inputs_K_preprocessed, (self.ensemble_size, 1, 1, 1))
            #### TO DO... for now, just see 1st model's prediction
            inputs_ = torch.from_numpy(this_dataX)
            first, second = torch.split(inputs_, [(self.inputSize - self.acSize), self.acSize], 3)
            second = torch.clamp(second, -1, 1)
            inputs_clipped = torch.cat([first, second], dim=3)
            curr_nn_outputs = []

            for i in range(self.ensemble_size):
                inputState = torch.flatten(inputs_clipped[i], start_dim=1)
                curr_output = self.networks[i].forward(inputState)
                curr_nn_outputs.append(curr_output)
            model_outputs = curr_nn_outputs
            # model_outputs = self.sess.run([self.predicted_outputs], feed_dict={self.inputs_: this_dataX})
            model_output = np.array(model_outputs[0].detach())

            #multiply by std and add mean back in
            state_differences = (
                model_output[0][0] * self.normalization_data.std_z) + self.normalization_data.mean_z

            #update the state info
            curr_state = curr_state + state_differences

            #remove current oldest element of K list (0th entry of 0th axis)
            curr_state_K = np.delete(curr_state_K, 0, 0)
            #add this new one to end of K list
            curr_state_K = np.append(curr_state_K, np.expand_dims(curr_state, 0), 0)

        state_list.append(np.copy(curr_state))
        return state_list
