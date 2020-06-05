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

import torch


class NetModel(torch.nn.Module):

    def __init__(self, inputState, intermediate_size, num_fc_layers, outputSize):
        super().__init__()
        network = []
        fc_1 = torch.nn.Linear(inputState, intermediate_size)
        torch.nn.init.xavier_uniform_(fc_1.weight)
        network.append(fc_1)
        network.append(torch.nn.ReLU())

        for _ in range(num_fc_layers):
            fc_i = torch.nn.Linear(intermediate_size, intermediate_size)
            torch.nn.init.xavier_uniform_(fc_i.weight)
            network.append(fc_i)
            network.append(torch.nn.ReLU())

        fc_z = torch.nn.Linear(intermediate_size, outputSize)
        torch.nn.init.xavier_uniform_(fc_z.weight)
        network.append(fc_z)

        self.network = torch.nn.Sequential(*network)

    def forward(self, x):
        return self.network(x.float())

# TODO: convert input states from tenspr to int for inputSize
def feedforward_network(inputSize, outputSize, num_fc_layers,
                        depth_fc_layers):

    intermediate_size = depth_fc_layers

    return NetModel(inputSize, intermediate_size, num_fc_layers, outputSize)

    # with tf.variable_scope(str(scope)):
    #
    #     #concat K entries together [bs x K x sa] --> [bs x ksa]
    #     inputState = torch.flatten(inputStates, start_dim=1)
    #
    #     #vars
    #     intermediate_size = depth_fc_layers
    #     reuse = False
    #     initializer = tf.contrib.layers.xavier_initializer(
    #         uniform=False, seed=None, dtype=tf_datatype)
    #     fc = tf.contrib.layers.fully_connected
    #
    #     # make hidden layers
    #     for i in range(num_fc_layers):
    #         if i==0:
    #             fc_i = fc(
    #                 inputState,
    #                 num_outputs=intermediate_size,
    #                 activation_fn=None,
    #                 weights_initializer=initializer,
    #                 biases_initializer=initializer,
    #                 reuse=reuse,
    #                 trainable=True)
    #         else:
    #             fc_i = fc(
    #                 h_i,
    #                 num_outputs=intermediate_size,
    #                 activation_fn=None,
    #                 weights_initializer=initializer,
    #                 biases_initializer=initializer,
    #                 reuse=reuse,
    #                 trainable=True)
    #         # h_i = tf.nn.relu(fc_i)
    #         h_i = torch.nn.functional.relu(fc_i)
    #
    #     # make output layer
    #     z = fc(
    #         h_i,
    #         num_outputs=outputSize,
    #         activation_fn=None,
    #         weights_initializer=initializer,
    #         biases_initializer=initializer,
    #         reuse=reuse,
    #         trainable=True)
    #
    # return z