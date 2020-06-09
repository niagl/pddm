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
