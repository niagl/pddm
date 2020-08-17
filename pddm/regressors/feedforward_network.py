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
import collections


class NetModel(torch.nn.Module):

    def __init__(self, inputState, intermediate_size, num_fc_layers, outputSize):
        super().__init__()
        network = collections.OrderedDict()

        network['layer_in'] = torch.nn.Linear(inputState, intermediate_size)
        network['relu_in'] = torch.nn.ReLU()

        for i in range(num_fc_layers - 1):
            network['layer_' + str(i + 1)] = torch.nn.Linear(intermediate_size, intermediate_size)
            network['relu_' + str(i + 1)] = torch.nn.ReLU()

        network['layer_out'] = torch.nn.Linear(intermediate_size, outputSize)

        self.network = torch.nn.Sequential(network)
        self.network.apply(self.init_weights_bias)

    def forward(self, x):
        return self.network(x.float())

    def init_weights_bias(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_uniform_(m.bias)
            # torch.nn.init.zeros_(m.weight)
            # torch.nn.init.constant_(m.weight, 0.5)
            torch.nn.init.constant_(m.bias, 0)

def feedforward_network(inputSize, outputSize, num_fc_layers,
                        depth_fc_layers):

    intermediate_size = depth_fc_layers
    return NetModel(inputSize, intermediate_size, num_fc_layers, outputSize)

class myDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]