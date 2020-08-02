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
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class NetModel(torch.nn.Module):

    def __init__(self, inputState, intermediate_size, num_fc_layers, outputSize):
        super().__init__()
        network = collections.OrderedDict()



        network['layer_in'] = torch.nn.Linear(inputState, intermediate_size)
        # torch.nn.init.zeros_(fc_1.weight)
        # torch.nn.init.zeros_(fc_1.bias)
        # network.append(fc_1)
        network['relu_in'] = torch.nn.ReLU()

        for i in range(num_fc_layers-1):
            network['layer_'+str(i+1)] = torch.nn.Linear(intermediate_size, intermediate_size)
            # torch.nn.init.zeros_(fc_i.weight)
            # torch.nn.init.zeros_(fc_i.bias)
            # network.append(fc_i)
            network['relu_'+str(i+1)] = torch.nn.ReLU()

        network['layer_out'] = torch.nn.Linear(intermediate_size, outputSize)
        # torch.nn.init.zeros_(fc_z.weight)
        # torch.nn.init.zeros_(fc_z.bias)
        # network.append(fc_z)

        self.network = torch.nn.Sequential(network)
        self.network.apply(self.init_weights_bias)

    def forward(self, x):
        return self.network(x.float())

    def init_weights_bias(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)


def feedforward_network(inputSize, outputSize, num_fc_layers,
                        depth_fc_layers):

    intermediate_size = depth_fc_layers

    return NetModel(inputSize, intermediate_size, num_fc_layers, outputSize)

def target(x):
    return x*x

def perform_experiment(exp_type, batch_size, n_iter, tag):
    if exp_type==1:
        input = np.arange(1, 51, dtype=np.float32)
        target = [x ** 2 for x in input]
        # input1 = input.reshape((batch_size, 1))
        input = torch.from_numpy(input)
        # input = torch.flatten(input2, start_dim=1)
        target = np.asarray(target, dtype=np.float32)
        target = torch.from_numpy(target)
    else:
        input = torch.zeros((50, ))
        target = torch.ones((50, ))

    # if exp_type == 1:
    #     input = np.arange(1, 500 + 1, dtype=np.float32).reshape(-1,1)
    #     output = [x * x for x in input]
    #     target = np.asarray(output, dtype=np.float32).reshape(-1,1)
    # else:
    #     input = torch.zeros((500, 1))
    #     target = torch.ones((500, 1))

    dataset = Dataset(input, target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()
    # print('input: ', input)
    for i in range(n_iter):

        for input, target in dataloader:
            curr_output = net.forward(input)
            opt.zero_grad()
            # this_mse = torch.mean(
            #     torch.pow(target - curr_output, 2))
            this_mse = loss(curr_output, target)
            writer.add_scalar('loss '+tag, this_mse, i)
            writer.flush()
            # print('-------------------------------------')

            # print('loss: ', this_mse)
            # print('output:', curr_output)
            # print('target:', target)
            print('--------iter-----', i)
            print('---weight bias values ----')
            print_net_weights(net)
            # print(net.state_dict().keys())
            this_mse.backward()
            print('---gradient values ----')
            print_net_grads(net)
            # plot_grad_flow(net.named_parameters())
            opt.step()
            print('---updated weight bias values ----')
            print_net_weights(net)
        print('iter: ', i)

def print_net_weights(net):
    vals = dict()
    for n, p in net.named_parameters():
            vals[n] = (p.data.abs().mean(), p.data.abs().max())
            print(n, vals)

def print_net_grads(net):
    grads = dict()
    for n, p in net.named_parameters():
            grads[n] = (p.grad.abs().mean(), p.grad.abs().max())
            print(n, grads)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    # print('Max_grad: ', max_grads)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, x,y):
        self.input = x
        self.target = y

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]



if __name__ == "__main__":
    writer = SummaryWriter('test/run_dataLoader_11', flush_secs=2)
    net = feedforward_network(1, 1, 2, 500)
    batch_size = 5
    n_iter = 10
    perform_experiment(1, 1, n_iter, tag='square_func_batch_size_1_trial11')

    # perform_experiment(0, 1, n_iter, tag='zeros_ones_batch_size_1_trial1')
    # perform_experiment(0, 50, n_iter, tag='zeros_ones_batch_size_50')
    # #
    # perform_experiment(1, 1, n_iter, tag='square_func_batch_size_1')
    # perform_experiment(1, 50, n_iter, tag='square_func_batch_size_50')
