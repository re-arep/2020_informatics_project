import numpy as np
import random as rd
import torch


class FrontalLobe:

    def __init__(self, size, distance, input_node_n, output_node_n, lrate, convey):
        self.size = size
        self.distance = distance
        self.padding = distance // 2
        self.input_node_n = input_node_n
        self.output_node_n = output_node_n
        self.lrate = lrate
        self.convey = convey
        self.input_node_list = []
        self.output_node_list = []
        self.frame = torch.zeros(size, size, size, dtype=torch.double)
        self.neuron_matrix = torch.zeros(size+self.padding, size+self.padding, size+self.padding, dtype=torch.double)
        self.neuron_matrix_dummy = torch.zeros(size+self.padding, size+self.padding, size+self.padding, dtype=torch.double)
        self.synapse_matrix = torch.zeros(size, size, size, dtype=torch.double)

    def node_batch(self, node_n):
        frame = torch.zeros(self.size, self.size, self.size, dtype=torch.double)
        m = self.padding
        n = self.padding + self.size
        node_list = []
        while True:
            i = rd.randrange(m, n)
            j = rd.randrange(m, n)
            k = rd.randrange(m, n)
            if frame[i][j][k] != 0:
                node_list.append([i, j, k])
                frame[i][j][k] = 1
            else:
                continue
            if len(node_list) == node_n:
                break
        return node_list

    def set(self):
        size = self.size
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.synapse_matrix[i][j][k] = torch.randn(self.distance, self.distance, self.distance)

    def input_d(self, vinput):
        self.input_node_list = node_batch(self.input_node_n)
        for i in range(self.input_node_n):
            self.neuron_matrix[self.input_node_list[i][0]][self.input_node_list[i][1]][self.input_node_list[i][2]]\
                += vinput[i]

    def output_d(self, voutput):
        self.output_node_list = node_batch(self.output_node_n)
        for i in range(self.output_node_n):
            voutput.append(self.neuron_matrix[self.output_node_list[i][0]][self.output_node_list[i][1]]
                           [self.output_node_list[i][2]])

    def soft_max(self, output):
        out_max = np.max(output)
        exp_out = np.exp(output - out_max)
        y = exp_out / np.sum(exp_out)
        return y

    def run(self):
        size = self.size
        padding = self.padding
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.neuron_matrix_dummy += (self.synapse_matrix[i][j][k] *
                                                 self.neuron_matrix[i:i+padding][j:j+padding][k:k+padding]).sum()

        self.neuron_matrix = np.where(self.neuron_matrix_dummy <= 0, 0, self.neuron_matrix_dummy * self.convey)
