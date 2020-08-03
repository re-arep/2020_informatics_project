import numpy as np
import random as rd
import torch


class Map3D:

    def __init__(self, size, distance, input_node_n, output_node_n, convey, vinput, rate):
        self.size = size
        self.distance = distance
        self.padding = distance // 2
        self.input_node_n = input_node_n
        self.output_node_n = output_node_n
        self.convey = convey
        self.input_node_list = []
        self.output_node_list = []
        self.frame = torch.zeros(size, size, size, dtype=torch.double)
        self.neuron_matrix = torch.zeros(size+self.padding, size+self.padding, size+self.padding, dtype=torch.double)
        self.neuron_matrix_dummy = torch.zeros(size+self.padding, size+self.padding, size+self.padding, dtype=torch.double)
        self.synapse_matrix = torch.zeros(size, size, size, dtype=torch.double)
        self.vinput = vinput
        self.rate = rate

    def set(self):
        size = self.size
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.synapse_matrix[i][j][k] = torch.randn(self.distance, self.distance, self.distance)

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

    def input_node(self):
        return Map3D.node_batch(self, self.input_node_n)

    def ouput_node(self):
        return Map3D.node_batch(self, self.output_node_n)

    def initial_set(self):
        Map3D.set()
        self.input_node_list = Map3D.input_node()
        self.output_node_list = Map3D.ouput_node()

    def input_data(self, vinput):
        self.input_node_list = Map3D.node_batch(self.input_node_n)
        for i in range(self.input_node_n):
            self.neuron_matrix[self.input_node_list[i][0]][self.input_node_list[i][1]][self.input_node_list[i][2]]\
                += vinput[i]

    def output_data(self):
        voutput = []
        self.output_node_list = Map3D.node_batch(self.output_node_n)
        for i in range(self.output_node_n):
            voutput.append(self.neuron_matrix[self.output_node_list[i][0]][self.output_node_list[i][1]]
                           [self.output_node_list[i][2]])

    def run(self, vinput):
        size = self.size
        padding = self.padding
        for i in range(self.input_node_n):
            self.neuron_matrix[self.input_node_list[i][0]][self.input_node_list[i][1]][self.input_node_list[i][2]]\
                += vinput[i]

        for rate in range(self.rate):
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        self.neuron_matrix_dummy += (self.synapse_matrix[i][j][k] *
                                                     self.neuron_matrix[i:i+padding][j:j+padding][k:k+padding]).sum()

            self.neuron_matrix = np.where(self.neuron_matrix_dummy <= 0, 0, self.neuron_matrix_dummy * self.convey)

        voutput = []
        for i in range(self.output_node_n):
            voutput.append(self.neuron_matrix[self.output_node_list[i][0]][self.output_node_list[i][1]]
                           [self.output_node_list[i][2]])

        return voutput

