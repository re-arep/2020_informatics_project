import torch
import torch.nn as nn
import torch.optim as optim
import load_data
import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Map3D(nn.Module):

    def __init__(self, size=16, distance=4, input_node_n=64, convey=0.9, rate=5):
        super(Map3D, self).__init__()
        self.size = size
        self.distance = distance
        self.padding = distance // 2
        self.input_node_n = input_node_n
        self.convey = convey
        self.input_node_list = []
        self.frame = nn.Parameter(torch.zeros(size, size, size, dtype=torch.double), requires_grad=True)
        self.neuron_matrix = nn.Parameter(torch.zeros(size+self.padding*2, size+self.padding*2, size+self.padding*2,
                                                      dtype=torch.double), requires_grad=True)
        self.neuron_matrix_dummy = \
            nn.Parameter(torch.zeros(size+self.padding*2, size+self.padding*2, size+self.padding*2, dtype=torch.double)
                         , requires_grad=True)
        self.synapse_matrix = nn.Parameter(torch.randn(size, size, size, distance, distance, distance, dtype=torch.double), requires_grad=True)
        self.rate = rate
        self.fc1 = nn.Linear(16*16*16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def set(self):
        size = self.size
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.synapse_matrix[i][j][k].add_(torch.randn(self.distance, self.distance, self.distance))
                    self.synapse_matrix = self.synapse_matrix.requires_grad_(True)
        print(self.synapse_matrix)

    def node_batch(self, node_n):
        frame = np.zeros((self.size+self.padding*2, self.size+self.padding*2, self.size+self.padding*2), dtype='i')
        m = self.padding
        n = self.padding + self.size
        node_list = []
        while True:
            i = rd.randrange(m, n)
            j = rd.randrange(m, n)
            k = rd.randrange(m, n)
            if frame[i][j][k] == 0:
                node_list.append([i, j, k])
                frame[i][j][k] = 1
            else:
                continue
            if len(node_list) == node_n:
                break
        return node_list

    def input_node(self):
        return Map3D.node_batch(self, self.input_node_n)

    def initial_set(self):
        Map3D.set(self)
        self.input_node_list = Map3D.input_node(self)

    def run(self, vinput):
        size = self.size
        padding = self.padding
        for i in range(self.input_node_n):
            self.neuron_matrix[self.input_node_list[i][0]][self.input_node_list[i][1]][self.input_node_list[i][2]]\
                += vinput[i]
        print(self.neuron_matrix)

        for rate in range(self.rate):
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        self.neuron_matrix_dummy += (self.synapse_matrix[i][j][k] *
                                                     self.neuron_matrix[i:i+padding][j:j+padding][k:k+padding]).sum()

            self.neuron_matrix = F.relu(self.neuron_matrix_dummy) * self.convey

        x = self.neuron_matrix.view(-1, 16*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Map3D()
net.initial_set()

trainloader, testloader, classes = load_data.dataset()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def running(epoch_n):
    for epoch in range(epoch_n):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            train_t = np.transpose(inputs[0], (2, 1, 0))
            print(inputs.shape)
            print(train_t.shape)
            optimizer.zero_grad()
            outputs = net.run(train_t)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    running(5)
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

