import torch.optim as optim
import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Map3D(nn.Module):

    def __init__(self, size=16, distance=4, input_node_n=64, convey=0.9, rate=5):
        super(Map3D, self).__init__()
        self.size = size
        self.desize = size-4
        self.distance = distance
        self.padding = distance // 2
        self.input_node_n = input_node_n
        self.convey = convey
        self.input_node_list = []
        self.frame = torch.zeros(size-4, size, size, 4, dtype=torch.double)
        self.neuron_matrix = torch.zeros(size+self.padding*2-4, size+self.padding*2, size+self.padding*2, 4,
                                                      dtype=torch.double)
        self.neuron_matrix_dummy = \
            torch.zeros(size+self.padding*2-4, size+self.padding*2, size+self.padding*2, 4, dtype=torch.double)

        self.synapse_matrix = torch.randn(size-4, size, size, distance, distance, distance, dtype=torch.double)
        self.rate = rate
        self.fc1 = nn.Linear(12*16*16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def run(self, vinput):
        size = self.size
        padding = self.padding
        distance = self.distance
        vv = vinput.reshape(12, 16, 16, 4)
        self.neuron_matrix = F.pad(torch.add(self.frame, vv), pad=[0, 0, self.padding, self.padding, self.padding, self.padding, self.padding, self.padding,], value=0)

        for rate in range(self.rate):
            for i in range(size-4):
                for j in range(size):
                    for k in range(size):
                        self.neuron_matrix_dummy[i][j][k] = (self.synapse_matrix[i][j][k]
                                                              * self.neuron_matrix[i:i+distance, j:j+distance, k:k+distance]).sum()

            self.neuron_matrix = F.relu(self.neuron_matrix_dummy) * self.convey

        x = self.neuron_matrix[padding:size-4+padding, padding:size+padding, padding:size+padding].float()
        x = x.view(-1, 12*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Map3D()

trainloader, testloader, classes = load_data.dataset()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def running(epoch_n):
    for epoch in range(epoch_n):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            input_t = np.transpose(inputs, (2, 3, 1, 0))

            optimizer.zero_grad()
            outputs = net.run(input_t)

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

