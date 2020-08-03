import torch
import torch.nn as nn
import torch.optim as optim
import load_data
from models import Map3D

net = Map3D

trainloader, testloader, classes = load_data.dataset()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def running(epoch_n):
    for epoch in range(epoch_n):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net.run(inputs)
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

