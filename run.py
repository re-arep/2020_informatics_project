import torch
import load_data

trainloader, testloader, classes = load_data.dataset()

def running(epoch_n):
    for epoch in range(epoch_n):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
