import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    train_loader = utils.load_train_cifar10()

    for epoch in range(20000):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for id, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            print 'Epoch: {4} | Loss: {0} | Acc: {1} ({2}/{3})'.format(
                train_loss / (id + 1),
                100. * correct / total,
                correct,
                total, epoch)
