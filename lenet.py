import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self, conv_feature_1=6, conv_feature_2=16, fc1=500, fc2=200):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, conv_feature_1, 5)
        self.conv2 = nn.Conv2d(conv_feature_1, conv_feature_2, 5)
        self.fc1 = nn.Linear(conv_feature_2*5*5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 10)

    def forward(self, x):
        '''
        :param x: (batch_size * 3 * 32 * 32)
        '''
        out = F.relu(self.conv1(x))  # (batch_size * 6 * 28 * 28)
        out = F.max_pool2d(out, 2)  # (batch_size * 6 * 14 * 14)
        out = F.relu(self.conv2(out))  # (batch_size * 16 * 10 * 10)
        out = F.max_pool2d(out, 2)  # (batch_size * 16 * 5 * 5)
        out = out.view(out.size(0), -1)  # (batch_size * 400)
        out = F.relu(self.fc1(out))  # (batch_size * 120)
        out = F.relu(self.fc2(out))  # (batch_size * 84)
        out = self.fc3(out)  # (batch_size * 10)
        return out


if __name__ == '__main__':
    batch_size = 32
    net = LeNet(conv_feature_1=20, conv_feature_2=50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=0.0005)
    train_loader = utils.load_train_cifar10(batch_size=batch_size)
    test_loader  = utils.load_test_cifar10(batch_size=batch_size)

    f = open('output.txt', 'w')

    for epoch in range(500):
        net.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for id, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.data).sum()

            print 'Epoch: {4} | Loss: {0} | Acc: {1} ({2}/{3})'.format(
                train_loss / (id + 1),
                100. * train_correct / train_total,
                train_correct,
                train_total, epoch)

        net.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        for id, (inputs, labels) in enumerate(test_loader):
            inputs, labels = Variable(inputs, volatile=True), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels.data).sum()

            print 'Epoch: {4} | Loss: {0} | Acc: {1} ({2}/{3}) | test'.format(
                test_loss / (id + 1),
                100. * test_correct / test_total,
                test_correct,
                test_total, epoch)

        print >> f, 'Epoch: {0} | Train Loss: {1} | Test Loss: {2} | Train Acc: {3} | Test Acc: {4}'.format(
            epoch,
            train_loss / train_total,
            test_loss / test_total,
            100. * train_correct / train_total,
            100. * test_correct / test_total
        )

    f.close()