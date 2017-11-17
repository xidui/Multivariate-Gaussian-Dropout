import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.autograd import Variable

import transformer
import dropout

conv_feature_1 = 30
conv_feature_2 = 80
conv_kernal_size = 5
fc1 = 800
fc2 = 300
fc3 = 10


model = {
	0: {
        'name': 'Conv2d',
        'parameters': {
            'in_channels': 3,
            'out_channels': conv_feature_1,
            'kernel_size': conv_kernal_size
        },
        'activate': 'ReLU',
        'dropout': {
            'type': 'multinomial',
            'rate': 0.3
        }
    },
    1: {
        'name': 'MaxPool2d',
        'parameters': {
            'kernel_size': 2
        },
        'dropout': None
    },
    2: {
        'name': 'Conv2d',
        'parameters': {
            'in_channels': conv_feature_1,
            'out_channels': conv_feature_2,
            'kernel_size': conv_kernal_size
        },
        'activate': 'ReLU',
        'dropout': None
    },
    3: {
        'name': 'MaxPool2d',
        'parameters': {
            'kernel_size': 2
        },
        'dropout': None
    },
    4: {
        'name': 'Linear',
        'parameters': {
            'in_features': conv_feature_2 * conv_kernal_size * conv_kernal_size,
            'out_features': fc1
        },
        'activate': 'ReLU',
        'transform': transformer.to_line,
        'dropout': None
    },
    5: {
        'name': 'Linear',
        'parameters': {
            'in_features': fc1,
            'out_features': fc2
        },
        'activate': 'ReLU',
        'dropout': None
    },
    6: {
        'name': 'Linear',
        'parameters': {
            'in_features': fc2,
            'out_features': fc3
        },
        'dropout': None
    },
    7: {
        'name': 'Softmax',
        'parameters': {},
    }
}


def layer_driver(layer, cfg):
    def driver(input):
        if cfg.get('transform'):
            input = cfg['transform'](input)
        if cfg.get('dropout'):
            input = getattr(dropout, cfg['dropout']['type'])(input, cfg['dropout']['rate'])
        input = layer(input)
        if cfg.get('activate'):
            input = getattr(nn, cfg['activate'])()(input)
        return input
    return driver


class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.model = []
        for index, cfg in sorted(model_config.items(), key=lambda x:x[0]):
            layer = getattr(nn, cfg['name'])(**cfg['parameters'])
            setattr(self, 'layer_{0}'.format(index), layer)
            self.model.append(layer_driver(layer, cfg))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


if __name__ == '__main__':
    batch_size = 32
    net = Model(model_config=model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=0.0005)
    train_loader = utils.load_train_cifar10(batch_size=batch_size)
    test_loader  = utils.load_test_cifar10(batch_size=batch_size)

    name = 'layer:{0}|conv_1:{1}|conv_2:{2}|fc1:{3}|fc2:{4}|.txt'.format(
        len(model),
        conv_feature_1,
        conv_feature_2,
        fc1,
        fc2
    )


    for epoch in range(100):
        f = open('result/{0}'.format(name), 'a')

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

            print('Epoch: {4} | Loss: {0} | Acc: {1} ({2}/{3})'.format(
                train_loss / (id + 1),
                100. * train_correct / train_total,
                train_correct,
                train_total, epoch))

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
