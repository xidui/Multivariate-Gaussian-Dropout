from __future__ import print_function
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.autograd import Variable

import transformer
import dropout

conv_feature_1 = 20
conv_feature_2 = 50
conv_kernal_size = 5
fc1 = 500
fc2 = 200
fc3 = 10

# dataset = 'MNIST'
dataset = 'CIFAR10'

picture_size = 32
if dataset == 'MNIST':
    picture_size = 28
    conv_feature_1 = 6
    conv_feature_2 = 20
    fc1 = 50
    fc2 = 20


model = {
	0: {
        'name': 'Conv2d',
        'parameters': {
            'in_channels': 3 if dataset == 'CIFAR10' else 1,
            'out_channels': conv_feature_1,
            'kernel_size': conv_kernal_size
        },
        'activate': 'ReLU',
        'dropout': None
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
            'in_features': conv_feature_2 * (((picture_size - 4) // 2 - 4) // 2) ** 2,
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


def run(dataset='CIFAR10', apply_layer='conv', rate=0.1, type='gaussian'):
    _model = copy.deepcopy(model)
    _dropout = {
        'type': type,
        'rate': rate
    }
    if apply_layer == 'conv':
        _model[0]['dropout'] = _dropout
        _model[2]['dropout'] = _dropout
    elif apply_layer == 'fc':
        _model[4]['dropout'] = _dropout
        _model[5]['dropout'] = _dropout

    batch_size = 32
    net = Model(model_config=_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=0.0005)
    train_loader, test_loader = None, None
    if dataset == 'CIFAR10':
        train_loader = utils.load_train_cifar10(batch_size=batch_size)
        test_loader = utils.load_test_cifar10(batch_size=batch_size)
    if dataset == 'MNIST':
        train_loader = utils.load_train_mnist(batch_size=batch_size)
        test_loader = utils.load_test_mnist(batch_size=batch_size)

    name = 'layer:{0}|conv_1:{1}:{5}|conv_2:{2}:{6}|fc1:{3}:{7}|fc2:{4}:{8}|{9}.txt'.format(
        len(_model),
        conv_feature_1,
        conv_feature_2,
        fc1,
        fc2,
        str(_model[0]['dropout']),
        str(_model[2]['dropout']),
        str(_model[4]['dropout']),
        str(_model[5]['dropout']),
        dataset
    )

    for epoch in range(50):
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

            print('Epoch: {4} | Loss: {0} | Acc: {1} ({2}/{3}) | test'.format(
                test_loss / (id + 1),
                100. * test_correct / test_total,
                test_correct,
                test_total, epoch))

        print('Epoch: {0} | Train Loss: {1} | Test Loss: {2} | Train Acc: {3} | Test Acc: {4}'.format(
            epoch,
            train_loss / train_total,
            test_loss / test_total,
            100. * train_correct / train_total,
            100. * test_correct / test_total
        ), file=f)

        f.close()


if __name__ == '__main__':
    run(dataset='MNIST', apply_layer='conv', rate=0.1, type='multinomial')
    run(dataset='MNIST', apply_layer='conv', rate=0.3, type='multinomial')
    run(dataset='MNIST', apply_layer='conv', rate=0.3, type='multinomial2')
    run(dataset='MNIST', apply_layer='conv', rate=0.5, type='multinomial')
    # run(dataset='MNIST', apply_layer='fc', rate=0.1, type='multinomial')
    # run(dataset='MNIST', apply_layer='fc', rate=0.3, type='multinomial')
    # run(dataset='MNIST', apply_layer='fc', rate=0.3, type='multinomial2')
    # run(dataset='MNIST', apply_layer='fc', rate=0.5, type='multinomial')
    pass
