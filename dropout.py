import torch
from torch.autograd import Variable


def bernoulli(input, rate):
    mask = Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(1 - rate)), requires_grad=True)
    return input * mask


def gaussian(input):
    return input
