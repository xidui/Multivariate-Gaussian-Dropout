import torch
from torch.autograd import Variable
import numpy as np


def bernoulli(input, rate):
    mask = Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(1 - rate)), requires_grad=True)
    return input * mask


def multinomial(input, rate):
    input_reshape = input.view(input.size()[0], -1)
    prob_multi = (1 + np.random.uniform(low=-0.1, high=0.1, size=input_reshape.size()[1])) / input_reshape.size()[0]
    probs_multi = np.tile(prob_multi, (input_reshape.size()[0], 1))
    prob = torch.Tensor(prob_multi)
    # epsilon
    mask = np.zeros(input_reshape.size())
    # draw samples according to multinomial distribution
    for i in range(input_reshape.size()[0]):
        out = torch.multinomial(prob, input_reshape.size()[0], replacement=True)
        for j in range(out.size()[0]):
            mask[i, out[j]] += 1.0
    mask = Variable(torch.Tensor(mask/(input_reshape.size()[1]*(1-rate))/probs_multi), requires_grad=True)
    mask = mask.view(input.size())


    mask = mask * Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(rate)), requires_grad=True)
    mask[mask==0] = 1.0

    return input * mask


def gaussian(input, rate):
    mask = Variable(torch.normal(torch.Tensor(input.size()).fill_(1 - rate), std=0.1), requires_grad=True)

    mask = mask * Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(rate)), requires_grad=True)
    mask[mask == 0] = 1.0
    return input * mask

