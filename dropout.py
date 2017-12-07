import torch
from torch.autograd import Variable
import numpy as np


def bernoulli(input, rate):
    mask = Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(1 - rate)), requires_grad=True)
    return input * mask


def _multinomial_pre(input, rate):
    input_reshape = input.view(input.size()[0], -1)
    prob_multi = torch.sqrt(torch.mean(input_reshape ** 2, 0, True))
    norconst = torch.sum(prob_multi)
    prob_multi = prob_multi / norconst
    probs_multi = np.tile(prob_multi.data.numpy(), (input_reshape.size()[0], 1))
    # epsilon
    mask = np.zeros(input_reshape.size()[1])
    # draw samples according to multinomial distribution
    out = torch.multinomial(prob_multi, input_reshape.size()[1], replacement=True)
    out = out.data.numpy()
    for j in range(len(out[0])):
        mask[out[0][j]] += 1.0
    mask = np.tile(mask, (input_reshape.size()[0], 1))
    mask = Variable(torch.Tensor(mask / (input_reshape.size()[1] * (1 - rate)) / probs_multi), requires_grad=True)
    mask = mask.view(input.size())
    return mask


def multinomial(input, rate):
    mask = _multinomial_pre(input, rate)
    return input * mask


def multinomial2(input, rate):
    mask = _multinomial_pre(input, rate)
    mask = mask * Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(rate)), requires_grad=True)
    mask[mask==0] = 1.0
    return input * mask


def _gaussian_pre(input, rate):
    input_reshape = input.view(input.size()[0], -1)
    stdev = torch.std(input_reshape, 0, True).data.numpy()
    # each variable is kept with probability of stored in probs
    probs = np.zeros(input_reshape.size()[1])
    for j in range(len(probs)):
        probs[j] = np.random.normal(loc=1 - rate, scale=stdev[0][j], size=1)
        if probs[j] > 1.0:
            probs[j] = 1.0
        if probs[j] < 0.0:
            probs[j] = 0.0
    probs = np.tile(probs, (input_reshape.size()[0], 1))
    mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
    mask = mask.view(input.size())
    return mask


def gaussian(input, rate):
    mask = _gaussian_pre(input, rate)
    return input * mask


def gaussian2(input, rate):
    mask = _gaussian_pre(input, rate)
    mask = mask * Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(rate)), requires_grad=True)
    mask[mask == 0] = 1.0
    return input * mask


def _multivariant_pre(input, rate):
    input_reshape = input.view(input.size()[0], -1)

    # calculate the data-dependent mean for the dropout probabilities
    prob_multi = torch.sqrt(torch.mean(input_reshape ** 2, 0, True))
    norconst = torch.sum(prob_multi)
    prob_multi = prob_multi / norconst
    probs_multi = np.tile(prob_multi.data.numpy(), (input_reshape.size()[0], 1))
    # epsilon
    means = np.zeros(input_reshape.size()[1])
    # draw samples according to multinomial distribution
    out = torch.multinomial(prob_multi, input_reshape.size()[1], replacement=True)
    out = out.data.numpy()
    for j in range(len(out[0])):
        means[out[0][j]] += 1.0
    means = np.tile(means, (input_reshape.size()[0], 1))
    means = means / (input_reshape.size()[1] * (1 - rate)) / probs_multi
   
    # calculate the data-dependent variance for the dropout probabilities
    stdev = torch.std(input_reshape, 0, True).data.numpy()

    # each variable is kept with probability of stored in probs
    probs = np.zeros(input_reshape.size()[1])
    for j in range(len(probs)):
        probs[j] = np.random.normal(loc=means[0][j], scale=stdev[0][j], size=1)
        if probs[j] > 1.0:
            probs[j] = 1.0
        if probs[j] < 0.0:
            probs[j] = 0.0
    probs = np.tile(probs, (input_reshape.size()[0], 1))
    mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
    mask = mask.view(input.size())
    return mask


# combine the approach of multinomial dropout and gaussian dropout
def multivariant(input, rate):
    mask = _multivariant_pre(input, rate)
    return input * mask


def multivariant2(input, rate):
    mask = _multivariant_pre(input, rate)
    mask = mask * Variable(torch.bernoulli(torch.Tensor(input.size()).fill_(rate)), requires_grad=True)
    mask[mask == 0] = 1.0
    return input * mask