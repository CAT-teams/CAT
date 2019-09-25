import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.entropy import shannon_entropy
from utils.huffman import huffman_encode




class Conv2d2(nn.Conv2d):
    def __init__(self,args,  in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.wBits = args.weightBitwidth
    def forward(self, input):
        #for first input
        output = []
        if not isinstance(input,list):
            input = [input,[]]
        if self.wBits < 32:
            origWeights = self.weight.data.clone()
            dynMax = torch.max(origWeights)
            dynMin = torch.min(origWeights)
            self.weight.data, mult, add = part_quant(self.weight.data, max=dynMax, min=dynMin,bitwidth=self.wBits)
            self.weight.data = self.weight.data * mult + add

            output.append(super(Conv2d2, self).forward(input[0]))
            self.weight.data = origWeights.data
        else:
            output.append(super(Conv2d2, self).forward(input[0]))


        output.append(input[1])
        return output

class BatchNorm2d2(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d2, self).__init__(num_features, eps, momentum, affine, track_running_stats)


    def forward(self, input):
        output = []
        output.append( super(BatchNorm2d2, self).forward(input[0]))
        output.append(input[1])
        return output






class ReLUProj(nn.ReLU):
    def __init__(self, args, inplace=False, relu6 = False):
        super(ReLUProj, self).__init__(inplace)
    #    self.corr = 0
        self.inplace = inplace
        self.actBitwidth = args.actBitwidth
        self.collectStats = True

        self.register_buffer('running_b', torch.zeros(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.zeros(1))

        self.quant = False
        self.c = torch.nn.Parameter(torch.zeros(1))
        self.c.requires_grad = True
        self.momentum = 0.9
        self.clip = args.clip
        self.temp = args.temp
        self.tempAnn = args.tempAnneal
        self.mode = args.method
        self.relu6 = relu6
        self.huffmanCode = args.huffmanCode

    def forward(self, input):

        prevEntr = input[1]
        input = input[0]
        if not self.relu6:
            input = super(ReLUProj, self).forward(input)
        else:
            input = F.relu6(input, inplace=self.inplace)

        dictOptimization = torch.zeros(2).to(input)
        dictOptimization[0] = input.numel()

        N, C, H, W = input.shape  # N x C x H x W

        input = input.permute(0, 2, 3, 1)  # N x H x W x C
        input = input.contiguous().view(-1, C).t()

        if self.collectStats:
            self.running_b.to(input.device).detach().mul_(self.momentum).add_(
                (input - input.mean()).abs().mean() * (1 - self.momentum))

            self.running_mean.to(input.device).detach().mul_(self.momentum).add_(
                input.mean() * (1 - self.momentum))

            self.running_std.to(input.device).detach().mul_(self.momentum).add_(
                input.std() * (1 - self.momentum))
            self.softEntrRange = np.arange(0, H * W, 16)

        elif self.actBitwidth < 30 and (self.quant or not self.training):
            self.temp *=  self.tempAnn
            if self.clip:
                input = F.relu(input) - F.relu(input - torch.abs(self.c))
            dynMax = torch.max(input)
            dynMin = torch.min(input)

            if self.training:
                if self.mode == 'entropy':
                    bit_per_entry = soft_entropy(input[:, self.softEntrRange], torch.max(input[:, self.softEntrRange]),
                                                 torch.min(input[:, self.softEntrRange]), self.actBitwidth,
                                                 temp=self.temp)
                    if torch.isnan(bit_per_entry):
                        self.softEntrRange = np.arange(4 * H * W, 5 * H * W, 8)
                        bit_per_entry = soft_entropy(input[:, self.softEntrRange],
                                                     torch.max(input[:, self.softEntrRange]),
                                                     torch.min(input[:, self.softEntrRange]), self.actBitwidth,
                                                     temp=self.temp)
                elif self.mode == 'compression':
                    bit_per_entry = input.norm(p=1) / input.norm(p=2)
                else:
                    pass

            input, mult, add = part_quant(input, max=dynMax, min=dynMin,
                                          bitwidth=self.actBitwidth)

            if not self.training:
                if self.huffmanCode:
                    bit_per_entry = huffman_encode(input) /input.numel()
                else:
                    bit_per_entry = shannon_entropy(input)

            dictOptimization[1] = bit_per_entry
            input = input * mult + add


        else:
            pass

        input = input.t().contiguous().view(N, H, W, C)
        input = input.permute(0, 3, 1, 2)
        return [input, prevEntr + [dictOptimization]]


def part_quant(x, max, min, bitwidth):
    if max != min:
        act_scale = (2 ** bitwidth - 1) / (max - min)
        q_x = Round.apply((x - min) * act_scale)
        return q_x, 1 / act_scale, min
    else:
        q_x = x
        return q_x, 1, 0


class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = x.round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None


#
#
def soft_entropy(x, max, min, bitwidth, temp):
    if max == min:
        return 0
    bins = int(2 ** bitwidth)
    centers = torch.linspace(0, bins - 1, bins).cuda()
    act_scale = (2 ** bitwidth - 1) / (max - min)
    x = x.contiguous().view(-1)
    x = (x - min) * act_scale

    x = (x.repeat(bins, 1).t() - centers) ** 2
    x = temp * x
    x = F.softmax(x, 1, _stacklevel=5)
    x = torch.sum(x, dim=0) / x.shape[0]
    x[x == 0] = 1  # hack
    x = -x * torch.log(x)
    return torch.sum(x) / np.log(2)
