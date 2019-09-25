import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn import functional as F

from entropy import shannon_entropy




class Conv2d2(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.wBits = 8

    def forward(self, input):
        #for first input
        if self.wBits < 32:
            origWeights = self.weight.data.clone()
            dynMax = torch.max(origWeights)
            dynMin = torch.min(origWeights)
            self.weight.data, mult, add = part_quant(self.weight.data, max=dynMax, min=dynMin,bitwidth=self.wBits)
            self.weight.data = self.weight.data * mult + add

            output = super(Conv2d2, self).forward(input)
            self.weight.data = origWeights.data
        else:
            output = super(Conv2d2, self).forward(input)
        return output


class ReLUProj(nn.ReLU):
    def __init__(self, inplace=False, relu6 = False):
        super(ReLUProj, self).__init__(inplace)
    #    self.corr = 0
        self.inplace = inplace
        self.collectStats = True

        self.register_buffer('running_b', torch.zeros(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.zeros(1))
        self.c = torch.nn.Parameter(torch.zeros(1))

        self.quant = False
        self.c.requires_grad = True
        self.momentum = 0.9

        # TODO: HARDCODE
        self.actBitwidth = 6
        self.clip = True
        self.temp = -10
        self.mode = 'entropy'
        self.relu6 = relu6

    def forward(self, input):
        if input is tuple:
            prevEntr = input[1]
            input = input[0]
        else:
            prevEntr = []
        if not self.relu6:
            input = super(ReLUProj, self).forward(input)
        else:
            input = F.relu6(input, inplace=self.inplace)

        dictOptimization = torch.zeros(2).cuda()
        dictOptimization[0] = input.numel()


        back = False
        if len(input.shape) == 3:
            back = True
            input = input.unsqueeze(0)
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

           # u = get_projection_matrix(input - mn, 'pca', 1.0)

           # self.u.data =   torch.eye(C).cuda() # (torch.from_numpy(hadamard(C)).type(torch.float) / np.sqrt(C)).cuda() #  u.t()
           # self.v.data =  self.u.data.t() #u
            self.softEntrRange = np.arange(0, H * W, 16)

        elif self.actBitwidth < 30 and (self.quant or not self.training):
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
                        self.softEntrRange = np.arange(0 * H * W, 1 * H * W, 8)
                        bit_per_entry = soft_entropy(input[:, self.softEntrRange],
                                                     torch.max(input[:, self.softEntrRange]),
                                                     torch.min(input[:, self.softEntrRange]), self.actBitwidth,
                                                     temp=self.temp)
                elif self.mode == 'compression':
                    bit_per_entry = input.norm(p=1) / input.norm(p=2)
                else:
                    pass
                self.ent = bit_per_entry

            input, mult, add = part_quant(input, max=dynMax, min=dynMin,
                                          bitwidth=self.actBitwidth)

            if not self.training:
                bit_per_entry = shannon_entropy(input)
                self.ent = bit_per_entry
                self.num = input.numel()

            dictOptimization[1] = bit_per_entry
            input = input * mult + add


        else:
            pass

        input = input.t().contiguous().view(N, H, W, C)
        input = input.permute(0, 3, 1, 2)
        #
        return input


def quantize1d_kmeans(x, num_bits=8, n_jobs=-1):
    orig_shape = x.shape
    x = np.expand_dims(x.flatten(), -1)
    kmeans = KMeans(n_clusters=2 ** num_bits, random_state=0, n_jobs=n_jobs)
    x_kmeans = kmeans.fit_predict(x)
    q_kmeans = np.array([kmeans.cluster_centers_[i] for i in x_kmeans])
    return q_kmeans.reshape(orig_shape)


def get_projection_matrix(im, projType, eigenVar):
    if projType == 'eye':
        u, s = torch.eye(im.shape[0]).to(im), torch.ones(im.shape[0]).to(im)
    else:
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        #  print(cov)
        # svd
        u, s, _ = torch.svd(cov)
        #   print (s)
        if projType == 'pcaQ':
            u = torch.tensor(quantize1d_kmeans(u.cpu().detach().numpy(), num_bits=8)).cuda()
        elif projType == 'pcaT':
            # find index where eigenvalues are more important
            sRatio = torch.cumsum(s, 0) / torch.sum(s)
            cutIdx = (sRatio >= eigenVar).nonzero()[0]
            # throw unimportant eigenvector
            u = u[:, :cutIdx]
            s = s[:cutIdx]

    return u  # , s


def featuresReshape(input, N, C, H, W, microBlockSz, channelsDiv):
    # check input
    assert (microBlockSz < H)

    if (channelsDiv > C):
        channelsDiv = C
    assert (C % channelsDiv == 0)
    extra = 0
    if H % microBlockSz > 0:
        extra = H % microBlockSz
        hExtra = torch.zeros(N, C, extra, W).cuda()
        input = torch.cat((input, hExtra), dim=2)
        wExtra = torch.zeros(N, C, H + extra, extra).cuda()
        input = torch.cat((input, wExtra), dim=3)

    Ct = C // channelsDiv
    featureSize = microBlockSz * microBlockSz * Ct

    input = input.view(-1, Ct, H + extra, W + extra)  # N' x Ct x H x W
    input = input.permute(0, 2, 3, 1)  # N' x H x W x Ct
    input = input.contiguous().view(-1, microBlockSz, W + extra, Ct).permute(0, 2, 1, 3)  # N'' x W x microBlockSz x Ct
    input = input.contiguous().view(-1, microBlockSz, microBlockSz, Ct).permute(0, 3, 2,
                                                                                1)  # N''' x Ct x microBlockSz x microBlockSz

    return input.contiguous().view(-1, featureSize).t()


def featuresReshapeBack(input, N, C, H, W, microBlockSz, channelsDiv):
    assert (microBlockSz < H)

    if (channelsDiv > C):
        channelsDiv = C
    assert (C % channelsDiv == 0)

    extra = H % microBlockSz

    input = input.t()
    Ct = C // channelsDiv

    input = input.view(-1, Ct, microBlockSz, microBlockSz).permute(0, 3, 2,
                                                                   1)  # N'''  x microBlockSz x microBlockSz x Ct
    input = input.contiguous().view(-1, H + extra, microBlockSz, Ct).permute(0, 2, 1, 3)  # N''  x microBlockSz x H x Ct
    input = input.contiguous().view(-1, H + extra, W + extra, Ct).permute(0, 3, 1, 2)  # N' x Ct x H x W X
    input = input.contiguous().view(N, C, H + extra, W + extra)  # N x C x H x W
    if extra > 0:
        input = input[:, :, :-extra, :-extra]
    return input


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
        return 0 * torch.sum(x)
    bins = int(2 ** bitwidth)
    centers = torch.linspace(0, bins - 1, bins).cuda()
    act_scale = (2 ** bitwidth - 1) / (max - min)
    x = x.contiguous().view(-1)
    x = (x - min) * act_scale

    x = temp * (x.repeat(bins, 1).t() - centers) ** 2
    x = F.softmax(x, 1, _stacklevel=5)
    x = torch.sum(x, dim=0) / x.shape[0]
    x[x == 0] = 1  # hack
    x = -x * torch.log(x)
    return torch.sum(x) / np.log(2)
