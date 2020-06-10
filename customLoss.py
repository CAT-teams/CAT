from torch.nn import CrossEntropyLoss, Module


class customLoss(Module):
    def __init__(self, args):
        super(customLoss, self).__init__()
        self.regul = args.regul
        self.regul2 = args.regul2
        self.crossEntropyLoss = CrossEntropyLoss().cuda()

    def forward(self, input, target, optParam, optParam2):
        crossEntropyLoss = self.crossEntropyLoss(input, target)
        totalLoss = crossEntropyLoss + (self.regul * optParam) + (self.regul2 * optParam2)

        return totalLoss, crossEntropyLoss, optParam,optParam2
