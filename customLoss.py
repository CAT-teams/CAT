from torch.nn import CrossEntropyLoss, Module


class customLoss(Module):
    def __init__(self, args):
        super(customLoss, self).__init__()
        self.regul = args.regul
        self.crossEntropyLoss = CrossEntropyLoss().cuda()

    def forward(self, input, target, optParam):
        crossEntropyLoss = self.crossEntropyLoss(input, target)
        totalLoss = crossEntropyLoss + self.regul * optParam

        return totalLoss, crossEntropyLoss, optParam
