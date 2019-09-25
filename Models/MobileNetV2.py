import math

import torch
import torch.nn as nn


from operations import ReLUProj as ReLU
from operations import Conv2d2 as Conv2d
from operations import BatchNorm2d2 as BatchNorm2d



def conv_bn(inp, oup, stride, args):
    return nn.Sequential(
        Conv2d(args, inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),

        ReLU(args, inplace=True, relu6 = True)
    )



def conv_1x1_bn(inp, oup, args):
    return nn.Sequential(
        Conv2d(args, inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        ReLU(args, inplace=True, relu6 = True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, args, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(args, hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                ReLU(args, inplace=True, relu6 = True),
                # pw-linear
                Conv2d(args, hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(args, inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                ReLU(args, inplace=True, relu6 = True),
                # dw
                Conv2d(args, hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                ReLU(args, inplace=True, relu6=True),
                # pw-linear
                Conv2d(args, hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            #return x + self.conv(x)
            y = self.conv(x)
            return [x[0] + y[0], y[1]]

        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, args):
        super(MobileNetV2, self).__init__()



        width_mult = 1
        n_class = 1000
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn( 3, input_channel, 2, args)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(args, input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(args, input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, args))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

        self.quantBlocks = self.buildQuantBlocks(args.noQuantEdges)
        self.depth = 35

    def buildQuantBlocks(self, noQuantEdges):
        modules_list = list(self.modules())
        quant_layers_list = [x for x in modules_list if
                             isinstance(x, ReLU)]

        if noQuantEdges:
            quant_layers_list[-1].actBitwidth = 32
        return quant_layers_list

    def forward(self, x):
        x = self.features(x)
        x[0] = x[0].mean(3).mean(2)
        x[0] = self.classifier(x[0])
        return x

    def loadPreTrained(self, pre_trained, device):
        self.load_state_dict(torch.load('./Models/mobilenet_v2.pth.tar'),False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
