import torch.nn as nn
from utils.utils import *


class GlobalModelForNUSWIDE(nn.Module):
    def __init__(self, args):
        super(GlobalModelForNUSWIDE, self).__init__()
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 50)
        self.classifier = nn.Linear(50, 5)
        self.args = args

    def forward(self, input_list):
        tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForNUSWIDE(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForNUSWIDE, self).__init__()
        self.args = args
        backbone_I = nn.Sequential(
                    nn.Linear(634, 320),
                    nn.ReLU(),
                    nn.Linear(320, 160),
                    nn.ReLU(),
                    nn.Linear(160, 80),
                    nn.ReLU(),
                    nn.Linear(80, 40),
                    nn.ReLU()
                )
        backbone_T = self.backbone = nn.Sequential(
                    nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 250),
                    nn.ReLU(),
                    nn.Linear(250, 125),
                    nn.ReLU(),
                    nn.Linear(125, 60),
                    nn.ReLU()
                )
        if client_number == 0:
            self.backbone = backbone_I
        else:
            self.backbone = backbone_T

    def forward(self, x):
        x = self.backbone(x)
        return x
