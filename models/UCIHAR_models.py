import torch.nn as nn
import math
from utils.utils import *


class GlobalModelForUCIHAR(nn.Module):
    def __init__(self, args):
        super(GlobalModelForUCIHAR, self).__init__()
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 6)
        self.args = args

    def forward(self, input_list):
        tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForUCIHAR(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForUCIHAR, self).__init__()
        self.args = args
        if client_number == 0:
            self.backbone = nn.Sequential(
                nn.Linear(math.ceil(561 / self.args.client_num), 140),
                nn.ReLU(),
                nn.Linear(140, 70),
                nn.ReLU(),
                nn.Linear(70, 35),
                nn.ReLU(),
                nn.Linear(35, 16),
                nn.ReLU()
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(round(561 / self.args.client_num), 140),
                nn.ReLU(),
                nn.Linear(140, 70),
                nn.ReLU(),
                nn.Linear(70, 35),
                nn.ReLU(),
                nn.Linear(35, 16),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.backbone(x)
        return x
