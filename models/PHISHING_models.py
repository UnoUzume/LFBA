import torch.nn as nn

from utils.utils import *


class GlobalModelForPHISHING(nn.Module):
	def __init__(self, args):
		super(GlobalModelForPHISHING, self).__init__()
		self.linear1 = nn.Linear(8, 4)
		self.classifier = nn.Linear(4, 2)
		self.args = args

	def forward(self, input_list):
		tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)

		# forward
		x = tensor_t
		x = self.linear1(x)
		x = self.classifier(x)
		return x


class LocalModelForPHISHING(nn.Module):
	def __init__(self, args, client_number):
		super(LocalModelForPHISHING, self).__init__()
		self.args = args
		if client_number == 0:
			self.backbone = nn.Sequential(nn.Linear(15, 8), nn.ReLU(), nn.Linear(8, 4))
		else:
			self.backbone = nn.Sequential(nn.Linear(15, 8), nn.ReLU(), nn.Linear(8, 4))

	def forward(self, x):
		x = self.backbone(x)
		return x
