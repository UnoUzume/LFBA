import argparse
import copy
import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from attack.attack import attack_lra, attack_rsa
from dataset.dataset import CIFAR10_VFL, NUSWIDE_VFL, PHISHING_VFL, UCIHAR_VFL
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10
from models.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from models.PHISHING_models import GlobalModelForPHISHING, LocalModelForPHISHING
from models.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from utils.trainer import Trainer
from utils.utils import (
	raise_attack_exception,
	raise_dataset_exception,
	set_seed,
)


def main(args):
	device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

	# create logger
	logger = logging.getLogger(__name__)
	logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
	logger.setLevel(level=logging.DEBUG)

	# create handler for writing logs
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)
	fh = logging.FileHandler(args.results_dir + '/experiment.log')

	# add handler
	logger.addHandler(fh)

	# params
	logger.info(args)

	# create dataset
	logger.info('=> Preparing Data...')
	if args.dataset == 'CIFAR10':
		# data transform
		transform_train = transforms.Compose(
			[
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			]
		)

		transform_test = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			]
		)
		# return data, label, index
		train_data = CIFAR10_VFL(
			root=args.data_dir, train=True, download=True, transform=transform_train
		)
		test_data = CIFAR10_VFL(
			root=args.data_dir, train=False, download=True, transform=transform_test
		)
		test_data_asr = copy.deepcopy(test_data)
	elif args.dataset == 'UCIHAR':
		# return data, label, index
		train_data = UCIHAR_VFL(root=args.data_dir, train=True, transforms=None)
		test_data = UCIHAR_VFL(root=args.data_dir, train=False, transforms=None)
		test_data_asr = copy.deepcopy(test_data)
	elif args.dataset == 'PHISHING':
		# return data, label, index
		train_data = PHISHING_VFL(root=args.data_dir, train=True, transforms=None)
		test_data = PHISHING_VFL(root=args.data_dir, train=False, transforms=None)
		test_data_asr = copy.deepcopy(test_data)
	elif args.dataset == 'NUSWIDE':
		# return data, label, index
		selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']
		train_data = NUSWIDE_VFL(
			root=args.data_dir, selected_labels=selected_labels, train=True, transforms=None
		)
		test_data = NUSWIDE_VFL(
			root=args.data_dir, selected_labels=selected_labels, train=False, transforms=None
		)
		test_data_asr = copy.deepcopy(test_data)
	else:
		raise_dataset_exception()

	# build vfl models
	if args.dataset == 'CIFAR10':
		model_list = []
		extractor_list = []
		model_list.append(GlobalModelForCifar10(args))
		extractor_list.append(GlobalModelForCifar10(args))
		for i in range(args.client_num):
			model_list.append(LocalModelForCifar10(args))
			extractor_list.append(LocalModelForCifar10(args))
		optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
		criterion = nn.CrossEntropyLoss().to(device)
	elif args.dataset == 'UCIHAR':
		model_list = []
		extractor_list = []
		model_list.append(GlobalModelForUCIHAR(args))
		extractor_list.append(GlobalModelForUCIHAR(args))
		for i in range(args.client_num):
			model_list.append(LocalModelForUCIHAR(args, i))
			extractor_list.append(LocalModelForUCIHAR(args, i))
		optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
		criterion = nn.CrossEntropyLoss().to(device)
	elif args.dataset == 'PHISHING':
		model_list = []
		extractor_list = []
		model_list.append(GlobalModelForPHISHING(args))
		extractor_list.append(GlobalModelForPHISHING(args))
		for i in range(args.client_num):
			model_list.append(LocalModelForPHISHING(args, i))
			extractor_list.append(LocalModelForPHISHING(args, i))
		optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
		criterion = nn.CrossEntropyLoss().to(device)
	elif args.dataset == 'NUSWIDE':
		model_list = []
		extractor_list = []
		model_list.append(GlobalModelForNUSWIDE(args))
		extractor_list.append(GlobalModelForNUSWIDE(args))
		for i in range(args.client_num):
			model_list.append(LocalModelForNUSWIDE(args, i))
			extractor_list.append(LocalModelForNUSWIDE(args, i))
		optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
		criterion = nn.CrossEntropyLoss().to(device)
	else:
		raise_dataset_exception()
	model_list = [model.to(device) for model in model_list]
	extractor_list = [model.to(device) for model in extractor_list]
	# test
	if args.test_checkpoint:
		if os.path.isfile(args.test_checkpoint):
			logger.info(f"=> loading test checkpoint '{args.test_checkpoint}'")
			checkpoint_test = torch.load(args.test_checkpoint, map_location=device)
			for i in range(len(model_list)):
				model_list[i].load_state_dict(checkpoint_test['state_dict'][i])
				optimizer_list[i].load_state_dict(checkpoint_test['optimizer'][i])
			logger.info(
				"=> loaded test checkpoint '{}' (epoch {}, best accuracy: {:.4f})".format(
					args.test_checkpoint, checkpoint_test['epoch'], checkpoint_test['best_acc']
				)
			)
		else:
			logger.info(f"=> no test checkpoint found at '{args.test_checkpoint}'")
	# train from checkpoints
	checkpoint = None
	if args.pretrained_checkpoint:
		if os.path.isfile(args.pretrained_checkpoint):
			logger.info(f"=> loading checkpoint '{args.pretrained_checkpoint}'")
			checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
			args.start_epoch = checkpoint['epoch']
			for i in range(len(model_list)):
				model_list[i].load_state_dict(checkpoint['state_dict'][i])
				optimizer_list[i].load_state_dict(checkpoint['optimizer'][i])
			logger.info(
				"=> loaded checkpoint '{}' (epoch {}, best accuracy: {:.4f})".format(
					args.pretrained_checkpoint, checkpoint['epoch'], checkpoint['best_acc']
				)
			)
		else:
			logger.info(f"=> no checkpoint found at '{args.pretrained_checkpoint}'")
	# load feature extractor
	if args.feature_extractor:
		if os.path.isfile(args.feature_extractor):
			logger.info(f"=> loading checkpoint '{args.feature_extractor}'")
			extractor_checkpoint = torch.load(args.feature_extractor, map_location=device)
			for i in range(len(extractor_list)):
				extractor_list[i].load_state_dict(extractor_checkpoint['state_dict'][i])
			logger.info(
				"=> loaded checkpoint '{}' (epoch {}, best accuracy: {:.4f})".format(
					args.feature_extractor, extractor_checkpoint['epoch'], extractor_checkpoint['best_acc']
				)
			)
		else:
			logger.info(f"=> no checkpoint found at '{args.feature_extractor}'")

	if args.dataset == 'CIFAR10':
		trigger_dimensions = []
	elif args.dataset == 'UCIHAR':
		ranges = range(math.ceil(train_data.data.shape[1] / args.client_num), train_data.data.shape[1])
		trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
	elif args.dataset == 'PHISHING':
		ranges = range(15, 30)
		trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
	elif args.dataset == 'NUSWIDE':
		ranges = range(634, 1634)
		trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
	else:
		raise_dataset_exception()

	if args.attack is None:
		test_data_asr.data = attack_rsa(args, logger, test_data_asr.data, trigger_dimensions, 1, 'test')
	elif args.attack == 'rsa':
		train_data.data = attack_rsa(
			args, logger, train_data.data, trigger_dimensions, args.poison_rate, 'train'
		)
		test_data_asr.data = attack_rsa(args, logger, test_data_asr.data, trigger_dimensions, 1, 'test')
	elif args.attack == 'lra':
		train_data.data, train_data.targets = attack_lra(
			args,
			logger,
			train_data.data,
			trigger_dimensions,
			train_data.targets,
			args.poison_rate,
			'train',
		)
		test_data_asr.data, _ = attack_lra(
			args, logger, test_data_asr.data, trigger_dimensions, test_data_asr.targets, 1, 'test'
		)
	elif args.attack == 'LFBA':
		test_data_asr.data = attack_rsa(args, logger, test_data_asr.data, trigger_dimensions, 1, 'test')
	else:
		raise_attack_exception()

	train_loader = torch.utils.data.DataLoader(
		dataset=train_data, batch_size=args.batch_size, shuffle=True
	)
	test_loader = torch.utils.data.DataLoader(
		dataset=test_data, batch_size=args.batch_size, shuffle=True
	)
	test_asr_loader = torch.utils.data.DataLoader(
		dataset=test_data_asr, batch_size=args.batch_size, shuffle=True
	)
	trainer = Trainer(
		device,
		model_list,
		extractor_list,
		extractor_list[args.attack_client_num + 1],
		optimizer_list,
		criterion,
		train_loader,
		test_loader,
		test_asr_loader,
		trigger_dimensions,
		logger,
		args,
		checkpoint,
	)
	if args.test_checkpoint:
		if args.do_tsne:
			train_data_add_trigger = copy.deepcopy(train_data)
			train_data_add_trigger.data = attack_rsa(
				args, logger, train_data_add_trigger.data, trigger_dimensions, 1, 'test'
			)
			tsne_loader = torch.utils.data.DataLoader(
				dataset=train_data_add_trigger, batch_size=train_data_add_trigger.data.shape[0]
			)
			trainer.perform_tsne(tsne_loader)
			return
		trainer.test(0)
		return
	trainer.train()


if __name__ == '__main__':
	currentDateAndTime = datetime.now().strftime('%Y%m%d_%H%M%S')
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', default='./data/', help='data directory')
	parser.add_argument('--dataset', default='NUSWIDE', help='name of dataset')
	parser.add_argument('--device', default=0, type=int, help='GPU number')
	parser.add_argument(
		'--results_dir',
		default='./data/logs/' + str(currentDateAndTime),
		help='the result directory',
	)
	parser.add_argument('--seed', default=100, type=int, help='the seed')
	parser.add_argument('--epoch', default=100, type=int, help='number of training epoch')
	parser.add_argument('--batch_size', default=256, type=int, help='the batch size')
	parser.add_argument('--client_num', default=2, type=int, help='the number of clients')
	parser.add_argument('--pretrained_checkpoint', default=None, help='the checkpoint file')
	parser.add_argument('--lr', default=0.001, type=float, help='the learning rate')
	parser.add_argument(
		'--start_epoch', default=0, type=int, help='the epoch number of starting training'
	)
	parser.add_argument(
		'--print_steps', default=10, type=int, help='the print step of training logging'
	)
	parser.add_argument(
		'--attack', default=None, help='attack method'
	)  # None: baseline, lba: label-based attack, nla: no-label attack
	parser.add_argument('--target_label', default=3, type=int, help='the target label for backdoor')
	parser.add_argument('--poison_rate', default=0.1, type=float, help='the rate of poison samples')
	parser.add_argument(
		'--poison_dimensions', default=5, type=int, help='the dimensions to be poisoned'
	)
	parser.add_argument(
		'--trigger_feature_clip', default=1, type=float, help='the clip ratio of feature trigger'
	)
	parser.add_argument('--attack_client_num', default=1, type=int, help='the adversary client')
	parser.add_argument('--num_clusters', default=10, type=int, help='the number of clusters')
	parser.add_argument('--feature_extractor', default='', help='the feature extractor path')
	parser.add_argument('--select_rate', default=1, type=float)
	parser.add_argument('--random_select', action='store_true')
	parser.add_argument('--select_replace', action='store_true')
	parser.add_argument('--poison_all', action='store_true')
	parser.add_argument('--anchor_idx', default=33930, type=int)
	parser.add_argument('--test_checkpoint', default=None)

	args = parser.parse_args()

	args.results_dir = './data/logs/' + args.dataset + '/' + str(currentDateAndTime)

	# set seed
	set_seed(args.seed)

	main(args)
