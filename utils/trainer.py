import copy
import os
from random import sample
from typing import cast

import numpy as np
import torch
import torch.linalg as la

from attack.attack import attack_LFBA, attack_LFBA_all, get_near_index
from dataset.utils import split_vfl


class Trainer:
	def __init__(
		self,
		device,
		model_list,
		extractor_list,
		extractor,
		optimizer_list,
		criterion,
		train_loader,
		test_loader,
		test_asr_loader,
		trigger_dimensions,
		logger,
		args=None,
		checkpoint=None,
	):
		self.device = device
		self.model_list = model_list
		self.extractor_list = extractor_list
		self.extractor = extractor
		self.optimizer_list = optimizer_list
		self.criterion = criterion
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.test_asr_loader = test_asr_loader
		self.logger = logger
		self.args = args
		self.checkpoint = checkpoint
		self.trigger_dimensions = trigger_dimensions

	def adjust_learning_rate(self, epoch):
		lr = self.args.lr * (0.1) ** (epoch // 20)
		for opt in self.optimizer_list:
			for param_group in opt.param_groups:
				param_group['lr'] = lr

	def train(self):
		if self.args.attack:
			self.logger.info(f'=> Start Training with {self.args.attack}...')
		else:
			self.logger.info('=> Start Training Baseline...')

		epoch_loss_list = []
		model_list = self.model_list
		model_list = [model.train() for model in model_list]

		best_epoch = 0
		best_acc = 0
		best_asr = 0
		best_trade_off = 0
		best_target = 0

		if self.checkpoint:
			best_acc = self.checkpoint['best_acc']

		# train and update
		for ep in range(self.args.start_epoch, self.args.epoch):
			batch_loss_list = []
			total = 0
			correct = 0
			if ep >= 1 and self.args.attack == 'LFBA':
				tsTrFeats = cast('torch.Tensor', self.grad_vec_epoch.cpu())  #: tsTrainFeatures
				tsTrLabels = self.target_epoch.cpu()
				naAllIdxs = cast('torch.Tensor', self.indexes_epoch.cpu())
				#: tsTrainShuffledIndexes 每个epoch都不同

				nPoisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
				nSelect = int(nPoisons * self.args.select_rate)

				kwargs = {
					'data': copy.deepcopy(self.train_loader.dataset.data_p),
					'target': self.train_loader.dataset.targets,
					'trigger_dimensions': self.trigger_dimensions,
					'rate': self.args.poison_rate,
					'mode': 'train',
				}

				# select sample set
				if ep == 1:
					iAncIdx = np.flatnonzero(naAllIdxs == self.args.anchor_idx).item()
					npCursors = get_near_index(tsTrFeats[iAncIdx], tsTrFeats, nPoisons)
					#: npCursors 索引列表的索引列表 idx of tsTrFeats
					self.naPoiIdxs = naAllIdxs[npCursors]  #: poison_indexes

					iAncLabel = tsTrLabels[iAncIdx].item()
					self.args.target_label = iAncLabel

					rConsistent: float = torch.eq(tsTrLabels[npCursors], iAncLabel).float().mean().item()
					self.logger.info(f'consistent_rate: {rConsistent}...')

				num_of_replace = int(len(self.naPoiIdxs) * self.args.select_rate)
				replace_all_list = list(
					set(naAllIdxs.numpy()).difference(set(torch.tensor(self.naPoiIdxs).numpy()))
				)
				naReplaceIdxs = sample(replace_all_list, num_of_replace)
				random_indexes_target = sample(list(self.naPoiIdxs), num_of_replace)

				if self.args.poison_all:
					if self.args.random_select:
						pIdxs = sample(list(self.naPoiIdxs), nSelect)
					else:
						pIdxs = self.naPoiIdxs

					self.train_loader.dataset.data = attack_LFBA_all(
						self.args,
						self.logger,
						pIdxs,
						**kwargs,
					)
				else:
					if self.args.random_select:
						replace_indexes_target = random_indexes_target
					else:
						npCursors = np.flatnonzero(np.isin(naAllIdxs, self.naPoiIdxs))
						#: npCursors 索引列表的索引列表
						tsCurFeats = tsTrFeats[npCursors]
						tsCurFeatsN2 = cast('torch.Tensor', la.vector_norm(tsCurFeats, 2, 1))
						_, tsSelIdxs = tsCurFeatsN2.topk(nSelect, 0)
						replace_indexes_target = naAllIdxs[npCursors[tsSelIdxs]]

					self.train_loader.dataset.data = attack_LFBA(
						self.args,
						self.logger,
						naReplaceIdxs,
						replace_indexes_target,
						self.naPoiIdxs,
						**kwargs,
					)
				self.logger.info(f'Target label:{self.args.target_label}')

			self.logger.info('=> Start Training for Injecting Backdoor...')

			grad_vec_epoch = []
			indexes_epoch = []
			target_epoch = []
			for step, (x0, _, y0, index) in enumerate(self.train_loader):
				x = x0.to(self.device).float()
				y = y0.to(self.device).long()
				# split data for vfl
				x_split_list = split_vfl(x, self.args)

				lbtmModels = model_list[1:]
				lBtmOut: list[torch.Tensor] = [
					model(x) for model, x in zip(lbtmModels, x_split_list, strict=True)
				]
				for t in lBtmOut:
					t.retain_grad()
				zTopOut = model_list[0](lBtmOut)

				# global model backward
				loss = self.criterion(zTopOut, y)

				for opt in self.optimizer_list:
					opt.zero_grad()

				loss.backward()

				if self.args.attack == 'LFBA':
					grad_vec_epoch.append(lBtmOut[-1].grad.to(self.device))
					indexes_epoch.append(index)
					target_epoch.append(y)

				for opt in self.optimizer_list:
					opt.step()

				batch_loss_list.append(loss.item())

				# calculate the training accuracy
				_, predicted = zTopOut.max(1)
				total += y.size(0)
				correct += predicted.eq(y).sum().item()

				# train_acc
				train_acc = correct / total
				loss = sum(batch_loss_list) / len(batch_loss_list)

				if step % self.args.print_steps == 0:
					self.logger.info(
						f'Epoch: {ep + 1}, {step + 1}/{len(self.train_loader)}: train loss: {loss:.4f}, '
						f'train main task accuracy: {train_acc:.4f}'
					)
			if self.args.attack == 'LFBA':
				self.grad_vec_epoch = torch.cat(grad_vec_epoch)
				self.indexes_epoch = torch.cat(indexes_epoch)
				self.target_epoch = torch.cat(target_epoch)

			epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
			epoch_loss_list.append(epoch_loss)
			self.adjust_learning_rate(ep + 1)
			test_acc, test_poison_accuracy, test_target, test_asr = self.test(ep)
			test_trade_off = (test_acc + test_asr) / 2
			if test_trade_off > best_trade_off:
				# best accuracy
				best_epoch = ep
				best_acc = test_acc
				best_asr = test_asr
				best_trade_off = test_trade_off

				poison_acc_for_best_epoch = test_poison_accuracy
				best_target = test_target

				# save model
				self.logger.info('=> Save best model...')
				state = {
					'epoch': ep + 1,
					'best_acc': best_acc,
					'asr': best_asr,
					'test_trade_off': test_trade_off,
					'test_target': best_target,
					'poison_acc': poison_acc_for_best_epoch,
					'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
					'optimizer': [
						self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))
					],
				}
				filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
				torch.save(state, filename)
			self.logger.info(
				f'=> End Epoch: {ep + 1}, best epoch: {best_epoch + 1}, '
				f'best trade off accuracy: {best_trade_off:.4f}, main task accuracy: {best_acc:.4f}, '
				f'test target accuracy: {best_target:.4f}, test asr: {best_asr:.4f}'
			)

	def test(self, ep):
		self.logger.info('=> Test ASR...')
		model_list = self.model_list
		model_list = [model.eval() for model in model_list]
		# test main task accuracy
		batch_loss_list = []
		total = 0
		correct = 0
		total_target = 0
		correct_target = 0
		for x0, _, y0, _ in self.test_loader:
			x = x0.to(self.device).float()
			y = y0.to(self.device).long()
			# split data for vfl
			x_split_list = split_vfl(x, self.args)

			lbtmModels = model_list[1:]
			lBtmOut: list[torch.Tensor] = [
				model(x) for model, x in zip(lbtmModels, x_split_list, strict=True)
			]
			for t in lBtmOut:
				t.retain_grad()
			zTopOut = model_list[0](lBtmOut)

			# global model backward
			loss = self.criterion(zTopOut, y)
			batch_loss_list.append(loss.item())

			# calculate the testing accuracy
			_, predicted = zTopOut.max(1)
			total += y.size(0)
			correct += predicted.eq(y).sum().item()

			total_target += (y == self.args.target_label).float().sum()
			correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

		# test poison accuracy and asr
		total_poison = 0
		correct_poison = 0
		total_asr = 0
		correct_asr = 0
		for step, (x, x_p, y, index) in enumerate(self.test_asr_loader):
			x = x.to(self.device).float()
			y = y.to(self.device).long()
			y_attack_target = torch.ones(size=y.shape).to(self.device).long()
			y_attack_target *= self.args.target_label
			# split data for vfl
			x_split_list = split_vfl(x, self.args)
			local_output_list = []
			global_input_list = []
			# get the local model outputs
			for i in range(self.args.client_num):
				local_output_list.append(model_list[i + 1](x_split_list[i]))
			# get the global model inputs, recording the gradients
			for i in range(self.args.client_num):
				global_input_t = local_output_list[i].detach().clone()
				global_input_t.requires_grad_(True)
				global_input_list.append(global_input_t)

			zTopOut = model_list[0](local_output_list)

			# calculate the poison accuracy
			_, predicted = zTopOut.max(1)
			total_poison += y.size(0)
			correct_poison += predicted.eq(y).sum().item()
			# calculate the asr
			total_asr += (y != self.args.target_label).float().sum()
			correct_asr += (
				(predicted[y != self.args.target_label] == self.args.target_label).float().sum()
			)

		# main task accuracy, poison_acc and asr
		test_acc = correct / total
		test_poison_accuracy = correct_poison / total_poison
		test_asr = correct_asr / total_asr
		test_target = correct_target / total_target
		epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
		test_trade_off = (test_acc + test_asr) / 2
		# main task accuracy on target set
		self.logger.info(
			f'=> Test Epoch: {ep + 1}, main task samples: {len(self.test_loader.dataset)}, attack samples: {len(self.test_asr_loader.dataset)}, test loss: {epoch_loss:.4f}, test trade off: {test_trade_off:.4f}, test main task '
			f'accuracy: {test_acc:.4f}, test target accuracy: {test_target:.4f}, test asr: {test_asr:.4f}'
		)

		return test_acc, test_poison_accuracy, test_target, test_asr
