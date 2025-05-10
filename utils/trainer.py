import copy
import os
from random import sample

import numpy as np
import torch

from attack.attack import attack_LFBA, attack_LFBA_all, get_near_index
from dataset.utils import split_vfl


class Trainer:
	def __init__(
		self,
		device,
		model_list,
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

		best_metrics = {
			'epoch': 0,
			'acc': 0,
			'asr': 0,
			'trade_off': 0,
			'target': 0,
			'poison_acc': 0,
		}

		if self.checkpoint:
			best_metrics['acc'] = self.checkpoint['best_acc']

		# train and update
		for ep in range(self.args.start_epoch, self.args.epoch):
			self.logger.info('=> Start Training for Injecting Backdoor...')

			grad_vec_epoch0 = []
			indexes_epoch0 = []
			target_epoch0 = []
			batch_loss_list = []
			total = 0
			correct = 0
			for step, (x0, _, y0, index) in enumerate(self.train_loader):
				x = x0.to(self.device).float()
				y = y0.to(self.device).long()
				# split data for vfl
				x_split_list = split_vfl(x, self.args)

				lbtmModels = model_list[1:]
				lBtmOut: list[torch.Tensor] = [
					model(x) for model, x in zip(lbtmModels, x_split_list, strict=True)
				]
				[t.retain_grad() for t in lBtmOut]

				zTopOut = model_list[0](lBtmOut)

				# global model backward
				loss = self.criterion(zTopOut, y)

				for opt in self.optimizer_list:
					opt.zero_grad()

				loss.backward()

				if self.args.attack == 'LFBA':
					grad_vec_epoch0.append(lBtmOut[-1].grad.to(self.device))
					indexes_epoch0.append(index)
					target_epoch0.append(y)

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

			epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
			epoch_loss_list.append(epoch_loss)
			self.adjust_learning_rate(ep + 1)
			test_acc, test_poison_accuracy, test_target, test_asr = self.test(ep)
			test_trade_off = (test_acc + test_asr) / 2
			if test_trade_off > best_metrics['trade_off']:
				best_metrics.update(
					{
						'epoch': ep,
						'acc': test_acc,
						'asr': test_asr,
						'trade_off': test_trade_off,
						'target': test_target,
						'poison_acc': test_poison_accuracy,
					}
				)

				# save model
				self.logger.info('=> Save best model...')
				state = {
					**best_metrics,
					'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
					'optimizer': [
						self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))
					],
				}
				filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
				torch.save(state, filename)

			self.logger.info(
				f'=> End Epoch: {ep + 1}, best epoch: {best_metrics["epoch"] + 1}, '
				f'best trade off accuracy: {best_metrics["trade_off"]:.4f}, main task accuracy: {best_metrics["acc"]:.4f}, '
				f'test target accuracy: {best_metrics["target"]:.4f}, test asr: {best_metrics["asr"]:.4f}'
			)

			if self.args.attack == 'LFBA':
				tsTrFeats = torch.cat(grad_vec_epoch0).cpu()  #: tsTrainFeatures
				tsTrLabels = torch.cat(target_epoch0).cpu()
				naAllIdxs = torch.cat(indexes_epoch0).cpu().numpy()
				#: tsTrainShuffledIndexes 每个epoch都不同

				nPoisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
				nSelect = int(nPoisons * self.args.select_rate)

				if ep == 0:
					self.select_sample_set(tsTrFeats, tsTrLabels, naAllIdxs, nPoisons)

				kwargs = {
					'data': copy.deepcopy(self.train_loader.dataset.data_p),
					'target': self.train_loader.dataset.targets,
					'trigger_dimensions': self.trigger_dimensions,
					'rate': self.args.poison_rate,
					'mode': 'train',
				}

				nReplace = int(len(self.naPoiIdxs) * self.args.select_rate)
				naOtherIdxs = np.setdiff1d(naAllIdxs, self.naPoiIdxs, True)
				naOtherIdxs.sort()  #! 为了保持一致 list(set(A).difference(set(B))
				naReplaceIdxs = sample(list(naOtherIdxs), nReplace)
				random_indexes_target = sample(list(self.naPoiIdxs), nReplace)

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
						naTargetIdxs = random_indexes_target
					else:
						naPoiCursors = np.flatnonzero(np.isin(naAllIdxs, self.naPoiIdxs))
						#: npCursors 索引列表的索引列表
						tsNorm2: torch.Tensor = torch.norm(tsTrFeats[naPoiCursors], 2, 1)
						_, tsSelIdxs = tsNorm2.topk(nSelect)
						naTargetIdxs = naAllIdxs[naPoiCursors[tsSelIdxs]]

					self.train_loader.dataset.data = attack_LFBA(
						self.args,
						self.logger,
						naReplaceIdxs,
						naTargetIdxs,
						self.naPoiIdxs,
						**kwargs,
					)

	def select_sample_set(self, tsTrFeats, tsTrLabels, naAllIdxs, nPoisons):
		iAncCursor = np.flatnonzero(naAllIdxs == self.args.anchor_idx).item()
		tsCursors = get_near_index(tsTrFeats[iAncCursor], tsTrFeats, nPoisons)
		naPoiCursors = tsCursors.numpy()
		#: npCursors 索引列表的索引列表 idx of tsTrFeats
		self.naPoiIdxs = naAllIdxs[naPoiCursors]  #! poison_indexes

		iAncLabel = tsTrLabels[iAncCursor].item()
		self.args.target_label = iAncLabel
		self.logger.info(f'Target label:{self.args.target_label}')

		rConsistent: float = torch.eq(tsTrLabels[tsCursors], iAncLabel).float().mean().item()
		self.logger.info(f'consistent_rate: {rConsistent}...')

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
		test_target = correct_target / total_target

		test_poison_accuracy = correct_poison / total_poison
		test_asr = correct_asr / total_asr

		epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
		test_trade_off = (test_acc + test_asr) / 2
		# main task accuracy on target set
		self.logger.info(
			f'=> Test Epoch: {ep + 1}, main task samples: {len(self.test_loader.dataset)}, attack samples: {len(self.test_asr_loader.dataset)}, test loss: {epoch_loss:.4f}, test trade off: {test_trade_off:.4f}, test main task '
			f'accuracy: {test_acc:.4f}, test target accuracy: {test_target:.4f}, test asr: {test_asr:.4f}'
		)

		return test_acc, test_poison_accuracy, test_target, test_asr
