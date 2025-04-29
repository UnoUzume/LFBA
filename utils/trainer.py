import copy
import time
from random import sample

from attack.attack import attack_LFBA, get_near_index
from dataset.utils import split_vfl
from utils.utils import *


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
		start_time_train = time.time()
		if self.args.attack:
			self.logger.info(f'=> Start Training with {self.args.attack}...')
			if self.args.pretrain_stage:
				self.logger.info('=> Pretrain...')
		else:
			self.logger.info('=> Start Training Baseline...')
		epoch_loss_list = []
		model_list = self.model_list
		model_list = [model.train() for model in model_list]
		best_acc = 0
		best_trade_off = 0
		best_epoch = 0
		asr_for_best_epoch = 0
		target_for_best_epoch = 0
		no_change = 0
		total_time_GPC = 0
		total_time_HS = 0
		self.select_his = torch.zeros(self.train_loader.dataset.data.shape[0])
		if self.checkpoint:
			best_acc = self.checkpoint['best_acc']
		# train and update
		for ep in range(self.args.start_epoch, self.args.epoch):
			batch_loss_list = []
			total = 0
			correct = 0
			if ep >= 1 and self.args.attack == 'LFBA':
				self.train_features, self.train_labels, self.train_indexes = (
					self.grad_vec_epoch,
					self.target_epoch,
					self.indexes_epoch,
				)
				self.train_features, self.train_labels, self.train_indexes = (
					self.train_features.cpu(),
					self.train_labels.cpu(),
					self.train_indexes.cpu(),
				)
				self.num_poisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
				self.num_select = int(self.num_poisons * self.args.select_rate)

				# select sample set
				if ep == 1:
					start_time = time.time()
					self.anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
					self.indexes = get_near_index(
						self.train_features[self.anchor_idx_t], self.train_features, self.num_poisons
					)
					end_time = time.time()
					print(f'The poison set construction time: {end_time - start_time}')
					total_time_GPC += end_time - start_time
					self.poison_indexes = self.train_indexes[self.indexes]
					self.consistent_rate = float(
						(self.train_labels[self.indexes] == int(self.train_labels[self.anchor_idx_t])).sum()
						/ len(self.indexes)
					)

				# For replace poisoning
				self.indexes = np.isin(
					self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy()
				)
				temp = np.array(range(len(self.train_indexes)))
				self.indexes = temp[self.indexes]
				self.l2_norm_features = torch.norm(self.train_features[self.indexes], p=2, dim=1)
				start_time = time.time()
				self.poison_features, self.select_indexes = self.l2_norm_features.topk(
					self.num_select, dim=0, largest=True, sorted=True
				)
				end_time = time.time()
				print(f'The hard-sample selection time: {end_time - start_time}')
				total_time_HS += end_time - start_time
				num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)
				replace_all_list = list(
					set(self.train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy()))
				)
				replace_indexes_others = sample(replace_all_list, num_of_replace)
				random_indexes_target = sample(list(self.poison_indexes), num_of_replace)
				selected_indexes_target = self.train_indexes[self.indexes[self.select_indexes]]

				if self.args.poison_all:
					if self.args.random_select:
						self.poison_indexes_t = sample(list(self.poison_indexes), self.num_select)
						self.indexes = np.isin(
							self.train_indexes.numpy(), torch.tensor(self.poison_indexes_t).numpy()
						)
					self.poisoning_labels = np.array(self.train_labels)[self.indexes]
					self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
					self.args.target_label = self.anchor_label
					self.logger.info(f'Target label:{self.anchor_label}')
					self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
					if self.args.random_select:
						self.train_loader.dataset.data = attack_LFBA(
							self.args,
							self.logger,
							[],
							[],
							self.train_indexes,
							self.poison_indexes_t,
							self.clean_data_p,
							self.train_loader.dataset.targets,
							self.trigger_dimensions,
							self.args.poison_rate,
							'train',
						)
					else:
						self.train_loader.dataset.data = attack_LFBA(
							self.args,
							self.logger,
							[],
							[],
							self.train_indexes,
							self.poison_indexes,
							self.clean_data_p,
							self.train_loader.dataset.targets,
							self.trigger_dimensions,
							self.args.poison_rate,
							'train',
						)
				else:
					if self.args.random_select:
						replace_indexes_target = random_indexes_target
					else:
						replace_indexes_target = selected_indexes_target
					self.poisoning_labels = np.array(self.train_labels)[self.indexes]
					self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
					self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
					self.train_loader.dataset.data = attack_LFBA(
						self.args,
						self.logger,
						replace_indexes_others,
						replace_indexes_target,
						self.train_indexes,
						self.poison_indexes,
						self.clean_data_p,
						self.train_loader.dataset.targets,
						self.trigger_dimensions,
						self.args.poison_rate,
						'train',
					)
					self.args.target_label = self.anchor_label
					self.logger.info(f'Target label:{self.anchor_label}')

			elif self.args.attack == 'rsa' or self.args.attack == 'lra' or self.args.attack is None:
				pass

			self.logger.info('=> Start Training for Injecting Backdoor...')

			self.grad_vec_epoch = []
			self.indexes_epoch = []
			self.target_epoch = []
			for step, (x_n, x_p, y, index) in enumerate(self.train_loader):
				x = x_n
				x = x.to(self.device).float()
				y = y.to(self.device).long()
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
					local_output_list[i].requires_grad_(True)
					local_output_list[i].retain_grad()
					x_split_list[i].requires_grad_(True)
					x_split_list[i].retain_grad()

				global_output = model_list[0](local_output_list)

				# global model backward
				loss = self.criterion(global_output, y)
				for opt in self.optimizer_list:
					opt.zero_grad()

				loss.backward()

				if self.args.attack == 'LFBA':
					self.grad_vec_epoch.append(
						local_output_list[self.args.attack_client_num].grad.to(self.device)
					)
					self.indexes_epoch.append(index)
					self.target_epoch.append(y)

				for opt in self.optimizer_list:
					opt.step()
				batch_loss_list.append(loss.item())

				# calculate the training accuracy
				_, predicted = global_output.max(1)
				total += y.size(0)
				correct += predicted.eq(y).sum().item()

				# train_acc
				train_acc = correct / total
				current_loss = sum(batch_loss_list) / len(batch_loss_list)

				if step % self.args.print_steps == 0:
					self.logger.info(
						f'Epoch: {ep + 1}, {step + 1}/{len(self.train_loader)}: train loss: {current_loss:.4f}, train main task accuracy: {train_acc:.4f}'
					)
			if self.args.attack == 'LFBA':
				self.grad_vec_epoch = torch.cat(self.grad_vec_epoch)
				self.indexes_epoch = torch.cat(self.indexes_epoch)
				self.target_epoch = torch.cat(self.target_epoch)

			epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
			epoch_loss_list.append(epoch_loss)
			self.adjust_learning_rate(ep + 1)
			test_acc, test_poison_accuracy, test_target, test_asr = self.test(ep)
			test_trade_off = (test_acc + test_asr) / 2
			if test_trade_off > best_trade_off:
				# best accuracy
				best_acc = test_acc
				best_trade_off = test_trade_off
				poison_acc_for_best_epoch = test_poison_accuracy
				asr_for_best_epoch = test_asr
				target_for_best_epoch = test_target
				no_change = 0
				best_epoch = ep
				# save model
				self.logger.info('=> Save best model...')
				state = {
					'epoch': ep + 1,
					'best_acc': best_acc,
					'test_trade_off': test_trade_off,
					'test_target': target_for_best_epoch,
					'poison_acc': poison_acc_for_best_epoch,
					'asr': asr_for_best_epoch,
					'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
					'optimizer': [
						self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))
					],
				}
				filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
				torch.save(state, filename)
			elif ep > self.args.pretrain_stage:
				no_change += 1
			self.logger.info(
				f'=> End Epoch: {ep + 1}, early stop epochs: {no_change}, best epoch: {best_epoch + 1}, best trade off accuracy: {best_trade_off:.4f}, main task accuracy: {best_acc:.4f}, test target accuracy: {target_for_best_epoch:.4f}, test asr: {asr_for_best_epoch:.4f}'
			)
			if no_change == self.args.early_stop:
				end_time_train = time.time()
				print(f'The total training time: {end_time_train - start_time_train}')
				print(
					f'The average training time of each epoch: {(end_time_train - start_time_train) / (ep + 1)}'
				)
				print(f'The poison set construction time: {total_time_GPC}')
				print(f'The average hard-sample selection time: {total_time_HS / (ep + 1)}')
				print(f'The total hard-sample selection time: {total_time_HS}')
				return

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
		for step, (x, x_p, y, index) in enumerate(self.test_loader):
			x = x.to(self.device).float()
			y = y.to(self.device).long()
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

			global_output = model_list[0](local_output_list)

			# global model backward
			loss = self.criterion(global_output, y)
			batch_loss_list.append(loss.item())

			# calculate the testing accuracy
			_, predicted = global_output.max(1)
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

			global_output = model_list[0](local_output_list)

			# calculate the poison accuracy
			_, predicted = global_output.max(1)
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
