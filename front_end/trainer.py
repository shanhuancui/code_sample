""" Single-GPU trainer of an embedding network """

import torch
from time import perf_counter
import os


class Trainer(object):
    def __init__(self, train_dataloader=None, test_dataloader=None, model=None, optim='sgd', weight_decay=1e-4,
                 lr='cyc_step_4-0.02@0,0.1@15,0.02@30,0.1@45,0.01@60,0.002@80', epochs=100, device='cuda:0',
                 ckpt_dir='model_ckpt', ckpt_num=None, save_freq=5, logger=None):

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.device = f'cuda:{device}' if isinstance(device, int) else device
        self.ckpt_dir = ckpt_dir
        self.ckpt_num = ckpt_num
        self.save_freq = save_freq
        self.logger = logger

        self.optimizer = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = LRScheduler(self.lr)
        self.train_metrics = None
        self.test_metrics = None
        self.epochs_trained = 0

    def setup_train(self, device):
        self.setup_model(device)
        self.setup_optimizer()

        # Update model, optimizer, loss_fn, and epochs_trained
        if os.listdir(self.ckpt_dir):
            self.load_checkpoint(map_location=torch.device(device))

        self.train_metrics = Metrics(device)
        self.test_metrics = Metrics(device)

    def setup_model(self, device):
        self.model = self.model.to(device)

        for name, para in self.model.named_parameters():
            if 'conv' in name or 'emb' in name:
                para.requires_grad = False

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()

    def setup_optimizer(self):
        if self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        elif self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            raise NotImplementedError

    def save_checkpoint(self, epoch):
        print(self.model.state_dict()['emb_layers.emb0.bn.running_mean'][:5])
        ckpt_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.loss_fn}
        torch.save(ckpt_dict, f'{self.ckpt_dir}/ckpt-{(epoch + 1) // self.save_freq}')

        checkpoint = torch.load(f'{self.ckpt_dir}/ckpt-{(epoch + 1) // self.save_freq}', map_location=self.device)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], 'module.')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(self.model.state_dict()['emb_layers.emb0.bn.running_mean'][:5])

    def load_checkpoint(self, map_location=None):
        if self.ckpt_num is None:
            self.ckpt_num = max([int(ckpt.split('-')[-1]) for ckpt in os.listdir(self.ckpt_dir)])

        ckpt_path = f'{self.ckpt_dir}/ckpt-{self.ckpt_num}'
        assert os.path.exists(ckpt_path), f'checkpoint path {ckpt_path} does NOT exist.'

        checkpoint = torch.load(ckpt_path, map_location=map_location)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], 'module.')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_fn = checkpoint['loss']
        self.epochs_trained = checkpoint['epoch'] + 1

        assert self.epochs_trained == self.ckpt_num * self.save_freq, 'Incorrect trained epochs!'
        self.logger.info(f'Model restored from {ckpt_path}.\n')

    def compute_loss(self, prediction, label):
        pred_loss = self.loss_fn(prediction, label)
        # reg_loss = self.weight_decay * sum([(para ** 2).sum() for para in self.model.parameters()])

        reg_loss, reg_loss_out_layer = 0., 0.

        for name, para in self.model.named_parameters():
            if 'bn' not in name:
                if 'out_layer' in name:
                    reg_loss_out_layer += (para ** 2).sum()
                else:
                    reg_loss += (para ** 2).sum()
        reg_loss = self.weight_decay * reg_loss + 1. * self.weight_decay * reg_loss_out_layer

        return pred_loss + reg_loss, reg_loss

    def train_step(self, data, label):
        pred, pred_cos = self.model(data, label)
        loss, reg_loss = self.compute_loss(pred, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_metrics.update(loss, pred_cos, label, reg_loss)

    def train_epoch(self, device):
        self.model.train()

        for data, label in self.train_dataloader:
            data, label = data.to(device), label.to(device)
            self.train_step(data, label)

    def test_step(self, data, label):
        pred, pred_cos = self.model(data, label)
        loss = self.loss_fn(pred, label)

        self.test_metrics.update(loss, pred_cos, label)

    def test_epoch(self, device):
        self.model.eval()

        with torch.no_grad():
            for data, label in self.test_dataloader:
                data, label = data.to(device), label.to(device)
                self.test_step(data, label)

    def train(self):
        self.setup_train(device=self.device)
        self.logger.info(f'No. of total epochs : {self.epochs}, No. of trained epochs : {self.epochs_trained}\n')

        for epoch in range(self.epochs_trained, self.epochs):
            print(f'epoch: {epoch}')
            ts = perf_counter()

            # Set learning rate
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr_scheduler.get_lr(epoch)

            # Train and test
            self.train_epoch(device=self.device)
            self.test_epoch(device=self.device)

            # Monitor metrics
            train_loss, train_acc, aux_loss = self.train_metrics.result()
            test_loss, test_acc, _ = self.test_metrics.result()

            self.logger.info(f'Epoch: {epoch}, train_loss: {train_loss:.3f}, train_acc: {100 * train_acc:.2f}%, '
                             f'test_loss: {test_loss:.3f}, test_acc: {100 * test_acc:.2f}%, aux_loss: {aux_loss:.3f}')
            self.logger.info(f'lr at epoch {epoch}: {self.optimizer.param_groups[0]["lr"]:.6f}')
            self.logger.info(f'Elapsed time of training epoch {epoch}: {perf_counter() - ts:.2f} s.\n')

            assert not torch.tensor([train_loss]).isnan().item(), 'NaN occurs, training aborted!\n\n\n'

            self.train_metrics.reset()
            self.test_metrics.reset()

            # Save training checkpoints
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch)

        self.logger.info('[*****] Training finished.\n\n\n')


class LRScheduler(object):
    def __init__(self, lr_conf='step-0.1@0,0.01@40,0.001@70', init_linear=False):
        self.mode, lr_conf = lr_conf.split('-')
        self.lrs = [float(lr_.split('@')[0]) for lr_ in lr_conf.split(',')]
        self.milestones = [int(lr_.split('@')[1]) for lr_ in lr_conf.split(',')]
        self.init_linear = init_linear

        assert len(self.lrs) == len(self.milestones), 'Unequal length of learning rates and milestones!'
        assert self.milestones[0] == 0, 'The first epoch of lr scheduler must start from ZERO!'

    def get_lr(self, epoch):
        lr = self.lrs[0]

        for i in range(len(self.lrs) - 1):
            if self.milestones[i] <= epoch < self.milestones[i + 1]:
                if self.mode == 'cyclic':
                    lr = self.lrs[i] + (self.lrs[i + 1] - self.lrs[i]) / \
                         (self.milestones[i + 1] - self.milestones[i]) * (epoch - self.milestones[i])
                elif self.mode == 'step':
                    lr = self.lrs[i]
                elif self.mode.startswith('cyc_step'):
                    cyc_end = int(self.mode.split('_')[-1])
                    assert cyc_end % 2 == 0, 'cyc_end should be an even number!'

                    if i < cyc_end:
                        lr = self.lrs[i] + (self.lrs[i + 1] - self.lrs[i]) / \
                             (self.milestones[i + 1] - self.milestones[i]) * (epoch - self.milestones[i])
                    else:
                        lr = self.lrs[i]
                else:
                    raise NotImplementedError

                break

        if epoch >= self.milestones[-1]:
            lr = self.lrs[-1]

        return lr


class Metrics(object):
    def __init__(self, device):
        self.device = device
        self.loss, self.aux_loss = torch.tensor([0.], device=self.device), torch.tensor([0.], device=self.device)
        self.n_batches, self.n_samples, self.n_correct = 0, 0, 0

    def reset(self):
        self.loss, self.aux_loss = torch.tensor([0.], device=self.device), torch.tensor([0.], device=self.device)
        self.n_batches, self.n_samples, self.n_correct = 0, 0, 0

    def update(self, loss, preds, labels, aux_loss=None):
        self.loss += loss
        self.n_correct += torch.eq(preds.argmax(1), labels).sum()
        self.n_batches += 1
        self.n_samples += labels.shape[0]

        if aux_loss is not None:
            self.aux_loss += aux_loss

    def result(self):
        # Get loss, acc, aux_loss
        if self.n_batches == 0:
            return 0., 0., 0.

        return self.loss.item() / self.n_batches, self.n_correct / self.n_samples, self.aux_loss.item() / self.n_batches
