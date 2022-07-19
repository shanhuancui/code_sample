""" Multi-GPU trainer of an embedding network """

import torch
import torch.distributed as dist
from time import perf_counter
from front_end.trainer import Trainer


class TrainerDDP(Trainer):
    def __init__(self, **kwargs):
        super(TrainerDDP, self).__init__(**kwargs)

    def setup_model(self, device):
        self.model = self.model.to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[torch.device(device)])

    def train(self):
        rank = dist.get_rank()

        self.setup_train(device=rank)
        self.logger.info(f'Rank {rank}, No. of epochs: {self.epochs}, No. of trained epochs : {self.epochs_trained}\n')

        for epoch in range(self.epochs_trained, self.epochs):
            if rank == 0:
                print(f'epoch: {epoch}')
            ts = perf_counter()

            # Set learning rate
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr_scheduler.get_lr(epoch)

            # Train and test
            self.train_epoch(device=rank)
            self.test_epoch(device=rank)

            # Monitor metrics
            train_loss, train_acc, aux_loss = self.train_metrics.result()
            test_loss, test_acc, _ = self.test_metrics.result()

            if rank == 0:
                self.logger.info(f'Epoch: {epoch}, train_loss: {train_loss:.3f}, train_acc: {100 * train_acc:.2f}%, '
                                 f'test_loss: {test_loss:.3f}, test_acc: {100 * test_acc:.2f}%, '
                                 f'aux_loss: {aux_loss:.3f}')
                self.logger.info(f'lr at epoch {epoch}: {self.optimizer.param_groups[0]["lr"]:.6f}')
                self.logger.info(f'Elapsed time of training epoch {epoch}: {perf_counter() - ts:.2f} s.\n')

            assert not torch.tensor([train_loss]).isnan().item(), f'Rank {rank}: NaN occurs, training aborted!\n\n\n'

            self.train_metrics.reset()
            self.test_metrics.reset()

            # Save training checkpoints
            if rank == 0 and (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch)

        dist.destroy_process_group()
        self.logger.info('[*****] Training finished.\n\n\n')
