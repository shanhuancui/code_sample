""" Train embedding networks """

from utils.my_utils import init_logger
from front_end.dataset import TrainDataset
from torch.utils.data import DataLoader
from front_end.model_tdnn import TDNN
from front_end.trainer import Trainer
from front_end.trainer_ddp import TrainerDDP
import argparse
from pathlib import Path
from datetime import datetime
import os
import torch
import torch.distributed as dist


# Set global parameters
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--feat_dim', type=int, default=40, help='dimension of acoustic features')
parser.add_argument('--min_len', type=int, default=200, help='minimum No. of frames of a training sample')
parser.add_argument('--max_len', type=int, default=400, help='maximum No. of frames of a training sample')
parser.add_argument('--batch_size', type=int, default=128, help='global mini-batch size')
parser.add_argument('--n_repeats', type=int, default=40,
                    help='No. of repeats of spk2utt during one epoch, 50 for voxceleb and 100 for sre')
parser.add_argument('--n_workers', type=int, default=0, help='No. of cpu processes used in the dataloader')
parser.add_argument('--is_utt_sampling', action='store_true', default=False, help='used in dataset creation')

parser.add_argument('--model', default='tdnn', help='tdnn, densenet, res2net, tdnn_mi or res2net_mi')
parser.add_argument('--filters', default='512-512-512-512-1500',
                    help='No. of channels of convolutional layers, 512-512-512-512-512-512-512-512-1500')
parser.add_argument('--kernel_sizes', default='5-3-3-1-1',
                    help='kernel size of convolutional layers, 5-1-3-1-3-1-3-1-1')
parser.add_argument('--dilations', default='1-2-3-1-1', help='dilation rate of convolutional layers, 1-1-2-1-3-1-4-1-1')
parser.add_argument('--pooling', default='stats', help='stats, attention-500-1')
parser.add_argument('--embedding_dims', default='192', help='embedding network config, 512-512')
parser.add_argument('--output_act', default='amsoftmax-0.25-30', help='softmax, amsoftmax-0.25-30, aamsoftmax-0.25-30')

parser.add_argument('--optim', default='sgd', help='adam or sgd')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
parser.add_argument('--lr', default='cyc_step_4-0.02@0,0.1@15,0.02@35,0.1@50,0.01@70,0.002@92',
                    help='step-0.1@0,0.01@50,0.001@75, cyc_step_4-0.02@0,0.1@15,0.02@30,0.1@45,0.01@60,0.002@80')
parser.add_argument('--epochs', type=int, default=100, help='No. of training epochs')
parser.add_argument('--device', default='cuda:1', help='cuda, cpu')
parser.add_argument('--ckpt_dir', nargs='?', help='directory of model checkpoint')
parser.add_argument('--ckpt_num', nargs='?', type=int, help='checkpoint number for resuming training, default: None')
parser.add_argument('--save_freq', type=int, default=5, help='frequency to save the model')
parser.add_argument('--n_gpus', type=int, default=2, help='No. of GPUs used in training')
args = parser.parse_args()

# Set ckpt time
cur_time = datetime.now().strftime("%Y%m%d_%H%M")
ckpt_dir = f'model_ckpt/ckpt_{cur_time}' if args.ckpt_dir is None else args.ckpt_dir
ckpt_time = '_'.join(ckpt_dir.split('/')[-1].split('_')[1:])
Path(f'{ckpt_dir}').mkdir(parents=True, exist_ok=True)

# Set log path
log_dir = 'log'
Path(f'{log_dir}').mkdir(parents=True, exist_ok=True)
log_path = f'{log_dir}/log_{ckpt_time}.log'
logger = init_logger(logger_name='train', log_path=log_path)

logger.info('----------------------------------------------------')
for arg, val in vars(args).items():
    if arg not in ['ckpt_dir']:
        logger.info(f'[*] {arg}: {val}')
logger.info(f'[*] ckpt_dir: {ckpt_dir}')
logger.info('----------------------------------------------------\n')


def train_func(rank, n_gpus):
    """ rank is only useful for DDP """
    # --------------------------------------------------------------------------------------------
    # DDP initialization
    # --------------------------------------------------------------------------------------------
    if n_gpus > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=n_gpus)

    # --------------------------------------------------------------------------------------------
    # Create training and test dataloaders
    # --------------------------------------------------------------------------------------------
    n_repeats = args.n_repeats if args.n_workers == 0 else args.n_repeats // args.n_workers
    train_dataset = TrainDataset(
        feat_dim=args.feat_dim, min_len=args.min_len, max_len=args.max_len,
        batch_size=args.batch_size, n_repeats=n_repeats, mode='train',
        is_utt_sampling=args.is_utt_sampling)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=None, num_workers=args.n_workers)

    test_dataset = TrainDataset(
        feat_dim=args.feat_dim, min_len=args.min_len, max_len=args.max_len,
        batch_size=args.batch_size, n_repeats=20, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=None, num_workers=args.n_workers)

    # --------------------------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------------------------
    if args.model == 'tdnn':
        model = TDNN(
            feat_dim=args.feat_dim, filters=args.filters, kernel_sizes=args.kernel_sizes,
            dilations=args.dilations, pooling=args.pooling, embedding_dims=args.embedding_dims,
            n_class=train_dataset.n_speaker, output_act=args.output_act)
    else:
        raise NotImplementedError

    if args.n_gpus == 1 or (n_gpus > 1 and rank == 0):
        logger.info('===============================================')
        logger.info(model)

        # logger.info('model.state_dict()')
        # for name in model.state_dict():
        #     logger.info(f'{name}: {model.state_dict()[name].size()}')

        total_paras = sum(para.numel() for para in model.parameters() if para.requires_grad)
        logger.info(f'Total No. of parameters: {total_paras / 1e6:.3f} M\n')
        logger.info('===============================================')

    # --------------------------------------------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------------------------------------------
    trainer_args = {
        'train_dataloader': train_dataloader, 'test_dataloader': test_dataloader, 'model': model,
        'optim': args.optim, 'weight_decay': args.weight_decay, 'lr': args.lr, 'epochs': args.epochs,
        'device': args.device, 'ckpt_dir': ckpt_dir, 'ckpt_num': args.ckpt_num,
        'save_freq': args.save_freq, 'logger': logger}

    if n_gpus > 1:
        trainer = TrainerDDP(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)

    return trainer.train()


if __name__ == '__main__':
    if args.n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.multiprocessing.spawn(train_func, args=(args.n_gpus, ), nprocs=args.n_gpus, join=True)
    else:
        train_func(args.device, args.n_gpus)

    logger.info('To the END.\n\n')
