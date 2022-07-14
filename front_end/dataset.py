""" Dataset subclassing torch.utils.data.IterableDataset """

from abc import ABC
import numpy as np
from utils.my_utils import rd_utt2x, rd_spk2x, rd_single_parameter
import random
import pandas as pd
from torch.utils.data import IterableDataset, Dataset, get_worker_info
import math


class TrainDataset(IterableDataset, ABC):
    def __init__(self, feat_dim=40, min_len=200, max_len=400, batch_size=64, n_repeats=50, mode='train',
                 is_utt_sampling=False):
        super(TrainDataset).__init__()
        """
        :param feat_dim: int, dimension of the acoustic feature
        :param min_len: int, minimum No. of frames of a training sample (a sample refers to a chunk of acoustic frames)
        :param max_len: int, maximum No. of frames of a training sample
        :param n_repeats: int, No. of repeats of spk2utt during one epoch
        :param mode: str, 'train', 'test', or 'validation'
        :param is_utt_sampling: bool, if True, form mini-batches by directly sampling the utterances;
                otherwise sampling the speakers first and then sampling the utterances for each speaker.
                Note that in the former case, each mini-batch may have samples coming from the same speaker.
        """

        self.feat_dim = feat_dim
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_repeats = n_repeats
        self.mode = mode
        self.is_utt_sampling = is_utt_sampling

        assert self.n_repeats >= 1, 'n_repeats must be no less than 1!'
        self._load_feat_info()

    def _load_feat_info(self):
        spk2utt_info_file = f'train/spk2utt_info_{self.mode}'
        utt2frame_info_file = 'train/utt2frame_info'
        total_num_frames_file = 'train/total_num_frames'
        memmap_file = 'train/train_combined.mmap'

        print(f'Loading spk2utt_info from {spk2utt_info_file}...')
        self.spk2utt_info = rd_spk2x(spk2utt_info_file, x_keys=['n_utt', 'utt_offset', 'label', 'utt_int'])
        self.spk2utt_info['utt_int'] = self.spk2utt_info['utt_int'].str.split()

        print(f'Loading utt2frame_info from {utt2frame_info_file}...')
        self.utt2frame_info = rd_utt2x(utt2frame_info_file, x_keys=['n_frame', 'frame_offset'])

        print(f'Loading total_num_frames from {total_num_frames_file}...')
        self.total_n_frames = rd_single_parameter(total_num_frames_file)

        print(f'Loading training mmap from {memmap_file}...\n')
        self.feats_mmap = np.memmap(memmap_file, dtype='float32', mode='r', shape=(self.total_n_frames, self.feat_dim))

        self.n_speaker = self.spk2utt_info.shape[0]
        self.n_utterance = self.utt2frame_info.shape[0]

    def __iter__(self):
        if self.mode == 'train':
            return self.generate_batch(is_utt_sampling=self.is_utt_sampling)
        else:
            return self.generate_batch(is_utt_sampling=True)

    def generate_batch(self, is_utt_sampling=False):
        """ is_utt_sampling: whether to form mini-batches by sampling from the utterances directly or by
        speaker-utterance (two-stage) sampling. If True, each mini-batch may have samples coming from the
        same speaker; otherwise no samples in the mini-batch will come from the same speaker. """

        if is_utt_sampling:
            spk2utt_info_tmp = self.spk2utt_info.copy()
            spk2utt_info_tmp = spk2utt_info_tmp.reindex(spk2utt_info_tmp.index.repeat(self.n_repeats))

            while spk2utt_info_tmp.shape[0] >= self.batch_size:
                frame_info_per_batch, spk2utt_info_tmp = self.sample_batch_info(spk2utt_info_tmp)
                feat_mats_per_batch, feat_labels_per_batch = self.load_batch_data(frame_info_per_batch)
                yield feat_mats_per_batch, feat_labels_per_batch
        else:
            for _ in range(self.n_repeats):
                spk2utt_info_tmp = self.spk2utt_info.copy()

                while spk2utt_info_tmp.shape[0] >= self.batch_size:
                    frame_info_per_batch, spk2utt_info_tmp = self.sample_batch_info(spk2utt_info_tmp)
                    feat_mats_per_batch, feat_labels_per_batch = self.load_batch_data(frame_info_per_batch)
                    yield feat_mats_per_batch, feat_labels_per_batch

    def load_batch_data(self, frame_info_per_batch):
        """ Load one mini-batch of frame-level data and speaker labels according to frame_info_per_batch """

        frame_offsets_per_batch = frame_info_per_batch['frame_offset']
        frame_lens_per_batch = frame_info_per_batch['frame_len']
        feat_mat_per_batch = []

        for frame_offset, frame_len in zip(frame_offsets_per_batch, frame_lens_per_batch):
            feat_mat = self.feats_mmap[frame_offset: frame_offset + frame_len, :]
            feat_mat_per_batch.append(feat_mat)

        return np.stack(feat_mat_per_batch).transpose((0, 2, 1)), frame_info_per_batch['label'].values

    def sample_batch_info(self, spk2utt_info):
        """
        Sample frame info of utterances in a mini-batch
        :param spk2utt_info: Dataframe, meta info to be sampled from
        :return:
            batch_info: Dataframe, containing
                frame_offsets: frame offsets of samples
                sample_lens: length of samples
                labels: speaker labels of samples
            spk2utt_info_remained: Dataframe, remained spk2utt_info after sampling
        """

        sample_len = random.randint(self.min_len, self.max_len)  # frame length of sampled utterances
        local_idx = np.random.permutation(spk2utt_info.shape[0])
        spk2utt_info_se = spk2utt_info.iloc[local_idx[:self.batch_size]]

        utt_ints_per_spk = sample_utt_int_per_spk(spk2utt_info_se['utt_int'])
        utt_offsets_se = spk2utt_info_se['utt_offset'].values + utt_ints_per_spk
        utt2frame_info_se = self.utt2frame_info.iloc[utt_offsets_se]

        frame_offsets_se = sample_frame_offsets(utt2frame_info_se['n_frame'], sample_len)
        frame_offsets_se = utt2frame_info_se['frame_offset'].values + frame_offsets_se
        sample_lens_se = np.asarray([sample_len] * self.batch_size)
        labels_se = spk2utt_info_se['label'].values

        # spk2utt_info_remained = spk2utt_info.drop(local_idx[:self.batch_size])
        spk2utt_info_remained = spk2utt_info.iloc[local_idx[self.batch_size:]]

        return pd.DataFrame(np.vstack((frame_offsets_se, sample_lens_se, labels_se)).T,
                            columns=['frame_offset', 'frame_len', 'label']), spk2utt_info_remained


class EvalDataset(Dataset):
    def __init__(self, source='voxceleb1_test', feat_dim=40, start_idx=0, end_idx=500, selected_dur=None):
        super(EvalDataset).__init__()
        """
        :param start_idx: start offset of a partition of a dataset, a partition is like dataset[start_idx: end_idx]
        :param end_idx: end index of a partition of the whole dataset
        :param selected_dur: duration of randomly selected segments in No. of frames
                This is for duration mismatch experiments. The default None means using full-length utts.
        """
        self.source = source
        self.feat_dim = feat_dim
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.selected_dur = selected_dur

        self._load_feat_info()

    def _load_feat_info(self):
        utt2frame_info_file = f'eval/feats/{self.source}_utt2frame_info'
        self.utt2frame_info = rd_utt2x(utt2frame_info_file, x_keys=['n_frame', 'frame_offset'])
        self.n_utterance = self.utt2frame_info.shape[0]

    def __len__(self):
        return max(self.end_idx - self.start_idx, 0)

    def __getitem__(self, idx):
        total_num_frames_file = f'eval/feats/{self.source}_total_num_frames'
        memmap_file = f'eval/feats/{self.source}.mmap'
        total_n_frames = rd_single_parameter(total_num_frames_file)
        feats_mmap = np.memmap(memmap_file, dtype='float32', mode='r', shape=(total_n_frames, self.feat_dim))

        idx += self.start_idx
        n_frames, frame_offsets = self.utt2frame_info['n_frame'], self.utt2frame_info['frame_offset']
        selected_dur, dur_offset = n_frames[idx], frame_offsets[idx]  # use full length of each utterance

        if self.selected_dur is not None and 0 < self.selected_dur <= n_frames[idx]:
            selected_dur = self.selected_dur
            dur_offset = frame_offsets[idx] + sample_frame_offset(n_frames[idx], selected_dur)

        return np.stack(feats_mmap[dur_offset: dur_offset + selected_dur]).transpose((1, 0))


class EvalIterDataset(IterableDataset, ABC):
    def __init__(self, source='voxceleb1_test', feat_dim=40, start_idx=0, end_idx=500, selected_dur=None):
        super(EvalIterDataset).__init__()
        """
        :param start_idx: start offset of a partition of a dataset, a partition is like dataset[start_idx: end_idx]
        :param end_idx: end index of a partition of the whole dataset
        :param selected_dur: duration of randomly selected segments in No. of frames
                This is for duration mismatch experiments. The default None means using full-length utts.
        """
        self.source = source
        self.feat_dim = feat_dim
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.selected_dur = selected_dur

        self._load_feat_info()

    def _load_feat_info(self):
        utt2frame_info_file = f'eval/feats/{self.source}_utt2frame_info'
        self.utt2frame_info = rd_utt2x(utt2frame_info_file, x_keys=['n_frame', 'frame_offset'])
        self.n_utterance = self.utt2frame_info.shape[0]

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            start_idx, end_idx = self.start_idx, self.end_idx
        else:
            n_utts_per_worker = int(math.ceil((self.end_idx - self.start_idx) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = self.start_idx + worker_id * n_utts_per_worker
            end_idx = min(start_idx + n_utts_per_worker, self.end_idx)

        return self.generate_sample(start_idx, end_idx)

    def generate_sample(self, start_idx, end_idx):
        total_num_frames_file = f'eval/feats/{self.source}_total_num_frames'
        memmap_file = f'eval/feats/{self.source}.mmap'
        total_n_frames = rd_single_parameter(total_num_frames_file)
        feats_mmap = np.memmap(memmap_file, dtype='float32', mode='r', shape=(total_n_frames, self.feat_dim))

        frame_offsets = self.utt2frame_info['frame_offset']
        n_frames = self.utt2frame_info['n_frame']

        for idx in range(start_idx, end_idx):
            selected_dur, dur_offset = n_frames[idx], frame_offsets[idx]  # use full length of each utt

            if self.selected_dur is not None and 0 < self.selected_dur <= n_frames[idx]:
                selected_dur = self.selected_dur
                dur_offset = frame_offsets[idx] + sample_frame_offset(n_frames[idx], selected_dur)

            yield np.stack(feats_mmap[dur_offset: dur_offset + selected_dur]).transpose((1, 0))


def sample_utt_int_per_spk(utt_ints_per_spk):
    """
    Randomly select an utterance (index) per speaker
    :param utt_ints_per_spk: ndarray or Series of list, e.g. [[], [], ..., []]
    :return: utts_se: ndarray of utterance per speaker
    """

    utt_ints_se = []

    for utt_ints in utt_ints_per_spk:
        idx = random.randint(0, len(utt_ints) - 1)
        utt_ints_se.append(utt_ints[idx])

    return np.asarray(utt_ints_se).astype(int)


def sample_frame_offsets(utt_lengths, sample_length):
    """ Get frame offsets of a number of utterances """
    return np.asarray([sample_frame_offset(utt_length, sample_length) for utt_length in utt_lengths])


def sample_frame_offset(utt_length, sample_length):
    """
    Get frame offset of an utterance
    :param utt_length: int, length of an utterance
    :param sample_length: int, length to be sampled
    :return: int, frame offset of the sample
    """
    assert 0 < sample_length <= utt_length, f'Sample length should be no greater than utterance length {utt_length}!'
    free_length = utt_length - sample_length

    return random.randint(0, free_length)


if __name__ == '__main__':
    # from torch.utils.data import DataLoader

    # random.seed(1)
    # np.random.seed(1)
    #
    # train_ds = TrainDataset(feat_dim=40, min_len=200, max_len=400, batch_size=256, n_repeats=1, mode='train')
    # train_dataloader = DataLoader(dataset=train_ds, batch_size=None, num_workers=2)
    #
    # for i, (mat, label) in enumerate(train_dataloader):
    #     print('---------------------mat---------------------')
    #     print(f'Batch {i}, mat.shape: {mat.shape}')
    #     print('---------------------label---------------------')
    #     print(f'Batch {i}, label.shape: {label.shape}\n')

    # eval_ds = EvalDataset(feat_dim=40, start_idx=0, end_idx=20)
    # eval_dataloader = DataLoader(dataset=eval_ds, num_workers=0)
    #
    # for mat in eval_dataloader:
    #     print('---------------------mat---------------------')
    #     print(f'mat.shape: {mat.shape}\n')

    print()
