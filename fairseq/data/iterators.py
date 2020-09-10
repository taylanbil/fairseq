# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import os

import numpy as np
import torch

from . import data_utils


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap

    Attributes:
        count (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=0):
        self.iterable = iterable
        self.count = start
        self.itr = iter(self)
        self.len = start + len(iterable)

    def __len__(self):
        return self.len

    def __iter__(self):
        for x in self.iterable:
            if self.count >= self.len:
                return
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.count < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.len = min(self.len, n)


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        raise NotImplementedError

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        raise NotImplementedError

    @property
    def iterations_in_epoch(self) -> int:
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError


class StreamingEpochBatchIterator(EpochBatchIterating):
    def __init__(
        self, dataset, epoch=0, num_shards=1, shard_id=0,
    ):
        # assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.epoch = epoch
        self._current_epoch_iterator = None
        self.num_shards = num_shards
        self.shard_id = shard_id

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self.epoch += 1
        self._current_epoch_iterator = CountingIterator(
            iterable=ShardedIterator(
                iterable=self.dataset,
                num_shards=self.num_shards,
                shard_id=self.shard_id,
            ),
        )
        return self._current_epoch_iterator

    def end_of_epoch(self) -> bool:
        return not self._current_epoch_iterator.has_next()

    @property
    def iterations_in_epoch(self) -> int:
        if self._current_epoch_iterator is not None:
            return self._current_epoch_iterator.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']


class EpochBatchIterator(EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 0).
    """

    def __init__(
        self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=0,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers

        self.epoch = epoch
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        # self.dataset.set_epoch(self.epoch)
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get('shuffle', True),
                offset=itr_pos,
            )

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):

        def shuffle_batches(batches, seed):
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        return CountingIterator(
            torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_sampler=batches[offset:],
                num_workers=self.num_workers,
            ),
            start=offset,
        )


class GroupedIterator(object):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
    """

    def __init__(self, iterable, chunk_size):
        self._len = int(math.ceil(len(iterable) / float(chunk_size)))
        self.offset = int(math.ceil(getattr(iterable, 'count', 0) / float(chunk_size)))
        self.itr = iterable
        self.chunk_size = chunk_size

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def get_len(self):
        lengths = [
            (11, 288),
            (12, 368),
            (13, 368),
            (14, 544),
            (15, 624),
            (16, 664),
            (17, 716),
            (18, 972),
            (19, 1160),
            (20, 1392),
            (21, 1488),
            (22, 1976),
            (23, 2376),
            (24, 2656),
            (25, 3312),
            (26, 4024),
            (27, 4928),
            (28, 5464),
            (29, 6032),
            (30, 6752),
            (31, 8312),
            (32, 9344),
            (33, 11040),
            (34, 12792),
            (35, 15640),
            (36, 18664),
            (37, 22264),
            (38, 26816),
            (39, 33548),
            (40, 41364),
            (41, 51440),
            (42, 65664),
            (43, 83984),
            (44, 109008),
            (45, 140025),
            (46, 182023),
            (47, 233128),
            (48, 305928),
            (49, 436128),
            (50, 597247),
            (51, 202808)
        ]
        lengths = np.array(lengths)
        p = lengths[:, 1]
        p = p / p.sum()
        length = np.random.choice(lengths[:, 0], p=p)
        lower, upper = length*10, min(511, length*10+10)
        length = np.random.randint(lower, upper)
        return length


    def adjust_len(self, item, length):
        try:
            itemlen = max(item['net_input']['src_lengths'])
        except KeyError:
            # in dist. mode, sometimes the last batch is empty
            return item
        if length == itemlen:
            return item
        bsz = item['nsentences']
        # ntokens
        item['ntokens'] = length * bsz
        # srclens
        item['net_input']['src_lengths'] = torch.tensor([length]*bsz)
        if length > itemlen:
            # srctokens
            x = item['net_input']['src_tokens']
            lasttokens = x[:, itemlen-1].clone()
            x[:, length-1] = lasttokens
            x[:, itemlen-1:length-1] = torch.randint(10, 10000, (bsz, length-itemlen))
            # target
            x = item['target']
            lasttokens = x[:, itemlen-1].clone()
            x[:, itemlen-1:] = 1
            x[:, length-1] = lasttokens
        else:
            # srctokens
            x = item['net_input']['src_tokens']
            x[:, length:] = 1
            x[:, length-1] = 2
            # target
            x = item['target']
            x[:, length-1:] = 1
        return item

    def __next__(self):
        chunk = []
        try:
            for _ in range(self.chunk_size):
                length = self.get_len()
                item = next(self.itr)
                item = self.adjust_len(item, length)
                try:
                    if item['net_input']['src_tokens'].shape[1] != 512:
                        raise ValueError(
                            str(item['net_input']['src_tokens'].shape[1])
                        )
                except KeyError:
                    pass
                chunk.append(item)
        except StopIteration as e:
            if len(chunk) == 0:
                raise e
        return chunk


class ShardedIterator(object):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]
