#code from openai
#https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random
import torch

import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, next_state):
        data = (state, action, next_state)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, obses_tp1 = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, obs_tp1 = data
            obses_t.append(obs_t)
            actions.append(action)
            obses_tp1.append(obs_tp1)
        return torch.stack(obses_t), torch.stack(actions), torch.stack(obses_tp1)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

import torch

class PrioritizedReplayBufferPure():
    def __init__(self, size, mode, init_list):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the memories of least priority are dropped.
        mode: str
            priority, random
        """
        super(PrioritizedReplayBufferPure, self).__init__()
        self._maxsize = size
        self._max_priority = 0.0
        self.mode = mode

        '''things to store'''
        self.storage = {}
        for name in init_list:
            self.storage[name] = None
        self.priority = None

        self.max_priority_batch = {}

    def torch_stack(self, x, new_x):
        """Stack new_x into x on batch axis.
        Parameters
        ----------
        x: torch.Tensor(batch, ...)
        new_x: torch.Tensor(1, ...)
        """
        if x is None:
            x = new_x
        else:
            x = torch.cat([x,new_x],0)
        return x

    def np_stack(self, x, new_x):
        """Stack new_x into x on batch axis.
        Parameters
        ----------
        x: np.array(batch, ...)
        new_x: np.array(1, ...)
        """
        if x is None:
            x = new_x
        else:
            x = np.concatenate((x,new_x),0)
        return x

    def torch_delete(self, a, idx):
        """Delete a slice at batch axis according to idx.
        Parameters
        ----------
        x: torch.Tensor(batch, ...)
        idx: int
        """
        return torch.cat([a[:idx], a[idx+1:]])

    def get_max_priority_batch(self, batch_size):
        """Get max_priority_batch.
        Parameters
        ----------
        batch_size: int
        """
        if batch_size not in self.max_priority_batch.keys():
            self.max_priority_batch[batch_size] = np.array([self._max_priority]*batch_size)

        return self.max_priority_batch[batch_size]

    def push(self, pushed):
        """Push data into storage and pop data if overflows.
        Parameters
        ----------
        state, action, next_state: torch.Tensor(batch, ...)
        """

        pushed = self.remove_inter_episode_transitions(pushed)

        for name in pushed.keys():
            self.storage[name] = self.torch_stack(self.storage[name], pushed[name])
        '''new data is assigned with _max_priority so that they are garanteed to be sampled for the
        first time, then their priority is refreshed in update_priorities(), so that they will be sampled
        according to priority since then.'''

        self.priority    = self.np_stack   (self.priority   , self.get_max_priority_batch(pushed[list(pushed.keys())[0]].size()[0]))

    def constrain_buffer_size(self):
        '''pop data, only leave the ones with max priority'''
        if self.priority.shape[0]>self._maxsize:

            self.storage, idxes = self.sample(
                batch_size = self._maxsize,
            )
            self.priority = np.take(self.priority,idxes)
            return 'constrained'

        else:
            return 'not constrained'

    def torch_take(self, x, idxes):
        """Take a slice at batch axis according to idxes.
        Parameters
        ----------
        x: torch.Tensor(batch, ...)
        idxes: torch.Tensor([int_idx0,int_idx1,...])
        """
        if len(x.size())==4:
            idxes = idxes.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        elif len(x.size())==2:
            idxes = idxes.unsqueeze(1)
        else:
            raise NotImplemented

        return x.gather(0, idxes.expand(idxes.size()[0],*x.size()[1:]))

    def remove_inter_episode_transitions(self,x):
        idxes = x['next_state_masks'].nonzero()[:,0]
        return self.torch_sample_storage_by_idxes(x,idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        """
        '''To get the indices of the batch_size largest elements'''
        if self.mode in ['priority']:
            idxes = np.argpartition(self.priority, -batch_size)[-batch_size:]
        elif self.mode in ['random']:
            idxes = np.random.randint(low=0, high=self.priority.shape[0], size=batch_size, dtype=np.int64)
        else:
            raise NotImplemented
        sampled = self.torch_sample_storage_by_idxes(self.storage, torch.from_numpy(idxes).cuda())
        return sampled, idxes

    def torch_sample_storage_by_idxes(self, to_sample, idxes):
        """simple torch to_sample dic according to idxes.
        Parameters
        ----------
        to_sample: dic of torch Tensor.
        idxes: torch int Tensor.
        """
        sampled = {}
        for name in to_sample.keys():
            sampled[name] = self.torch_take(to_sample[name],idxes)
        return sampled

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: np.array([int_idx0,int_idx1,...])
            List of idxes of sampled transitions
        priorities: np.array([float_priority0,float_priority1,...])
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        np.put(self.priority, idxes, priorities)
        self._max_priority = np.amax([self._max_priority, np.amax(priorities)])

    def store(self, save_dir):
        to_save = {}
        for name in self.storage.keys():
            to_save[name] = self.storage[name].cpu().numpy()
        to_save['priority'] = self.priority
        try:
            np.save(
                '{}.npy'.format(save_dir),
                to_save,
            )
            print('{}: Store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Store Failed, due to {}.'.format(self.__class__.__name__,e))

    def restore(self, save_dir):
        try:
            print('{}: Restoring {}.'.format(self.__class__.__name__,save_dir))
            loaded = np.load('{}.npy'.format(save_dir))
            for name in self.storage.keys():
                self.storage[name] = torch.from_numpy(loaded[()][name]).cuda()
            self.priority = np.squeeze(loaded[()][name],1)
            print('{}: Restore Successed, {} samples restored.'.format(self.__class__.__name__, self.storage[list(self.storage.keys())[0]].size()[0]))
        except Exception as e:
            print('{}: Restore Failed, due to {}.'.format(self.__class__.__name__,e))

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        # self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 0.00001

    def push(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        # self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        # weights = []
        # p_min = self._it_min.min() / self._it_sum.sum()
        # max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            # weights.append(weight / max_weight)
        # weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        # return tuple(list(encoded_sample) + [weights, idxes])
        return tuple(list(encoded_sample) + [idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            # self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
