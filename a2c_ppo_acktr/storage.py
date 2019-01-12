import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

class PrioritizedReplayBuffer():
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
        super(PrioritizedReplayBuffer, self).__init__()
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

    def push(self, pushed, is_remove_inter_episode_transitions):
        """Push data into storage and pop data if overflows.
        Parameters
        ----------
        state, action, next_state: torch.Tensor(batch, ...)
        """

        if is_remove_inter_episode_transitions:
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

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            self.onehot_actions = torch.zeros(num_steps, num_processes, action_space.n)
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def put_process_axis_into_batch_axis(self, x):
        return x.view(x.size()[0]*x.size()[1], *x.size()[2:])

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.onehot_actions = self.onehot_actions.to(device)
        self.masks = self.masks.to(device)


    def insert_1(self, actions):
        self.actions[self.step].copy_(actions)
        self.onehot_actions[self.step].fill_(0.0).scatter_(1,self.actions[self.step],1.0)

    def insert_2(self, obs, recurrent_hidden_states, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]


    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
