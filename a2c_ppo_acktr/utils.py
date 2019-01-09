import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


import numpy as np
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import update_mean_var_count_from_moments

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.stack = []

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self,x):
        '''the normalize can minus mean is because the env is not returning any
        terminal signal, meaning that the episode length is not
        controlled by the agent.'''
        # return (x-np.asscalar(self.mean)) / np.asscalar(np.sqrt(self.var))
        return (x) / np.asscalar(np.sqrt(self.var))

    def stack_cuda_torch(self,x):
        self.stack += [x.cpu().numpy()]

    def update_from_stack(self):
        rffs_mean, rffs_std, rffs_count = mpi_moments(np.stack(self.stack).ravel())
        self.stack = []
        self.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)

    def stack_and_normalize(self, x):
        self.stack_cuda_torch(x)
        return self.normalize(x)

    def store(self, save_dir):
        to_save = {}
        to_save['mean'] = self.mean
        to_save['var'] = self.var
        to_save['count'] = self.count
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
            self.mean = loaded[()]['mean']
            self.var = loaded[()]['var']
            self.count = loaded[()]['count']
            print('{}: Restore Successed, {} restored.'.format(self.__class__.__name__, loaded))
        except Exception as e:
            print('{}: Restore Failed, due to {}.'.format(self.__class__.__name__,e))


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
