import torch
import torch.nn as nn
from a2c_ppo_acktr.envs import VecNormalize
import tensorflow as tf
import os
import numpy as np

class TF_Summary(object):
    """docstring for tf_summary."""
    def __init__(self, log_dir):
        super(TF_Summary, self).__init__()
        self.summary_writer = tf.summary.FileWriter(log_dir)

    def summary_and_flush(self, summay_dic, step):
        '''vis by tensorboard'''
        summary = tf.Summary()
        for name in summay_dic.keys():
            summary.value.add(
                tag = name,
                simple_value = summay_dic[name],
            )
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()


def store_learner(args, actor_critic, envs, j):
    import copy
    from a2c_ppo_acktr.utils import get_vec_normalize
    '''store learner'''
    try:
        # A really ugly way to save a model to CPU
        save_model = actor_critic
        if args.cuda:
            save_model = copy.deepcopy(actor_critic).cpu()
        save_model = [save_model,
                      getattr(get_vec_normalize(envs), 'ob_rms', None)]
        torch.save(save_model, os.path.join(args.save_dir, 'learner' + ".pt"))
        np.save(
            os.path.join(args.save_dir, "j.npy"),
            np.array([j]),
        )
        print('store learner ok.')
    except Exception as e:
        print('store learner failed: {}.'.format(e))

def restore_learner(args, actor_critic, envs):
    try:
        actor_critic, ob_rms = torch.load(os.path.join(args.save_dir, 'learner' + ".pt"))
        envs.ob_rms = ob_rms
        j = np.load(
            os.path.join(args.save_dir, "j.npy"),
        )[0]
        print('restore learner ok.')
        return j
    except Exception as e:
        print('restore learner failed: {}.'.format(e))
        return 0

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
