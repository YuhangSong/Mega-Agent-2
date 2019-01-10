import torch
import torch.nn as nn
from a2c_ppo_acktr.envs import VecNormalize
import tensorflow as tf
import os
import numpy as np

class IndexHashCountBouns():
    def __init__(self, k, batch_size, count_data_type, epsilon=0.01):
        """IndexHashCountBouns"""
        self.k = k
        self.batch_size = batch_size
        self.count_data_type = count_data_type
        self.epsilon = epsilon

        self.count = torch.Tensor(
            1,int(self.k**2)
        ).cuda().fill_(0)
        if self.count_data_type in ['long']:
            self.count = self.count.long()
            '''avoid initial error'''
            self.count += 1
        elif self.count_data_type in ['double']:
            self.count = self.count.double()
            '''avoid initial error'''
            self.count += 1.0
        else:
            raise NotImplemented

        self.check_data_type()

    def check_data_type(self):
        if self.count_data_type in ['long']:
            assert self.count.dtype == torch.long
        elif self.count_data_type in ['double']:
            assert self.count.dtype == torch.double
        else:
            raise NotImplemented

    def get_bouns_map(self):
        return (self.count.double()+self.epsilon).pow(0.5).reciprocal().float()

    def update_count(self, states):
        states_sum = states.sum(dim=0,keepdim=True)
        if self.count_data_type in ['long']:
            states_sum = states_sum.long()
        elif self.count_data_type in ['double']:
            states_sum = states_sum.double()
        else:
            raise NotImplemented
        self.count += states_sum

    def compute_bouns(self, states, keepdim):
        return (states*self.get_bouns_map().expand(states.size())).sum(dim=1,keepdim=keepdim)

    def get_bouns(self, states):
        bouns = self.compute_bouns(states)
        self.update_count(states)
        return bouns

    def store(self, save_dir):
        to_save = {}
        to_save['count'] = self.count.cpu().numpy()

        try:
            np.save(
                '{}.npy'.format(save_dir),
                to_save,
            )
            print('{}: Store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Store Failed.'.format(self.__class__.__name__))

    def restore(self, save_dir):
        try:
            # print('{}: Restoring {}.'.format(self.__class__.__name__,save_dir))
            loaded = np.load('{}.npy'.format(save_dir))
            self.count = torch.from_numpy(loaded[()]['count']).cuda()
            print('{}: Restore Successed, self.count: {}.'.format(self.__class__.__name__,self.count))
        except Exception as e:
            print('{}: Restore Failed.'.format(self.__class__.__name__))

        self.check_data_type()

def to_batch_version(x, batch_size):
    return x.repeat(batch_size, *(tuple([1]*len(x.size()[1:]))))

class DirectControlMask(object):
    """docstring for DirectControlMask."""
    def __init__(self, args):
        super(DirectControlMask, self).__init__()
        self.args = args

        import os
        path = os.path.join(
            './direct_control_masks', '{}_{}x{}.txt'.format(
                args.env_name.split('NoFrameskip')[0],
                args.num_grid,
                args.num_grid,
            ),
        )
        try:
            mask = torch.from_numpy(self.read_grid_map(path)).float().cuda()
        except Exception as e:
            print('# WARNING: No direct_control_mask loaded, as default')
            mask = torch.ones(self.args.num_grid,self.args.num_grid).float().cuda()
        assert mask.size()[0]==args.num_grid and mask.size()[1]==args.num_grid

        '''add batch dim'''
        mask = mask.unsqueeze(0)
        '''flatten'''
        mask = mask.view(mask.size()[0],-1)

        self.mask_batch = to_batch_version(mask,args.num_processes)

    def mask(self, x):
        return x*self.mask_batch

    def get_mask_batch(self):
        return self.mask_batch

    def read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array


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

def restore_learner(args, actor_critic, envs, j):
    try:
        actor_critic, ob_rms = torch.load(os.path.join(args.save_dir, 'learner' + ".pt"))
        if args.cuda:
            actor_critic = actor_critic.cuda()
        envs.ob_rms = ob_rms
        j = np.load(
            os.path.join(args.save_dir, "j.npy"),
        )[0]
        print('restore learner ok.')
    except Exception as e:
        print('restore learner failed: {}.'.format(e))

    return actor_critic, envs, j

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
