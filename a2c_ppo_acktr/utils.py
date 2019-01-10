import torch
import torch.nn as nn
from a2c_ppo_acktr.envs import VecNormalize
import tensorflow as tf
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def figure_to_array(fig):
    canvas=fig.canvas
    buf = io.BytesIO()
    canvas.print_png(buf)
    data=buf.getvalue()
    buf.close()
    buf=io.BytesIO()
    buf.write(data)
    img=Image.open(buf)
    img = np.asarray(img)
    return img

def to_plot(curves):
    import cv2
    import matplotlib.pyplot as plt
    plt.clf()
    line_list = []
    for key in curves.keys():
        line, = plt.plot(curves[key], label=key)
        line_list += [line]
    plt.legend(handles=line_list)
    state_img = figure_to_array(plt.gcf())
    state_img = cv2.cvtColor(state_img, cv2.cv2.COLOR_RGBA2RGB)
    return state_img

def to_mask_img(x, latent_control_model):
    x = latent_control_model.to_mask_img(x)[0].squeeze(0)
    x = torch_add_edge(x,add_value=1.0)
    return (x*255.0).cpu().numpy().astype(np.uint8)

def torch_add_edge(x, add_value=1.0):
    return torch.cat(
        [
            x,
            (x[:,:10]*0.0+add_value),
        ],
        dim = 1,
    )

def numpy_add_edge(x, add_value=255):
    return np.concatenate(
        (
            x,
            (x[:,:10]*0.0+add_value),
        ),
        axis = 1,
    ).astype(np.uint8)

def torch_end_point_norm(x,dim):
    x_max  = x.max (dim=dim,keepdim=True)[0].expand(x.size())
    x_min  = x.min (dim=dim,keepdim=True)[0].expand(x.size())
    return (x-x_min)*((x_max-x_min).reciprocal())

def display_normed_obs(obs):
    '''(xx,xx) -> (xx,xx) 0-255'''
    obs = obs.astype(np.float)
    return ((obs-np.amin(obs))/(np.amax(obs)-np.amin(obs))*255.0).astype(np.uint8)

def draw_obs_from_rollout(x, latent_control_model):
    return numpy_add_edge(
        latent_control_model.draw_grid_on_img(
            display_normed_obs(
                x.squeeze(0).cpu().numpy()
            )
        ),
        add_value = 255,
    )

def mask_img(x, img_mask):
    '''
        x: (xx,xx) 0-255
        img_mask: (xx,xx) {0,255}
    '''
    return (
        x.astype(np.float)
        *
        (img_mask.astype(np.float)/255.0)
    ).astype(np.uint8)

class VideoSummary(object):
    """docstring for VideoSummary."""
    def __init__(self, args):
        super(VideoSummary, self).__init__()
        self.args = args

        self.video_length = 0
        self.video_count = 0
        self.reset_summary()

    def reset_summary(self):
        self.curves = {}
        self.video_writer = None

    def summary_a_video(self, video_length):
        if self.video_count==self.video_length:
            '''no video is being summarized now'''
            self.video_length = video_length
            self.video_count = 0
            self.reset_summary()

    def stack(self, args, last_states, now_states, onehot_actions, latent_control_model, direct_control_mask, M, G, delta_uG, curves, num_trained_frames):

        if self.video_count<self.video_length:

            '''last_state and now_state'''
            state_img = np.concatenate(
                (
                    draw_obs_from_rollout(last_states[0,-1:],latent_control_model), # last state
                ),
                1,
            )

            if latent_control_model is not None:

                batch_size = now_states.size()[0]
                if args.random_noise_frame:
                    latent_control_model.randomize_noise_masks(batch_size)
                    now_states = latent_control_model.add_noise_masks(now_states)
                    last_states = latent_control_model.add_noise_masks(last_states)

                latent_control_model.eval()
                '''(batch_size, ...) -> (batch_size*to_each_grid, ...)'''
                predicted_now_states, _,  _, _ = latent_control_model.get_predicted_now_states(
                    last_states    = last_states,
                    now_states     = now_states,
                    onehot_actions = onehot_actions,
                )
                predicted_now_states = predicted_now_states.detach()
                '''(batch_size*to_each_grid, ...) -> (batch_size, to_each_grid, ...)'''
                predicted_now_states = latent_control_model.extract_grid_axis_from_batch_axis(predicted_now_states)
                '''(batch_size, to_each_grid, ...) -> (batch_size, ...)'''
                predicted_now_states = latent_control_model.degrid_states(predicted_now_states)

                state_img = np.concatenate(
                    (
                        state_img,
                        draw_obs_from_rollout(now_states[0], latent_control_model),
                        draw_obs_from_rollout(predicted_now_states[0], latent_control_model), # predicted now state
                    ),
                    1,
                )

            '''direct_control_mask'''
            state_img = np.concatenate(
                (
                    state_img,
                    mask_img(
                        x = draw_obs_from_rollout(now_states[0], latent_control_model),
                        img_mask = to_mask_img(direct_control_mask.get_mask_batch()[:1],latent_control_model),
                    ),
                ),
                1,
            )

            if M is not None:
                state_img = np.concatenate(
                    (
                        state_img,
                        to_mask_img(M[:1],latent_control_model),
                    ),
                    1,
                )

            if G is not None:
                if args.latent_control_intrinsic_reward_type.split('__')[4] in ['clip_G']:
                    state_img = np.concatenate(
                        (
                            state_img,
                            to_mask_img(G[:1],latent_control_model),
                        ),
                        1,
                    )
                elif args.latent_control_intrinsic_reward_type.split('__')[4] in ['NONE']:
                    state_img = np.concatenate(
                        (
                            state_img,
                            to_mask_img(torch_end_point_norm(G[:1]),latent_control_model),
                        ),
                        1,
                    )
                else:
                    raise NotImplemented

            if delta_uG is not None:
                state_img = np.concatenate(
                    (
                        state_img,
                        to_mask_img(
                            (delta_uG[:1]+1.0)/2.0,
                            latent_control_model,
                        ),
                    ),
                    1,
                )

            # '''bouns_map'''
            # try:
            #     state_img = np.concatenate(
            #         (
            #             state_img,
            #             to_mask_img(
            #                 utils.torch_end_point_norm(
            #                     hash_count_bouns.get_bouns_map(),
            #                     dim = 1,
            #                 ),
            #                 latent_control_model,
            #             ),
            #         ),
            #         1,
            #     )
            # except Exception as e:
            #     pass

            for name in curves.keys():
                try:
                    self.curves[name] += [curves[name]]
                except Exception as e:
                    self.curves[name] = [curves[name]]

            '''episode_curve_stack'''
            state_img = np.concatenate(
                (
                    state_img,
                    cv2.cvtColor(
                        cv2.resize(
                            to_plot(
                                self.curves
                            ),
                            (state_img.shape[1], state_img.shape[1]),
                        ),
                        cv2.cv2.COLOR_RGB2GRAY,
                    ),
                ),
                0,
            )

            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(
                    '{}/video_summary_{}.avi'.format(
                        self.args.log_dir,
                        num_trained_frames,
                    ),
                    cv2.VideoWriter_fourcc('M','J','P','G'),
                    5,
                    (state_img.shape[1],state_img.shape[0]),
                    False
                )

            print('SUMMARY [{}]'.format(self.video_count))
            self.video_writer.write(state_img)

            self.video_count += 1

            if self.video_count>=self.video_length:
                self.video_writer.release()
                self.reset_summary()

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
            input('# WARNING: No direct_control_mask loaded, as default')
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
    def __init__(self, args, is_visdom=False):
        super(TF_Summary, self).__init__()
        self.args = args
        self.summary_writer = tf.summary.FileWriter(args.log_dir)
        self.is_visdom = is_visdom
        if self.is_visdom:
            from visdom import Visdom
            self.viz = Visdom(port=args.port)
            self.win = None

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

        '''vis by visdom'''
        if self.is_visdom:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                self.win = visdom_plot(self.viz, self.win, self.args.log_dir, self.args.env_name,
                                  self.args.algo, self.args.num_env_steps)
            except IOError:
                pass


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
