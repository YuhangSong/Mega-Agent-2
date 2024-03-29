from baselines.common.running_mean_std import update_mean_var_count_from_moments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def to_batch_version(x, batch_size):
    return x.repeat(batch_size, *(tuple([1] * len(x.size()[1:]))))


spaces = ''
max_print_len = 80
for i in range(max_print_len):
    spaces += ' '

# from baselines.common.mpi_moments import mpi_moments


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):
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

    def normalize(self, x):
        '''the normalize can minus mean is because the env is not returning any
        terminal signal, meaning that the episode length is not
        controlled by the agent.'''
        # return (x-np.asscalar(self.mean)) / np.asscalar(np.sqrt(self.var))
        return (x) / np.asscalar(np.sqrt(self.var))

    def stack_cuda_torch(self, x):
        self.stack += [x.cpu().numpy()]

    def update_from_stack(self):
        rffs_mean, rffs_std, rffs_count = mpi_moments(
            np.stack(self.stack).ravel())
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
            print('# INFO: {} store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('# WARNING: {} store Failed, due to {}.'.format(
                self.__class__.__name__, e))

    def restore(self, save_dir):
        try:
            loaded = np.load('{}.npy'.format(save_dir))
            self.mean = loaded[()]['mean']
            self.var = loaded[()]['var']
            self.count = loaded[()]['count']
            print('# INFO: {} restore Successed, {} restored.'.format(
                self.__class__.__name__, loaded))
        except Exception as e:
            print('# WARNING: {} restore Failed, due to {}.'.format(
                self.__class__.__name__, e))


def clear_print_line():
    print(spaces, end="\r")


def clear_print(string_to_print):
    clear_print_line()
    print(string_to_print, end="\r")


class ObsNorm(object):
    """docstring for ObsNorm."""

    def __init__(self, envs, num_processes, nsteps):
        super(ObsNorm, self).__init__()
        self.envs = envs
        self.num_processes = num_processes
        self.nsteps = nsteps

    def random_agent_ob_mean_std(self):

        obs = self.envs.reset()[:, -1:].cpu()

        action = torch.LongTensor(self.num_processes, 1).cuda()

        for i in range(self.nsteps):
            clear_print(
                '# INFO: Running ObsNorm [{}/{}]'.format(i, self.nsteps))
            action.random_(0, self.envs.action_space.n)
            obs_new = self.envs.step(action)[0][:, -1:].cpu()
            obs = torch.cat(
                [obs, obs_new],
                dim=0,
            )

        self.ob_mean = to_batch_version(
            obs.mean(dim=0, keepdim=True).cuda(),
            self.num_processes,
        )
        self.ob_std = obs.std(dim=0).mean().item()
        self.ob_bound = 255.0 / self.ob_std

    def obs_norm_batch(self, obs):
        return ((obs - self.ob_mean) / self.ob_std)

    def obs_denorm_single(self, obs):
        return ((obs * self.ob_std) + self.ob_mean[0][-1:])

    def obs_display_norm_single(self, obs):
        return ((obs * self.ob_std) + 255.0) / 2.0

    def restore(self, log_dir):
        try:
            self.ob_mean = torch.from_numpy(
                np.load(log_dir + '/ob_mean.npy')).cuda()
            self.ob_std = np.load(log_dir + '/ob_std.npy')[0]
            self.ob_bound = np.load(log_dir + '/ob_bound.npy')[0]
            print('# INFO: Restore ObsNorm: Successed.')
        except Exception as e:
            print('# WARNING: Restore ObsNorm: Failed')
            self.random_agent_ob_mean_std()
        print('# INFO: Estimated mean shape {}; std {} bound {}'.format(
            self.ob_mean.size(),
            self.ob_std,
            self.ob_bound),
        )

    def store(self, log_dir):
        try:
            np.save(
                log_dir + '/ob_mean.npy',
                self.ob_mean.cpu().numpy(),
            )
            np.save(
                log_dir + '/ob_std.npy',
                np.asarray([self.ob_std]),
            )
            np.save(
                log_dir + '/ob_bound.npy',
                np.asarray([self.ob_bound]),
            )
            print('# INFO: Store ObsNorm: Successed.')
        except Exception as e:
            print('# WARNING: Store ObsNorm: Failed')


def figure_to_array(fig):
    canvas = fig.canvas
    buf = io.BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()
    buf.close()
    buf = io.BytesIO()
    buf.write(data)
    img = Image.open(buf)
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


def points_to_mask_img(x, args):
    x = x.unsqueeze(2).expand(-1, -1, args.size_grid)
    x = x.contiguous().view(x.size()[0], args.num_grid, -1)
    x = torch.cat([x] * args.size_grid, dim=2).view(x.size()[0],
                                                    args.size_grid * args.num_grid, args.size_grid * args.num_grid)
    return x.unsqueeze(1)


def to_mask_img(x, args):
    x = points_to_mask_img(x, args)[0].squeeze(0)
    x = torch_add_edge(x, add_value=1.0)
    return (x * 255.0).cpu().numpy().astype(np.uint8)


def torch_add_edge(x, add_value=1.0):
    return torch.cat(
        [
            x,
            (x[:, :10] * 0.0 + add_value),
        ],
        dim=1,
    )


def numpy_add_edge(x, add_value=255):
    return np.concatenate(
        (
            x,
            (x[:, :10] * 0.0 + add_value),
        ),
        axis=1,
    ).astype(np.uint8)


def torch_end_point_norm(x, dim):
    x_max = x.max(dim=dim, keepdim=True)[0].expand(x.size())
    x_min = x.min(dim=dim, keepdim=True)[0].expand(x.size())
    return (x - x_min) * ((x_max - x_min).reciprocal())


def display_normed_obs(obs, epsilon):
    '''(xx,xx) (-epsilon)-(255+epsilon)-> (xx,xx) 0-255'''
    # return ((obs.astype(np.float)+epsilon)/(255.0+epsilon*2.0)*255.0).astype(np.uint8)
    return ((obs.astype(np.float) - np.min(obs)) / (np.max(obs) - np.min(obs)) * 255.0).astype(np.uint8)


class GridImg(object):
    """docstring for GridImg."""

    def __init__(self, args):
        super(GridImg, self).__init__()
        self.args = args

        self.grid_mask = np.zeros(
            (self.args.obs_size, self.args.obs_size), dtype=np.float)

        for i in range(self.args.num_grid):
            self.grid_mask[i * self.args.size_grid, :] = 1.0

        for j in range(self.args.num_grid):
            self.grid_mask[:, j * self.args.size_grid] = 1.0

    def draw_grid_on_img(self, img):
        '''(xx,xx) 0-255 >> (xx,xx) 0-255'''
        img = (img.astype(np.float) * (1.0 - self.grid_mask) +
               self.grid_mask * 255.0).astype(np.uint8)
        return img


def draw_obs_from_rollout(x, grid_img, epsilon):
    return numpy_add_edge(
        grid_img.draw_grid_on_img(
            display_normed_obs(
                x.squeeze(0).cpu().numpy(),
                epsilon,
            )
        ),
        add_value=255,
    )


def mask_img(x, img_mask):
    '''
        x: (xx,xx) 0-255
        img_mask: (xx,xx) {0,255}
    '''
    return (
        x.astype(np.float)
        *
        (img_mask.astype(np.float) / 255.0)
    ).astype(np.uint8)


class VideoSummary(object):
    """docstring for VideoSummary."""

    def __init__(self, args):
        super(VideoSummary, self).__init__()
        self.args = args

        self.video_length = 0
        self.video_count = 0
        self.reset_summary()
        self.grid_img = GridImg(self.args)

    def reset_summary(self):
        self.curves = {}
        self.video_writer = None

    def summary_a_video(self, video_length):
        if self.video_count == self.video_length:
            '''no video is being summarized now'''
            self.video_length = video_length
            self.video_count = 0
            self.reset_summary()

    def is_summarizing(self):
        return (self.video_count < self.video_length)

    def stack(self, args, last_states, now_states, onehot_actions, latent_control_model,
              direct_control_mask, hash_count_bouns, obs_norm, M, G, delta_uG,
              curves, num_trained_frames, map_to_use, x_mean_to_norm):

        if self.video_count < self.video_length:

            '''last_state and now_state'''
            state_img = np.concatenate(
                (
                    draw_obs_from_rollout(
                        obs_norm.obs_denorm_single(
                            last_states[0, -1:],
                        ),
                        self.grid_img,
                        self.args.epsilon,
                    ),  # last state
                    draw_obs_from_rollout(
                        obs_norm.obs_display_norm_single(
                            last_states[0, -1:],
                        ),
                        self.grid_img,
                        self.args.epsilon,
                    ),  # last state
                ),
                1,
            )

            if latent_control_model is not None:

                batch_size = now_states.size()[0]
                if args.random_noise_frame:
                    latent_control_model.randomize_noise_masks(batch_size)
                    now_states = latent_control_model.add_noise_masks(
                        now_states)
                    last_states = latent_control_model.add_noise_masks(
                        last_states)

                latent_control_model.eval()
                '''(batch_size, ...) -> (batch_size*to_each_grid, ...)'''
                predicted_now_states, _,  _, _ = latent_control_model.get_predicted_now_states(
                    last_states=last_states,
                    now_states=now_states,
                    onehot_actions=onehot_actions,
                )
                predicted_now_states = predicted_now_states.detach()
                '''(batch_size*to_each_grid, ...) -> (batch_size, to_each_grid, ...)'''
                predicted_now_states = latent_control_model.extract_grid_axis_from_batch_axis(
                    predicted_now_states)
                '''(batch_size, to_each_grid, ...) -> (batch_size, ...)'''
                predicted_now_states = latent_control_model.degrid_states(
                    predicted_now_states)

                state_img = np.concatenate(
                    (
                        state_img,
                        draw_obs_from_rollout(
                            obs_norm.obs_denorm_single(
                                now_states[0],
                            ),
                            self.grid_img,
                            self.args.epsilon,
                        ),
                        draw_obs_from_rollout(
                            obs_norm.obs_denorm_single(
                                predicted_now_states[0],
                            ),
                            self.grid_img,
                            self.args.epsilon,
                        ),  # predicted now state
                    ),
                    1,
                )

            '''direct_control_mask'''
            state_img = np.concatenate(
                (
                    state_img,
                    mask_img(
                        x=draw_obs_from_rollout(
                            obs_norm.obs_denorm_single(
                                now_states[0],
                            ),
                            self.grid_img,
                            self.args.epsilon,
                        ),
                        img_mask=to_mask_img(
                            direct_control_mask.get_mask_batch()[:1], self.args),
                    ),
                ),
                1,
            )

            if M is not None:
                state_img = np.concatenate(
                    (
                        state_img,
                        to_mask_img(M[:1], self.args),
                    ),
                    1,
                )

            if G is not None:
                if args.latent_control_intrinsic_reward_type.split('__')[4] in ['clip_G']:
                    state_img = np.concatenate(
                        (
                            state_img,
                            to_mask_img(G[:1], self.args),
                        ),
                        1,
                    )
                elif args.latent_control_intrinsic_reward_type.split('__')[4] in ['NONE']:
                    state_img = np.concatenate(
                        (
                            state_img,
                            to_mask_img(torch_end_point_norm(
                                G[:1]), self.args),
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
                            (delta_uG[:1] + 1.0) / 2.0,
                            self.args,
                        ),
                    ),
                    1,
                )

            state_img = np.concatenate(
                (
                    state_img,
                    to_mask_img(map_to_use[:1], self.args),
                ),
                1,
            )

            if x_mean_to_norm is not None:
                state_img = np.concatenate(
                    (
                        state_img,
                        to_mask_img(x_mean_to_norm[:1], self.args),
                    ),
                    1,
                )

            if hash_count_bouns is not None:
                try:
                    '''bouns_map'''
                    state_img = np.concatenate(
                        (
                            state_img,
                            to_mask_img(
                                hash_count_bouns.get_bouns_map(),
                                self.args,
                            ),
                        ),
                        1,
                    )
                except Exception as e:
                    pass

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
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    5,
                    (state_img.shape[1], state_img.shape[0]),
                    False
                )

            clear_print(
                '# INFO: [SUMMARY {}/{}]'.format(self.video_count, self.video_length))
            self.video_writer.write(state_img)

            self.video_count += 1

            if self.video_count >= self.video_length:
                self.video_writer.release()
                self.reset_summary()


class RunningBinaryNorm():
    def __init__(self):
        """SimHashCountBouns"""
        self.mean = None
        self.count = 0
        self.check_data_type()

    def check_data_type(self):
        pass

    def norm(self, x, is_stack):

        if is_stack:
            '''moment'''
            x_mean = x.mean(dim=0, keepdim=True)
            x_count = x.size()[0]
            '''update from moment'''
            if self.mean is None:
                self.mean = x_mean
            else:
                self.mean = self.mean * self.count / \
                    (self.count + x_count) + x_mean * \
                    x_count / (self.count + x_count)
            self.count += x_count
        else:
            self.mean = x.mean(dim=0, keepdim=True)

        '''norm to binary'''
        x_mean_to_norm = self.mean.expand(x.size())
        return (x - x_mean_to_norm).sign().clamp(0.0, 1.0), x_mean_to_norm

    def store(self, save_dir):
        to_save = {}
        to_save['count'] = np.array([self.count])
        if self.mean is not None:
            to_save['mean'] = self.mean.cpu().numpy()

        try:
            np.save(
                '{}.npy'.format(save_dir),
                to_save,
            )
            print('# INFO: {} store Successed. Store {}.'.format(
                self.__class__.__name__, to_save))
        except Exception as e:
            print('# WARNING: store Failed.'.format(self.__class__.__name__))

    def restore(self, save_dir):
        try:
            loaded = np.load('{}.npy'.format(save_dir))
            self.count = loaded[()]['count'][0]
            self.mean = torch.from_numpy(loaded[()]['mean']).cuda()
            print('# INFO: {} restore Successed. Restore {}.'.format(
                self.__class__.__name__, loaded[()]))
        except Exception as e:
            print('# WARNING: restore Failed.'.format(self.__class__.__name__))

        self.check_data_type()


class SimHashCountBouns():
    def __init__(self, D, k, batch_size):
        """SimHashCountBouns"""

        self.D = D
        self.k = k
        self.batch_size = batch_size
        self.m = 2

        '''to be build according to batch_size'''
        A = torch.FloatTensor(1, self.D, self.k).normal_(
            mean=0.0, std=1.0).cuda()
        bin_to_hex = torch.from_numpy(
            self.m**np.arange(self.k)
        ).unsqueeze(0).cuda()
        self.As = to_batch_version(A, batch_size)
        self.bin_to_hexs = to_batch_version(bin_to_hex, batch_size)

        '''count is maitained in cpu to sace gpu memory'''
        self.count = torch.LongTensor(
            int(np.sum(
                (np.array([self.m - 1] * self.k))
                *
                (self.m**np.arange(self.k))
            ) + 1)
        ).cpu().fill_(1)

        self.check_data_type()

    def check_data_type(self):
        assert self.bin_to_hexs.dtype == torch.long
        assert self.count.dtype == torch.long

    def get_bouns(self, states, keepdim, is_stack):
        '''SimHash'''
        # (b,1,D) * (b,D,k) = (b,1,k)
        hashes = torch.bmm(
            states.unsqueeze(1),
            self.As,
        ).squeeze(1).sign().clamp(min=0, max=1).long()

        '''hashes to indexes'''
        indexes = (hashes * self.bin_to_hexs).sum(dim=1, keepdim=False)

        if is_stack:
            '''count'''
            for i in range(indexes.size()[0]):
                self.count[indexes[i]] += 1

        '''compute bouns'''
        bouns = self.count.gather(
            0,
            indexes.cpu(),
        ).cuda().float().pow(0.5).reciprocal()

        if keepdim:
            bouns = bouns.unsqueeze(1)

        return bouns

    def store(self, save_dir):
        to_save = {}
        to_save['As'] = self.As.cpu().numpy()
        to_save['bin_to_hexs'] = self.bin_to_hexs.cpu().numpy()
        # to_save['count'] = self.count.numpy()

        try:
            np.save(
                '{}.npy'.format(save_dir),
                to_save,
            )
            print('# INFO: {} store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('# WARNING: {} store Failed.'.format(self.__class__.__name__))

    def restore(self, save_dir):
        try:
            loaded = np.load('{}.npy'.format(save_dir))
            self.As = torch.from_numpy(loaded[()]['As']).cuda()
            self.bin_to_hexs = torch.from_numpy(
                loaded[()]['bin_to_hexs']).cuda()
            # self.count = torch.from_numpy(loaded[()]['count']).cpu()
            print('# INFO: {} restore Successed, self.count: {}.'.format(
                self.__class__.__name__, self.count.size()))
        except Exception as e:
            print('# WARNING: {} restore Failed.'.format(self.__class__.__name__))

        self.check_data_type()


class HardHashCountBouns():
    def __init__(self, k, m, batch_size):
        self.k = k
        self.m = m
        self.batch_size = batch_size

        '''to be build according to batch_size'''
        bin_to_hex = torch.from_numpy(
            self.m**np.arange(self.k)
        ).unsqueeze(0).cuda()
        self.bin_to_hexs = to_batch_version(bin_to_hex, batch_size)

        '''count is maitained in cpu to save gpu memory'''
        self.count = torch.LongTensor(
            int(np.sum(
                (np.array([self.m - 1] * self.k))
                *
                (self.m**np.arange(self.k))
            ) + 1)
        ).cpu().fill_(1)

        self.check_data_type()

    def check_data_type(self):
        assert self.bin_to_hexs.dtype == torch.long
        assert self.count.dtype == torch.long

    def get_bouns(self, states, keepdim, is_stack):
        '''HardHash'''
        hashes = (states * self.m).floor().long().clamp(min=0, max=(self.m - 1))

        '''hashes to indexes'''
        indexes = (hashes * self.bin_to_hexs).sum(dim=1, keepdim=False)

        if is_stack:
            '''count'''
            for i in range(indexes.size()[0]):
                self.count[indexes[i]] += 1

        '''compute bouns'''
        bouns = self.count.gather(
            0,
            indexes.cpu(),
        ).cuda().float().pow(0.5).reciprocal()

        if keepdim:
            bouns = bouns.unsqueeze(1)

        return bouns

    def store(self, save_dir):
        to_save = {}
        # to_save['count'] = self.count.numpy()

        try:
            np.save(
                '{}.npy'.format(save_dir),
                to_save,
            )
            print('# INFO: {} store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('# WARNING: {} store Failed.'.format(self.__class__.__name__))

    def restore(self, save_dir):
        try:
            loaded = np.load('{}.npy'.format(save_dir))
            self.count = torch.from_numpy(loaded[()]['count']).cpu()
            print('# INFO: {} restore Successed, self.count: {}.'.format(
                self.__class__.__name__, self.count.size()))
        except Exception as e:
            print('# WARNING: {} restore Failed.'.format(self.__class__.__name__))

        self.check_data_type()


class IndexHashCountBouns():
    def __init__(self, k, batch_size, count_data_type, is_normalize):
        """IndexHashCountBouns"""
        self.k = k
        self.batch_size = batch_size
        self.count_data_type = count_data_type
        self.is_normalize = is_normalize

        self.count = torch.Tensor(
            1, int(self.k**2)
        ).cuda().fill_(0)
        if self.count_data_type in ['long']:
            self.count = self.count.long().fill_(1)
        elif self.count_data_type in ['double']:
            self.count = self.count.double().fill_(1.0)
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
        bouns_map = self.count.double().pow(0.5).reciprocal().float()
        if self.is_normalize:
            bouns_map = F.softmax(bouns_map, dim=1)
        return bouns_map

    def update_count(self, states):
        states_sum = states.sum(dim=0, keepdim=True)
        if self.count_data_type in ['long']:
            states_sum = states_sum.long()
        elif self.count_data_type in ['double']:
            states_sum = states_sum.double()
        else:
            raise NotImplemented
        self.count += states_sum

    def compute_bouns(self, states, keepdim):
        return (states * self.get_bouns_map().expand(states.size())).sum(dim=1, keepdim=keepdim)

    def get_bouns(self, states, keepdim, is_stack):
        bouns = self.compute_bouns(states, keepdim)
        if is_stack:
            self.update_count(states)
        return bouns

    def store(self, log_dir):
        to_save = {}
        to_save['count'] = self.count.cpu().numpy()

        try:
            np.save(
                '{}.npy'.format(log_dir),
                to_save,
            )
            print('{}: Store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Store Failed.'.format(self.__class__.__name__))

    def restore(self, log_dir):
        try:
            loaded = np.load('{}.npy'.format(log_dir))
            self.count = torch.from_numpy(loaded[()]['count']).cuda()
            print('{}: Restore Successed, self.count: {}.'.format(
                self.__class__.__name__, self.count))
        except Exception as e:
            print('{}: Restore Failed.'.format(self.__class__.__name__))

        self.check_data_type()


def to_batch_version(x, batch_size):
    return x.repeat(batch_size, *(tuple([1] * len(x.size()[1:]))))


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
            mask = torch.ones(self.args.num_grid,
                              self.args.num_grid).float().cuda()
        assert mask.size()[0] == args.num_grid and mask.size()[
            1] == args.num_grid

        '''add batch dim'''
        mask = mask.unsqueeze(0)
        '''flatten'''
        mask = mask.view(mask.size()[0], -1)

        self.mask_batch = to_batch_version(mask, args.num_processes)

    def mask(self, x):
        return x * self.mask_batch

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
                tag=name,
                simple_value=summay_dic[name],
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
        torch.save(save_model, os.path.join(args.log_dir, 'learner' + ".pt"))
        np.save(
            os.path.join(args.log_dir, "j.npy"),
            np.array([j]),
        )
        print('# INFO: store learner ok.')
    except Exception as e:
        print('# WARNING: store learner failed: {}.'.format(e))


def restore_learner(args, actor_critic, envs, j):
    try:
        actor_critic, ob_rms = torch.load(
            os.path.join(args.log_dir, 'learner' + ".pt"))
        if args.cuda:
            actor_critic = actor_critic.cuda()
        envs.ob_rms = ob_rms
        j = np.load(
            os.path.join(args.log_dir, "j.npy"),
        )[0]
        print('# INFO: restore learner ok.')
    except Exception as e:
        print('# WARNING: restore learner failed: {}.'.format(e))

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
