import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--aux', type=str, default='',
                        help='some aux information you may want to record along with this run')

    '''Mega Agent'''
    parser.add_argument('--train-with-reward', type=str,
                        help='ex, in, ex_in' )
    parser.add_argument('--intrinsic-reward-type', type=str,
                        help='direct, latent' )
    parser.add_argument('--num-grid', type=int,
                        help='num grid of direct_control and indirect_control' )
    parser.add_argument('--G-skip', type=int,
                        help='num steps of per intrinsic reward generation' )
    parser.add_argument('--random-noise-frame', action='store_true',
                         help='if add a random noise to frame')
    parser.add_argument('--epsilon', type=float,
                         help='epsilon for random-noise-frame')
    parser.add_argument('--latent-control-intrinsic-reward-type', type=str,
                        help='M/G/delta_uG/__binary/NONE__relu/NONE__sum/hash_count_bouns/__clip_G/NONE' )
    parser.add_argument('--latent-control-discount', type=float,
                        help='G map of latent control discount' )
    parser.add_argument('--norm-rew', action='store_true', default=False,
                         help='if normalize the reward')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    import os
    args.log_dir = ''
    args.log_dir = os.path.join(args.log_dir, 'en-{}'.format(args.env_name))
    args.log_dir = os.path.join(args.log_dir, 'algo-{}'.format(args.algo))

    '''Mega Agent'''
    args.log_dir = os.path.join(args.log_dir, 'twr-{}'.format(args.train_with_reward))
    if 'in' in args.train_with_reward:
        args.log_dir = os.path.join(args.log_dir, 'irt-{}'.format(args.intrinsic_reward_type))
        args.log_dir = os.path.join(args.log_dir, 'ng-{}'.format(args.num_grid))
        args.log_dir = os.path.join(args.log_dir, 'gs-{}'.format(args.G_skip))
        args.log_dir = os.path.join(args.log_dir, 'nr-{}'.format(args.norm_rew))

        args.prioritized_replay_buffer_mode = 'random'
        args.log_dir = os.path.join(args.log_dir, 'prbm-{}'.format(args.prioritized_replay_buffer_mode))

        args.log_dir = os.path.join(args.log_dir, 'lcirt-{}'.format(args.latent_control_intrinsic_reward_type))

        args.control_model_mini_batch_size = args.num_processes
        args.train_control_model_every = args.num_steps
        args.new_sample_every_train_control_model = args.train_control_model_every*args.num_processes
        args.prioritized_replay_buffer_size = args.new_sample_every_train_control_model * 2
        args.num_interations_complete_a_push = int(args.new_sample_every_train_control_model/args.control_model_mini_batch_size)
        args.num_nobootup_iterations = args.num_interations_complete_a_push * 2

        '''num updates when
        1, agent acting randomly
        2, policy not updating'''
        args.num_bootup_updates = 600

        args.norm_rew = False
        if args.norm_rew:
            args.num_estimate_norm_rew_updates = 20
        else:
            args.num_estimate_norm_rew_updates = 0

        args.num_frames_no_norm_rew_updates        = (args.num_bootup_updates                                   )*args.num_processes*args.num_steps
        args.num_frames_random_act_no_agent_update = (args.num_bootup_updates+args.num_estimate_norm_rew_updates)*args.num_processes*args.num_steps

        args.log_dir = os.path.join(args.log_dir, 'rnf-{}'.format(args.random_noise_frame))

        if args.random_noise_frame:
            if args.env_name in ['PongNoFrameskip-v4']:
                if args.epsilon!= 1.0:
                    args.epsilon = 1.0*1.5930770635604858
                    print('# WARNING: Special case, args.epsilon={} for {}'.format(
                        args.epsilon,
                        args.env_name,
                    ))
            args.log_dir = os.path.join(args.log_dir, 'e-{}'.format(str(args.epsilon).replace('.','_')))

        args.log_dir = os.path.join(args.log_dir, 'lcirt-{}'.format(args.latent_control_intrinsic_reward_type))
        args.log_dir = os.path.join(args.log_dir, 'lcd-{}'.format(str(args.latent_control_discount).replace('.','_')))

        '''default settings'''
        args.is_remove_inter_episode_transitions = True
        args.is_lantent_control_action_conditional = True
        if args.prioritized_replay_buffer_mode=='priority' and (args.is_remove_inter_episode_transitions==False or args.is_lantent_control_action_conditional==False):
            input('# ACTION REQUIRED: args.prioritized_replay_buffer_mode = {}. This may not work since args.is_remove_inter_episode_transitions={} and args.is_lantent_control_action_conditional = {}'.format(
                args.prioritized_replay_buffer_mode,
                args.is_remove_inter_episode_transitions,
                args.is_lantent_control_action_conditional,
            ))

    args.log_dir = os.path.join(args.log_dir, 'a-{}'.format(args.aux))

    args.log_dir = args.log_dir.replace('/','--')
    args.log_dir = os.path.join('../results',args.log_dir)

    args.obs_size = 84
    try:
        args.crop_obs = {
            "PongNoFrameskip-v4": {
                'h': [14,args.obs_size  ],
                'w': [0 ,args.obs_size  ],
            },
            "BreakoutNoFrameskip-v4": {
                'h': [14,args.obs_size  ],
                'w': [0 ,args.obs_size  ],
            },
        }[args.env_name]
    except Exception as e:
        args.crop_obs = None
        print('# WARNING: args.crop_obs = None')

    try:
        # in_channels, out_channels, kernel_size, stride
        args.model_structure = {
            4: {
                'DirectControlModel': {
                    # 84/4 = 21
                    'conv_0': ('X', 8, 5, 2),
                    # (21-5)/2+1 = 9
                    'conv_1': (8, 16, 4, 1),
                    # (9-4)/1+1 = 6
                    'conved_shape': (16, 6, 6),
                    'linear_size': 64,
                },
                'LatentControlModel': {
                    # 84/4 = 21
                    'conv_0': ('X', 16, 5, 2),
                    # (21-5)/2+1 = 9
                    'conv_1': (16, 32, 4, 1),
                    # (9-4)/1+1 = 6
                    'conved_shape': (32, 6, 6),
                    'linear_size': 1024,
                    'deconv_1': (32, 16, 4, 1),
                    # (6−1)×1+4 = 9
                    'deconv_0': (16, 1, 5, 2),
                    # (9−1)×2+5 = 21
                },
            },
            6: {
                'DirectControlModel': {
                    # 84/6 = 14
                    'conv_0': ('X', 8, 4, 2),
                    # (14-4)/2+1 = 6·
                    'conv_1': (8, 16, 4, 1),
                    # (6-4)/1+1 = 3
                    'conved_shape': (16, 3, 3),
                    'linear_size': 64,
                },
                'LatentControlModel': {
                    # 84/6 = 14
                    'conv_0': ('X', 8, 4, 2),
                    # (14-4)/2+1 = 6
                    'conv_1': (8, 16, 4, 1),
                    # (6-4)/1+1 = 3
                    'conved_shape': (16, 3, 3),
                    'linear_size': 1024,
                    'deconv_1': (16, 8, 4, 1),
                    # (3−1)×1+4 = 6
                    'deconv_0': (8, 1, 4, 2),
                    # (6−1)×2+4 = 14
                },
            },
            7: {
                'DirectControlModel': {
                    # 84/7 = 12
                    'conv_0': ('X', 8, 4, 2),
                    # (12-4)/2+1 = 5·
                    'conved_shape': (8, 5, 5),
                    'linear_size': 64,
                },
                'LatentControlModel': {
                    # 84/7 = 12
                    'conv_0': ('X', 8, 4, 2),
                    # (12-4)/2+1 = 5
                    'conved_shape': (8, 5, 5),
                    'linear_size': 1024,
                    'deconv_0': (8, 1, 4, 2),
                    # (5−1)×2+4 = 12
                },
            },
        }[args.num_grid]
    except Exception as e:
        input('# ACTION REQUIRED: args.crop_obs = None')

    return args
