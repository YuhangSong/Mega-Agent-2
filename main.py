import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule, store_learner, restore_learner, DirectControlMask, VideoSummary, clear_print
from a2c_ppo_acktr.visualize import visdom_plot

import cv2
import numpy as np

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args.log_dir)
    print('# INFO: Dir empty, make new log dir :{}'.format(args.log_dir))
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)
    print('# INFO: Dir exists.')

eval_log_dir = args.log_dir + "/eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    summary_dic = {}

    ex_raw = []

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, args.crop_obs)
    args.obs_size = envs.observation_space.shape[1]
    args.size_grid = int(args.obs_size/args.num_grid)

    from a2c_ppo_acktr.utils import TF_Summary, VideoSummary, GridImg, ObsNorm
    tf_summary = TF_Summary(args)
    video_summary = VideoSummary(args)
    direct_control_mask = DirectControlMask(args=args)
    obs_norm = ObsNorm(
        envs = envs,
        num_processes = args.num_processes,
        nsteps = 10000,
    )
    obs_norm.restore(args.save_dir)
    obs_norm.store(args.save_dir)
    args.epsilon = args.epsilon/obs_norm.ob_std

    hash_count_bouns = None
    if args.latent_control_intrinsic_reward_type.split('__')[3] in ['hash_count_bouns']:
        from a2c_ppo_acktr.utils import IndexHashCountBouns
        hash_count_bouns = IndexHashCountBouns(
            k = int(args.num_grid),
            batch_size = args.num_processes,
            count_data_type = 'double',
            is_normalize = True,
        )
        hash_count_bouns.restore('{}/hash_count_bouns'.format(args.save_dir))

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    direct_control_model = None
    latent_control_model = None

    if 'in' in args.train_with_reward:

        '''replay_buffer'''
        from a2c_ppo_acktr.storage import PrioritizedReplayBuffer
        init_list = [
            'states',
            'actions',
            'next_states',
        ]
        if args.is_remove_inter_episode_transitions:
            init_list += ['next_state_masks']
        if args.G_skip>1:
            init_list += ['skipped_next_states']
        prioritized_replay_buffer = PrioritizedReplayBuffer(
            size=args.prioritized_replay_buffer_size,
            mode=args.prioritized_replay_buffer_mode,
            init_list = init_list,
            is_remove_inter_episode_transitions = args.is_remove_inter_episode_transitions,
        )

        '''direct_control_model'''
        from a2c_ppo_acktr.model import DirectControlModel
        direct_control_model = DirectControlModel(
            num_grid = args.num_grid,
            num_stack = envs.observation_space.shape[0],
            action_space_n = envs.action_space.n,
            obs_size = args.obs_size,
            model_structure = args.model_structure['DirectControlModel'],
        )
        direct_control_model.restore(args.save_dir+'/direct_control_model.pth')
        direct_control_model.to(device)

        '''latent_control_model'''
        if args.intrinsic_reward_type in ['latent']:
            from a2c_ppo_acktr.model import LatentControlModel
            latent_control_model = LatentControlModel(
                num_grid = args.num_grid,
                num_stack = envs.observation_space.shape[0],
                action_space_n = envs.action_space.n,
                obs_size = args.obs_size,
                random_noise_frame = args.random_noise_frame,
                epsilon = args.epsilon,
                ob_bound = obs_norm.ob_bound,
                model_structure = args.model_structure['LatentControlModel'],
                is_action_conditional = args.is_lantent_control_action_conditional,
            )
            latent_control_model.to(device)
            latent_control_model.restore(args.save_dir+'/latent_control_model.pth')

        brain = algo.MEGA(
             direct_control_model = direct_control_model,
             latent_control_model = latent_control_model,
             num_iterations = args.num_nobootup_iterations,
             mini_batch_size = args.control_model_mini_batch_size,
             latent_control_discount = args.latent_control_discount,
             latent_control_intrinsic_reward_type = args.latent_control_intrinsic_reward_type,
             empty_value = 0.0,
             G_skip = args.G_skip,
        )

    j = 0

    actor_critic, envs, j = restore_learner(args, actor_critic, envs, j)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    obs = obs_norm.obs_norm_batch(obs)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    num_trained_frames_start = j * args.num_processes * args.num_steps

    G = None

    while True:

        num_trained_frames = j * args.num_processes * args.num_steps

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                if ('in' in args.train_with_reward) and (num_trained_frames<args.num_frames_random_act_no_agent_update):
                    action.random_(0, envs.action_space.n)

            # Obser reward and next obs
            obs, extrinsic_reward, done, infos = envs.step(action)
            obs = obs_norm.obs_norm_batch(obs)

            for info in infos:
                if 'episode' in info.keys():
                    ex_raw.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done]).cuda()

            rollouts.insert_1(action)

            if args.train_with_reward in ['in', 'ex_in']:

                if step%args.G_skip==0:
                    M, G, delta_uG = brain.generate_direct_and_latent_control_map(
                        last_states = rollouts.obs[step],
                        now_states = obs[:,-1:],
                        onehot_actions = rollouts.onehot_actions[rollouts.step],
                        G = G,
                        masks = masks,
                        direct_control_mask = direct_control_mask,
                    )
                    intrinsic_reward = brain.generate_intrinsic_reward(
                        M = M,
                        G = G,
                        delta_uG = delta_uG,
                        masks = masks,
                        hash_count_bouns = hash_count_bouns,
                        is_hash_count_bouns_stack = (num_trained_frames > args.num_frames_random_act_no_agent_update),
                    )
                else:
                    '''M, G, delta_uG are just kept, but intrinsic_reward will be empty_value during the period'''
                    intrinsic_reward = brain.generate_empty_intrinsic_reward(extrinsic_reward)

                if args.train_with_reward in ['in']:
                    reward = intrinsic_reward
                elif args.train_with_reward in ['ex_in']:
                    input('# WARNING: if extrinsic_reward can be nagetive, the empty_intrinsic_reward may be buggy')
                    reward = extrinsic_reward + intrinsic_reward
                else:
                    raise NotImplemented

            elif args.train_with_reward in ['ex']:
                reward = extrinsic_reward
                M, G, delta_uG = None, None, None

            else:
                raise NotImplemented

            video_summary.stack(
                args = args,
                last_states = rollouts.obs[step][:1],
                now_states = obs[:1,-1:],
                onehot_actions = rollouts.onehot_actions[rollouts.step][:1],
                latent_control_model = latent_control_model,
                direct_control_mask = direct_control_mask,
                hash_count_bouns = hash_count_bouns,
                obs_norm = obs_norm,
                M = M,
                G = G,
                delta_uG = delta_uG,
                curves = {
                    'reward': reward[0,0].item(),
                },
                num_trained_frames = num_trained_frames,
            )

            rollouts.insert_2(obs, recurrent_hidden_states, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if ('in' in args.train_with_reward) and (num_trained_frames<args.num_frames_random_act_no_agent_update):
            agent_update_status_str = '[random_act_no_agent_update]'
        else:
            agent_update_status_str = '[agent_updating]'
            summary_dic.update(
                agent.update(rollouts)
            )

        '''train intrinsic reward models'''
        if 'in' in args.train_with_reward:

            total_steps = rollouts.obs.size()[0]

            pushed = {
                'states'                      : rollouts.put_process_axis_into_batch_axis(rollouts.obs           [0          :total_steps-args.G_skip         ]),
                'actions'                     : rollouts.put_process_axis_into_batch_axis(rollouts.onehot_actions[0          :total_steps-args.G_skip         ]),
                'next_states'                 : rollouts.put_process_axis_into_batch_axis(rollouts.obs           [1          :total_steps-args.G_skip+1 ,:,-1:]),
            }
            if args.is_remove_inter_episode_transitions:
                pushed['next_state_masks']    = rollouts.put_process_axis_into_batch_axis(rollouts.masks         [1          :total_steps-args.G_skip+1       ])
            if args.G_skip>1:
                pushed['skipped_next_states'] = rollouts.put_process_axis_into_batch_axis(rollouts.obs           [args.G_skip:total_steps               ,:,-1:])

            prioritized_replay_buffer.push(pushed)
            result_info = prioritized_replay_buffer.constrain_buffer_size()
            summary_dic.update(
                brain.update(prioritized_replay_buffer)
            )

        rollouts.after_update()

        '''save models and video summary'''
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            store_learner(args, actor_critic, envs, j)
            if 'in' in args.train_with_reward:
                direct_control_model.store(args.save_dir+'/direct_control_model.pth')
                if args.intrinsic_reward_type in ['latent']:
                    latent_control_model.store(args.save_dir+'/latent_control_model.pth')
            video_summary.summary_a_video(video_length=1000)
            obs_norm.store(args.save_dir)
            if args.latent_control_intrinsic_reward_type.split('__')[3] in ['hash_count_bouns']:
                hash_count_bouns.store('{}/hash_count_bouns'.format(args.save_dir))

        '''log info by print'''
        if j % args.log_interval == 0:
            end = time.time()
            print_str = "# INFO: [{}/{}][F-{}][FPS {}]".format(
                j,num_updates,
                num_trained_frames,
                int(((num_trained_frames+args.num_processes * args.num_steps)-num_trained_frames_start) / (end - start)),
            )
            try:
                print_str += '[R-{:.2f}]'.format(summary_dic['ex_raw'])
            except Exception as e:
                pass
            try:
                print_str += '[E_R-{}]'.format(summary_dic['eval_ex_raw'])
            except Exception as e:
                pass
            print_str += agent_update_status_str
            clear_print(print_str)

        '''vis curves'''
        if j % args.vis_interval == 0:

            if len(ex_raw)>0:
                summary_dic['ex_raw'] = np.mean(ex_raw)
                ex_raw = []

            tf_summary.summary_and_flush(
                summay_dic = summary_dic,
                step = num_trained_frames,
            )

        '''eval'''
        if (args.eval_interval is not None
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True, args.crop_obs)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            obs = obs_norm.obs_norm_batch(obs)
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                obs = obs_norm.obs_norm_batch(obs)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
                        clear_print("# INFO: [Evaluate {} episodes]".format(
                            len(eval_episode_rewards),
                        ))

            eval_envs.close()

            summary_dic['eval_ex_raw'] = np.mean(eval_episode_rewards)

        j += 1
        if j == num_updates:
            input('# ACRION REQUIRED: Run over, press Ctrl+C to release.')

if __name__ == "__main__":
    main()
