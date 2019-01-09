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
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot


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
    print('Dir empty, making new log dir :{}'.format(args.log_dir))
    os.makedirs(args.log_dir)
except Exception as e:
    if e.__class__.__name__ in ['FileExistsError']:
        print('Dir exsit, checking checkpoint...')
    else:
        raise e

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    final_reward = {}
    episode_reward = {}
    epoch_loss = {}

    if args.norm_rew:
        from a2c_ppo_acktr.utils import RunningMeanStd
        running_mean_std_extrinsic_reward = RunningMeanStd()
        running_mean_std_extrinsic_reward.restore('{}/running_mean_std_extrinsic_reward'.format(args.save_dir))

    if args.vis:
        import tensorflow as tf
        summary_writer = tf.summary.FileWriter(args.log_dir)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

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
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    for j in range(num_updates):

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

            # Obser reward and next obs
            obs, extrinsic_reward_clipped, done, infos = envs.step(action)

            if args.norm_rew:
                extrinsic_reward_normalized = running_mean_std_extrinsic_reward.stack_and_normalize(
                    extrinsic_reward_clipped
                )
            else:
                extrinsic_reward_normalized = extrinsic_reward_clipped

            for info in infos:
                if 'episode' in info.keys():
                    final_reward['ex_raw'] = info['episode']['r']

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, extrinsic_reward_normalized, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        if args.norm_rew:
            running_mean_std_extrinsic_reward.update_from_stack()

        rollouts.after_update()

        '''save for every interval-th episode or for the last epoch'''
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":

            '''policy and vec_normalize'''
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]
            torch.save(save_model, os.path.join(args.save_dir, 'policy_vec_normalize' + ".pt"))

            '''norm_rew'''
            if args.norm_rew:
                running_mean_std_extrinsic_reward.store(
                    '{}/running_mean_std_extrinsic_reward'.format(args.save_dir)
                )

        num_trained_frames = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0:
            end = time.time()
            print_str = "[{}/{}][F-{}][FPS {}]".format(
                j,num_updates,
                num_trained_frames,
                int(num_trained_frames / (end - start)),
            )
            try:
                print_str += '[R-{}]'.format(final_reward['ex_raw'])
            except Exception as e:
                pass
            print(print_str)

        if args.vis and j % args.vis_interval == 0:

            summary = tf.Summary()

            for episode_reward_type in final_reward.keys():
                summary.value.add(
                    tag = 'final_reward_{}'.format(
                        episode_reward_type,
                    ),
                    simple_value = final_reward[episode_reward_type],
                )

            for epoch_loss_type in epoch_loss.keys():
                summary.value.add(
                    tag = 'epoch_loss_{}'.format(
                        epoch_loss_type,
                    ),
                    simple_value = epoch_loss[epoch_loss_type],
                )

            summary_writer.add_summary(summary, num_trained_frames)
            summary_writer.flush()


if __name__ == "__main__":
    main()
