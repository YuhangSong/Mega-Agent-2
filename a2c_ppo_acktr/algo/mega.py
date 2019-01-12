import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MEGA():
    def __init__(self,
                 direct_control_model,
                 latent_control_model,
                 num_iterations,
                 mini_batch_size,
                 latent_control_discount,
                 latent_control_intrinsic_reward_type,
                 empty_value):

        self.direct_control_model = direct_control_model
        self.latent_control_model = latent_control_model

        self.num_iterations = num_iterations
        self.mini_batch_size = mini_batch_size

        self.latent_control_discount = latent_control_discount
        self.latent_control_intrinsic_reward_type = latent_control_intrinsic_reward_type

        self.empty_value = empty_value

        self.optimizer_direct_control_model = optim.Adam(self.direct_control_model.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.optimizer_latent_control_model = optim.Adam(self.latent_control_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

        self.empty_intrinsic_reward = {}

    def update(self, prioritized_replay_buffer):
        epoch_loss = {}

        e = 0
        while True:

            if self.num_iterations>0:
                if e>=self.num_iterations:
                    break
            else:
                pass

            sampled, idxes = prioritized_replay_buffer.sample(
                batch_size = self.mini_batch_size,
            )

            '''
            update direct_control model
            '''
            '''reset grad'''
            self.optimizer_direct_control_model.zero_grad()
            '''forward'''
            self.direct_control_model.train()
            loss_action, loss_action_each, loss_ent_direct = self.direct_control_model(
                last_states   = sampled['states'][:,-1:],
                now_states    = sampled['next_states'],
                action_lables = sampled['actions'].nonzero()[:,1],
            )

            '''integrate losses'''
            loss_direct_control_model = loss_action + loss_action_each + 0.001*loss_ent_direct
            '''backward'''
            loss_direct_control_model.backward()
            '''optimize'''
            self.optimizer_direct_control_model.step()

            '''
            update latent_control model
            '''
            if self.latent_control_model is not None:
                '''reset grad'''
                self.optimizer_latent_control_model.zero_grad()
                '''forward'''
                self.latent_control_model.train()
                loss_transition, loss_transition_each, loss_ent_latent = self.latent_control_model(
                    last_states    = sampled['states'],
                    now_states     = sampled['skipped_next_states'],
                    onehot_actions = sampled['actions'],
                )

                if prioritized_replay_buffer.mode=='priority':
                    prioritized_replay_buffer.update_priorities(
                        idxes = idxes,
                        priorities = loss_transition.detach().cpu().numpy(),
                    )
                '''(batch_size) -> (1)'''
                loss_transition = loss_transition.mean(dim=0,keepdim=False)
                '''integrate losses'''
                loss_latent_control_model = loss_transition + loss_transition_each + 0.001*loss_ent_latent
                '''backward'''
                loss_latent_control_model.backward()
                '''optimize'''
                self.optimizer_latent_control_model.step()

            e += 1

        epoch_loss['loss_action'] = loss_action.item()
        epoch_loss['loss_action_each'] = loss_action_each.item()
        epoch_loss['loss_ent_direct'] = loss_ent_direct.item()
        epoch_loss['loss_direct_control_model'] = loss_direct_control_model.item()
        if self.latent_control_model is not None:
            epoch_loss['loss_transition'] = loss_transition.item()
            epoch_loss['loss_transition_each'] = loss_transition_each.item()
            epoch_loss['loss_ent_latent'] = loss_ent_latent.item()
            epoch_loss['loss_latent_control_model'] = loss_latent_control_model.item()

        return epoch_loss


    def generate_direct_and_latent_control_map(self, last_states, now_states, onehot_actions, G, masks, direct_control_mask):

        '''get M'''
        self.direct_control_model.eval()
        M = self.direct_control_model.get_mask(
            now_states = now_states,
        ).detach()
        M = direct_control_mask.mask(M)

        if self.latent_control_model is not None:
            '''update G'''
            if G is None:
                G = M
                new_G = M
                new_uG = M
            else:
                new_uG = G * masks
                self.latent_control_model.eval()
                new_uG = self.latent_control_model.update_C(
                    C = new_uG,
                    last_states    = last_states,
                    now_states     = now_states,
                    onehot_actions = onehot_actions,
                ).detach()

                if self.latent_control_intrinsic_reward_type.split('__')[5] in ['hold_uG']:
                    new_uG = torch.cat(
                        [G.unsqueeze(2), new_uG.unsqueeze(2)],
                        dim = 2,
                    ).max(dim=2, keepdim=False)[0]
                elif self.latent_control_intrinsic_reward_type.split('__')[5] in ['NONE']:
                    pass
                else:
                    raise NotImplemented

                new_G = (new_uG*self.latent_control_discount + M)

                if self.latent_control_intrinsic_reward_type.split('__')[4] in ['clip_G']:
                    new_G = new_G.clamp(min=0.0,max=1.0)
                elif self.latent_control_intrinsic_reward_type.split('__')[4] in ['NONE']:
                    pass
                else:
                    raise NotImplemented

            delta_uG = new_uG - G
            G = new_G

        else:
            G, delta_uG = None, None

        return M, G, delta_uG

    def generate_intrinsic_reward(self, M, G, delta_uG, masks, is_hash_count_bouns_stack, hash_count_bouns):

        if self.latent_control_intrinsic_reward_type.split('__')[0] in ['M']:
            map_to_use = M
        elif self.latent_control_intrinsic_reward_type.split('__')[0] in ['G']:
            map_to_use = G
        elif self.latent_control_intrinsic_reward_type.split('__')[0] in ['delta_uG']:
            '''delta_uG is stationary in a episode, so use directly'''
            map_to_use = delta_uG
            if self.latent_control_intrinsic_reward_type.split('__')[4] in ['NONE']:
                '''G is not clipped with in 0-1, so G is increasing in an
                episode, so normalize [may be] needed'''
                map_to_use = utils.torch_end_point_norm(map_to_use,dim=1)
        else:
            raise NotImplemented

        if self.latent_control_intrinsic_reward_type.split('__')[1] in ['binary']:
            map_to_use, _ = running_binary_norm.norm(
                map_to_use,
            )
        elif self.latent_control_intrinsic_reward_type.split('__')[1] in ['NONE']:
            pass
        else:
            raise NotImplemented

        if self.latent_control_intrinsic_reward_type.split('__')[2] in ['relu']:
            map_to_use = F.relu(map_to_use)
        elif self.latent_control_intrinsic_reward_type.split('__')[2] in ['NONE']:
            pass
        else:
            raise NotImplemented

        if self.latent_control_intrinsic_reward_type.split('__')[3] in ['hash_count_bouns']:
            intrinsic_reward = hash_count_bouns.get_bouns(
                states = map_to_use,
                keepdim = True,
                is_stack = is_hash_count_bouns_stack,
            )
        elif self.latent_control_intrinsic_reward_type.split('__')[3] in ['sum']:
            intrinsic_reward = map_to_use.sum(dim=1,keepdim=True)
        elif self.latent_control_intrinsic_reward_type.split('__')[3] in ['NONE']:
            pass
        else:
            raise NotImplemented

        intrinsic_reward *= masks

        return intrinsic_reward

    def generate_empty_intrinsic_reward(self, extrinsic_reward):
        if extrinsic_reward.size()[0] not in self.empty_intrinsic_reward.keys():
            self.empty_intrinsic_reward[extrinsic_reward.size()[0]] = extrinsic_reward.clone().fill_(self.empty_value)

        return self.empty_intrinsic_reward[extrinsic_reward.size()[0]]
