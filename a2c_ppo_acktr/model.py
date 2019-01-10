import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init

class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x*self.scale

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DeFlatten(nn.Module):
    def __init__(self, shape):
        super(DeFlatten, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.zero_loss = torch.FloatTensor([0.0]).cuda()[0]

    def store(self, save_path):
        try:
            from shutil import copyfile
            copyfile(save_path, save_path.replace('.pth','_old.pth'))
            print('{}: Reserve old model successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Reserve old model failed.'.format(self.__class__.__name__))

        try:
            torch.save(self.state_dict(), save_path)
            print('{}: Store successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Store failed.'.format(self.__class__.__name__))

    def restore(self, save_path):
        try:
            self.load_state_dict(torch.load(save_path))
            print('{}: Restore Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Restore Failed.'.format(self.__class__.__name__))


class GridModel(BaseModel):
    def __init__(self, num_grid, num_stack, action_space_n, obs_size):
        super(GridModel, self).__init__()
        self.num_grid = num_grid
        self.num_stack = num_stack
        self.action_space_n = action_space_n
        self.obs_size = obs_size
        self.size_grid = int(self.obs_size/self.num_grid)

        self.linear_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.relu_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.leakrelu_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('leaky_relu'))

        self.tanh_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('tanh'))

        self.coordinates = {}
        self.relative_coordinates = {}

        self.coordinates_size = int((self.num_grid)**2)
        self.relative_coordinates_size = int((self.num_grid*2-1)**2)

        self.grid_mask = np.zeros((self.obs_size,self.obs_size), dtype=np.float)

        for i in range(self.num_grid):
            self.grid_mask[i*self.size_grid,:] = 1.0

        for j in range(self.num_grid):
            self.grid_mask[:,j*self.size_grid] = 1.0

    def draw_grid_on_img(self,img):
        '''(xx,xx) 0-255 >> (xx,xx) 0-255'''
        img = (img.astype(np.float)*(1.0-self.grid_mask)+self.grid_mask*255.0).astype(np.uint8)
        return img

    def slice_grid(self, states, i, j):
        '''
        (batch_size, feature, height, width) -> (batch_size, feature, size_grid, size_grid)
        '''
        return states[:,:,i*self.size_grid:(i+1)*self.size_grid,j*self.size_grid:(j+1)*self.size_grid]

    def deslice_grid(self, states, i, j, desliced_states):
        '''
        (batch_size, feature, size_grid, size_grid) -> (batch_size, feature, height, width)
        '''
        desliced_states[:,:,i*self.size_grid:(i+1)*self.size_grid,j*self.size_grid:(j+1)*self.size_grid] = states

    def grid_states(self, states, is_flatten=True):
        '''
        (batch_size, self.obs_size, self.obs_size) -> (batch_size, each_grid, num_channels*self.size_grid**2)
        '''
        grided_states = []
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                temp = self.slice_grid(states,i,j)
                if is_flatten:
                    temp = flatten(temp)
                grided_states += [temp.unsqueeze(1)]
        grided_states = torch.cat(grided_states,1)

        return grided_states

    def flatten_cell(self,x):
        return x.view(x.size()[0], -1)

    def deflatten_cell(self, x, num_channels):
        return x.view(x.size()[0], num_channels, self.size_grid, self.size_grid)

    def degrid_states(self, states):
        '''
        (batch_size, each_grid, num_channels*self.size_grid**2) -> (batch_size, self.obs_size, self.obs_size)
        '''
        num_channels = int(states.size()[2]/(self.size_grid**2))
        degrided_states = torch.FloatTensor(states.size()[0],num_channels,self.obs_size,self.obs_size).cuda()
        each_grid_i = 0
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                self.deslice_grid(self.deflatten_cell(states[:,each_grid_i],num_channels), i, j, degrided_states)
                each_grid_i += 1

        return degrided_states

    def get_absolute_coordinates(self,states):
        '''
        (batch_size, ...) -> (batch_size, each_grid, self.coordinates)
        '''
        batch_size = states.size()[0]
        if batch_size not in self.coordinates.keys():
            coordinates = []
            for i in range(self.num_grid):
                for j in range(self.num_grid):

                    temp = torch.zeros(batch_size,self.coordinates_size).cuda()

                    temp[:,(i*self.num_grid+j)].fill_(1.0)

                    coordinates += [temp.unsqueeze(1)]

            coordinates = torch.cat(coordinates,1)
            self.coordinates[batch_size] = coordinates
        return self.coordinates[batch_size]

    def get_relative_coordinates(self,states,base_coordinates):
        '''
        (batch_size, ...) -> (batch_size, each_grid, self.relative_coordinates)
        '''
        batch_size = states.size()[0]
        if batch_size not in self.relative_coordinates.keys():
            coordinates = []
            for i in range(self.num_grid):
                for j in range(self.num_grid):

                    temp = torch.zeros(batch_size,self.relative_coordinates_size).cuda()

                    for b in range(base_coordinates.size()[0]):
                        base_coordinate_tamp = base_coordinates[b].nonzero()[0,0].item()

                        i_base = base_coordinate_tamp // self.num_grid # 0-3
                        j_base = base_coordinate_tamp % self.num_grid # 0-3

                        relative_i = (i-i_base+(self.num_grid-1)) # 0-6
                        relative_j = (j-j_base+(self.num_grid-1)) # 0-6

                        posi = int(relative_i*(self.num_grid*2-1)+relative_j)

                        temp[b,posi].fill_(1.0)

                    coordinates += [temp.unsqueeze(1)]
            coordinates = torch.cat(coordinates,1)
            self.relative_coordinates[batch_size] = coordinates
        return self.relative_coordinates[batch_size]

    def put_grid_axis_to_batch_axis(self, x):
        '''
            (batch_size, each_grid, ...) -> (batch_size * each_grid, ...)
        '''
        return x.view(x.size()[0]*x.size()[1],*x.size()[2:])

    def extract_grid_axis_from_batch_axis(self, x):
        '''
            (batch_size * each_grid, ...) -> (batch_size, each_grid, ...)
        '''
        return x.view(int(x.size()[0]/(self.num_grid**2)),int(self.num_grid**2),*x.size()[1:])

    def repeat_on_each_grid_axis(self, x, repeat_times):
        '''
            (batch_size, ...) -> (batch_size, each_grid, ...)
        '''
        return x.unsqueeze(1).repeat(1,repeat_times,*(tuple([1]*len(x.size()[1:]))))

    def integrate_phi_gamma(self, phi, gamma):
        '''
            phi (batch_size, from_each_grid, ...) + gamma (batch_size, from_each_grid) -> (batch_size, ...)
        '''
        return (phi*gamma.unsqueeze(2).expand(-1,-1,phi.size()[2])).sum(
            dim = 1,
            keepdim = False,
        )

    def get_gamma_entropy_loss(self, gamma):
        '''mean over batch'''
        return (gamma*gamma.log()).mean()

    def to_mask_img(self, x):
        x = x.unsqueeze(2).expand(-1,-1,self.size_grid)
        x = x.contiguous().view(x.size()[0], self.num_grid, -1)
        x = torch.cat([x]*self.size_grid,dim=2).view(x.size()[0],self.size_grid*self.num_grid,self.size_grid*self.num_grid)
        return x.unsqueeze(1)


class DirectControlModel(GridModel):
    def __init__(self, num_grid, num_stack, action_space_n, obs_size, loss_action_each=False, loss_action_entropy=False):
        super(DirectControlModel, self).__init__(num_grid, num_stack, action_space_n, obs_size)

        self.conved_shape = (16,6,6)
        self.conved_size = self.conved_shape[0]*self.conved_shape[1]*self.conved_shape[2]
        self.linear_size = 64
        self.loss_action_each = loss_action_each
        self.loss_action_entropy = loss_action_entropy

        self.Phi_conv = nn.Sequential(

            Scale(1.0/255.0),

            # (21-5)/2+1 = 9
            self.leakrelu_init_(nn.Conv2d(2, 8, 5, stride=2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),

            # (9-4)/1+1 = 6
            self.leakrelu_init_(nn.Conv2d(8, 16, 4, stride=1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            Flatten(),

            self.leakrelu_init_(nn.Linear(self.conved_size, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU(inplace=True),
        )

        self.Phi_coordinate_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.coordinates_size, self.linear_size)),
            #
            #
        )

        self.Phi_output = nn.Sequential(
            self.leakrelu_init_(nn.Linear(self.linear_size, int(self.linear_size/2))),
            nn.BatchNorm1d(int(self.linear_size/2)),
            nn.LeakyReLU(inplace=True),

            self.linear_init_(nn.Linear(int(self.linear_size/2), self.action_space_n)),
        )

        self.Gamma_conv = nn.Sequential(

            Scale(1.0/255.0),

            # (21-5)/2+1 = 9
            self.leakrelu_init_(nn.Conv2d(1, 8, 5, stride=2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),

            # (9-4)/1+1 = 6
            self.leakrelu_init_(nn.Conv2d(8, 16, 4, stride=1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            Flatten(),

            self.leakrelu_init_(nn.Linear(self.conved_size, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU(inplace=True),
        )

        self.Gamma_coordinate_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.coordinates_size, self.linear_size)),
            #
            #
        )

        self.Gamma_output = nn.Sequential(
            self.leakrelu_init_(nn.Linear(self.linear_size, int(self.linear_size/2))),
            nn.BatchNorm1d(int(self.linear_size/2)),
            nn.LeakyReLU(inplace=True),

            self.linear_init_(nn.Linear(int(self.linear_size/2), 1)),
            #
            #
        )

        self.NLLLoss = nn.NLLLoss(reduction='mean')

    def get_gamma(self, now_states, coordinates):

        '''(batch_size*from_each_grid, ...) -> (batch_size*from_each_grid, 1)'''
        gamma_bar = self.Gamma_output(
            self.Gamma_conv(now_states)
            *
            self.Gamma_coordinate_linear(coordinates)
        )

        '''(batch_size*from_each_grid, 1) -> (batch_size, from_each_grid, 1)'''
        gamma_bar = self.extract_grid_axis_from_batch_axis(gamma_bar)

        '''(batch_size, from_each_grid, 1) -> (batch_size, from_each_grid)'''
        gamma = F.softmax(gamma_bar.squeeze(2), dim=1)

        return gamma

    def get_phi(self, now_last_states, now_states, coordinates):

        '''(batch_size*from_each_grid, ...) -> (batch_size*from_each_grid, self.action_space_n)'''
        phi = self.Phi_output(
            self.Phi_conv(
                torch.cat(
                    [now_last_states,now_states],
                    dim = 1,
                )
            )
            *
            self.Phi_coordinate_linear(coordinates)
        )

        '''(batch_size*from_each_grid, self.action_space_n) -> (batch_size, from_each_grid, self.action_space_n)'''
        phi = self.extract_grid_axis_from_batch_axis(phi)

        return phi

    def get_coordinates_now_states(self, now_states):

        '''(batch_size, ...) -> (batch_size*from_each_grid, ...)'''
        coordinates = self.put_grid_axis_to_batch_axis(self.get_absolute_coordinates(now_states))
        now_states  = self.put_grid_axis_to_batch_axis(self.grid_states(now_states,is_flatten=False))

        return coordinates, now_states

    def get_now_last_states(self, now_states, last_states):

        '''(batch_size, ...) -> (batch_size*from_each_grid, ...)'''
        now_last_states  = self.put_grid_axis_to_batch_axis(self.grid_states(now_states-last_states,is_flatten=False))

        return now_last_states

    def get_mask(self, now_states):

        '''(batch_size, ...) -> (batch_size*from_each_grid, ...)'''
        coordinates, now_states = self.get_coordinates_now_states(
            now_states = now_states,
        )

        '''(batch_size*from_each_grid, ...) -> (batch_size, from_each_grid)'''
        gamma = self.get_gamma(
            now_states = now_states,
            coordinates = coordinates,
        )

        return gamma

    def forward(self, last_states, now_states, action_lables):

        '''(batch_size, ...) -> (batch_size*from_each_grid, ...)'''
        now_last_states = self.get_now_last_states(
            now_states = now_states,
            last_states = last_states,
        )

        '''(batch_size, ...) -> (batch_size*from_each_grid, ...)'''
        coordinates, now_states = self.get_coordinates_now_states(
            now_states = now_states,
        )

        '''(batch_size*from_each_grid, ...) -> (batch_size, from_each_grid)'''
        gamma = self.get_gamma(
            now_states = now_states,
            coordinates = coordinates,
        )

        '''(batch_size*from_each_grid, ...) -> (batch_size, from_each_grid, self.action_space_n)'''
        phi = self.get_phi(
            now_last_states = now_last_states,
            now_states = now_states,
            coordinates = coordinates,
        )

        '''(batch_size, from_each_grid, self.action_space_n) and (batch_size, from_each_grid) -> (batch_size, self.action_space_n)'''
        predicted_action_log_probs = self.integrate_phi_gamma(phi,gamma)
        predicted_action_log_probs = F.log_softmax(predicted_action_log_probs,1)

        '''(batch_size, self.action_space_n) -> mean over batch_size'''
        loss_action = self.NLLLoss(predicted_action_log_probs, action_lables)

        if self.loss_action_each:
            '''(batch_size, from_each_grid, self.action_space_n)'''
            predicted_action_log_probs_each = F.log_softmax(phi,dim=2)
            '''(batch_size, self.action_space_n) -> (batch_size, from_each_grid, self.action_space_n)'''
            action_lables_each = self.repeat_on_each_grid_axis(action_lables, int(self.num_grid**2))

            '''(batch_size, from_each_grid, self.action_space_n) -> (batch_size*from_each_grid, self.action_space_n)'''
            predicted_action_log_probs_each = self.put_grid_axis_to_batch_axis(predicted_action_log_probs_each)
            action_lables_each = self.put_grid_axis_to_batch_axis(action_lables_each)

            '''(batch_size*from_each_grid, self.action_space_n) -> mean over batch_size*from_each_grid'''
            loss_action_each = self.NLLLoss(
                predicted_action_log_probs_each,
                action_lables_each,
            )
        else:
            loss_action_each = self.zero_loss

        if self.loss_action_entropy:
            '''(batch_size, from_each_grid) -> (batch_size*from_each_grid)'''
            gamma = self.put_grid_axis_to_batch_axis(gamma)
            '''(batch_size*from_each_grid) -> mean over batch_size'''
            loss_ent_direct = self.get_gamma_entropy_loss(gamma)*(self.num_grid**2)
        else:
            loss_ent_direct = self.zero_loss

        return loss_action, loss_action_each, loss_ent_direct

class LatentControlModel(GridModel):
    def __init__(self, num_grid, num_stack, action_space_n, obs_size, random_noise_frame=True, epsilon=1.0, C_keepsum=False, loss_transition_each=False, loss_transition_entropy=False):
        super(LatentControlModel, self).__init__(num_grid, num_stack, action_space_n, obs_size)

        self.random_noise_frame = random_noise_frame
        self.epsilon = epsilon
        self.C_keepsum = C_keepsum
        self.loss_transition_each = loss_transition_each
        self.loss_transition_entropy = loss_transition_entropy

        self.conved_shape = (32,6,6)
        self.conved_size = self.conved_shape[0]*self.conved_shape[1]*self.conved_shape[2]

        self.linear_size = 1024

        self.Phi_conv = nn.Sequential(

            Scale(1.0/255.0),

            # (21-5)/2+1 = 9
            self.leakrelu_init_(nn.Conv2d(self.num_stack, 16, 5, stride=2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            # (9-4)/1+1 = 6
            self.leakrelu_init_(nn.Conv2d(16, 32, 4, stride=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            Flatten(),

            self.leakrelu_init_(nn.Linear(self.conved_size, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU(inplace=True),
        )

        self.Phi_coordinate_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.relative_coordinates_size, self.linear_size)),
            #
            #
        )

        self.Phi_action_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.action_space_n, self.linear_size)),
            #
            #
        )

        # (L_in−1)×stride+kernel_size
        self.Phi_deconv = nn.Sequential(
            self.leakrelu_init_(nn.Linear(self.linear_size, self.conved_size)),
            nn.BatchNorm1d(self.conved_size),
            nn.LeakyReLU(inplace=True),

            DeFlatten(self.conved_shape),

            self.leakrelu_init_(nn.ConvTranspose2d(32, 16, 4, stride=1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            self.tanh_init_(nn.ConvTranspose2d(16, 1, 5, stride=2)),
            #
            nn.Sigmoid(),

            Flatten(),

            Scale(255.0),
        )

        self.Gamma_conv = nn.Sequential(

            Scale(1.0/255.0),

            # (21-5)/2+1 = 9
            self.leakrelu_init_(nn.Conv2d(self.num_stack, 16, 5, stride=2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            # (9-4)/1+1 = 6
            self.leakrelu_init_(nn.Conv2d(16, 32, 4, stride=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            Flatten(),

            self.leakrelu_init_(nn.Linear(self.conved_size, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU(inplace=True),
        )

        self.Gamma_coordinate_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.relative_coordinates_size, self.linear_size)),
            #
            #
        )

        self.Gamma_output = nn.Sequential(
            self.tanh_init_(nn.Linear(self.linear_size, int(self.linear_size/2))),
            nn.BatchNorm1d(int(self.linear_size/2)),
            nn.Tanh(),

            self.linear_init_(nn.Linear(int(self.linear_size/2), 1)),
            #
            #
        )

        if self.random_noise_frame:
            self.noise_masks = {}

    def randomize_noise_masks(self, batch_size):
        if batch_size not in self.noise_masks.keys():
            self.noise_masks[batch_size] = torch.zeros(batch_size,1,self.obs_size,self.obs_size).cuda()

        return self.noise_masks[batch_size].uniform_(-1.0,1.0).sign_()*self.epsilon

    def add_noise_masks(self,x):
        x_return = x.clone()
        x_return[:,-1:] = x_return[:,-1:] + self.noise_masks[x.size()[0]]
        return x_return

    def get_gamma(self, last_states, coordinates, onehot_actions):

        '''(batch_size*to_each_grid*from_each_grid, ...) - > (batch_size*to_each_grid*from_each_grid, 1)'''
        gamma_bar = self.Gamma_output(
            self.Gamma_conv(last_states)
            *
            self.Gamma_coordinate_linear(coordinates)
        )

        '''(batch_size*to_each_grid*from_each_grid, 1)  -> (batch_size*to_each_grid, from_each_grid, 1)'''
        gamma_bar = self.extract_grid_axis_from_batch_axis(gamma_bar)

        '''(batch_size*to_each_grid, from_each_grid, 1) -> (batch_size*to_each_grid, from_each_grid)'''
        gamma     = F.softmax(gamma_bar.squeeze(2), dim=1)

        return gamma

    def get_phi(self, last_states, coordinates, onehot_actions):

        '''(batch_size*to_each_grid*from_each_grid, ...) - > (batch_size*to_each_grid*from_each_grid, ...)'''
        phi = self.Phi_deconv(
            self.Phi_conv(last_states)
            *
            self.Phi_coordinate_linear(coordinates)
            *
            self.Phi_action_linear(onehot_actions)
        )

        '''(batch_size*to_each_grid*from_each_grid, ...) - > (batch_size*to_each_grid, from_each_grid, ...)'''
        phi = self.extract_grid_axis_from_batch_axis(phi)

        return phi

    def get_coordinates_last_states_now_states_onehot_actions_now_states_target(self, now_states, last_states, onehot_actions):

        '''(batch_size, ...) -> (batch_size, to_each_grid, ...)'''
        base_coordinates = self.get_absolute_coordinates(now_states)
        now_states       = self.grid_states(now_states, is_flatten = False)
        last_states      = self.repeat_on_each_grid_axis(last_states   , int(self.num_grid**2))
        onehot_actions   = self.repeat_on_each_grid_axis(onehot_actions, int(self.num_grid**2))

        '''(batch_size, to_each_grid, ...) -> (batch_size*to_each_grid, ...)'''
        base_coordinates = self.put_grid_axis_to_batch_axis(base_coordinates    )
        now_states       = self.put_grid_axis_to_batch_axis(now_states          )
        last_states      = self.put_grid_axis_to_batch_axis(last_states         )
        onehot_actions   = self.put_grid_axis_to_batch_axis(onehot_actions      )
        now_states_target = self.flatten_cell(now_states)

        '''(batch_size*to_each_grid, ...) -> (batch_size*to_each_grid, from_each_grid, ...)'''
        relative_coordinates = self.get_relative_coordinates(last_states, base_coordinates)
        now_states           = self.repeat_on_each_grid_axis(now_states    , int(self.num_grid**2))
        last_states          = self.grid_states(last_states, is_flatten=False)
        onehot_actions       = self.repeat_on_each_grid_axis(onehot_actions, int(self.num_grid**2))


        '''(batch_size*to_each_grid, from_each_grid, ...) -> (batch_size*to_each_grid*from_each_grid, ...)'''
        relative_coordinates = self.put_grid_axis_to_batch_axis(relative_coordinates)
        now_states           = self.put_grid_axis_to_batch_axis(now_states)
        last_states          = self.put_grid_axis_to_batch_axis(last_states)
        onehot_actions       = self.put_grid_axis_to_batch_axis(onehot_actions)

        return relative_coordinates, now_states, last_states, onehot_actions, now_states_target

    def update_C(self, C, last_states, now_states, onehot_actions):

        batch_size = last_states.size()[0]

        if self.random_noise_frame:
            self.randomize_noise_masks(batch_size)
            now_states = self.add_noise_masks(now_states)
            last_states = self.add_noise_masks(last_states)

        if self.C_keepsum:
            '''to one'''
            C_sum = C.sum(dim=1,keepdim=True).expand(C.size())
            C = F.softmax(C,dim=1)

        self.eval()

        '''(batch_size, ...) -> (batch_size*to_each_grid*from_each_grid, ...)'''
        relative_coordinates, now_states, last_states, onehot_actions, _ = self.get_coordinates_last_states_now_states_onehot_actions_now_states_target(
            now_states = now_states,
            last_states = last_states,
            onehot_actions = onehot_actions,
        )

        '''(batch_size*to_each_grid*from_each_grid, ...)  -> (batch_size*to_each_grid, from_each_grid)'''
        gamma = self.get_gamma(
            last_states = last_states,
            coordinates = relative_coordinates,
            onehot_actions = onehot_actions,
        )

        '''(batch_size*to_each_grid, from_each_grid) -> (batch_size, to_each_grid, from_each_grid)'''
        gamma = self.extract_grid_axis_from_batch_axis(gamma)

        '''(batch_size, from_each_grid) -> (batch_size, to_each_grid, from_each_grid)'''
        C = self.repeat_on_each_grid_axis(C, int(self.num_grid**2))

        '''(batch_size, to_each_grid, from_each_grid) -> (batch_size, to_each_grid)'''
        C = (C * gamma).sum(dim=2,keepdim=False)

        if self.C_keepsum:
            '''back to C_sum'''
            C = F.softmax(C,dim=1)
            C = C*C_sum

        return C

    def get_predicted_now_states(self, last_states, now_states, onehot_actions):

        '''(batch_size, ...) -> (batch_size*to_each_grid*from_each_grid, ...)'''
        relative_coordinates, now_states, last_states, onehot_actions, now_states_target = self.get_coordinates_last_states_now_states_onehot_actions_now_states_target(
            now_states = now_states,
            last_states = last_states,
            onehot_actions = onehot_actions,
        )

        '''(batch_size*to_each_grid*from_each_grid, ...) -> (batch_size*to_each_grid, from_each_grid, ...)'''
        phi = self.get_phi(
            last_states = last_states,
            coordinates = relative_coordinates,
            onehot_actions = onehot_actions,
        )

        '''(batch_size*to_each_grid*from_each_grid, ...)  -> (batch_size*to_each_grid, from_each_grid)'''
        gamma = self.get_gamma(
            last_states = last_states,
            coordinates = relative_coordinates,
            onehot_actions = onehot_actions,
        )

        '''(batch_size*to_each_grid, from_each_grid, ...) -> (batch_size*to_each_grid, ...)'''
        predicted_now_states = self.integrate_phi_gamma(phi, gamma)

        '''(batch_size*to_each_grid, from_each_grid) -> (batch_size*to_each_grid*from_each_grid)'''
        gamma = self.put_grid_axis_to_batch_axis(gamma)
        '''(batch_size*to_each_grid, from_each_grid, ...) -> (batch_size*to_each_grid*from_each_grid, ...)'''
        phi = self.put_grid_axis_to_batch_axis(phi)

        return predicted_now_states, now_states_target, gamma, phi

    def forward(self, last_states, now_states, onehot_actions):

        batch_size = last_states.size()[0]

        if self.random_noise_frame:
            self.randomize_noise_masks(batch_size)
            now_states = self.add_noise_masks(now_states)
            last_states = self.add_noise_masks(last_states)

        self.train()

        '''
            (batch_size, ...) ->
            predicted_now_states: (batch_size*to_each_grid               , ...)
            now_states_target:    (batch_size*to_each_grid               , ...)
            gamma:                (batch_size*to_each_grid*from_each_grid, ...)
            phi:                  (batch_size*to_each_grid*from_each_grid, ...)
        '''
        predicted_now_states, now_states_target, gamma, phi = self.get_predicted_now_states(
            last_states    = last_states,
            now_states     = now_states,
            onehot_actions = onehot_actions,
        )

        '''loss transition, (batch_size*to_each_grid, ...) -> mean over batch_size*to_each_grid'''
        loss_transition = F.mse_loss(
            input  = predicted_now_states,
            target = now_states_target,
            reduction='mean',
        )

        if self.loss_transition_each:
            '''(batch_size*to_each_grid, ...) -> (batch_size*to_each_grid, from_each_grid, ...)'''
            now_states_target    = self.repeat_on_each_grid_axis(now_states_target, int(self.num_grid**2))
            '''(batch_size*to_each_grid, from_each_grid, ...) -> (batch_size*to_each_grid*from_each_grid, ...)'''
            now_states_target    = self.put_grid_axis_to_batch_axis(now_states_target)

            '''(batch_size*to_each_grid*from_each_grid, ...) -> mean over batch_size*to_each_grid*from_each_grid '''
            loss_transition_each = F.mse_loss(
                input  = phi,
                target = now_states_target,
                reduction='mean',
            )
        else:
            loss_transition_each = self.zero_loss

        if self.loss_transition_entropy:
            '''(batch_size*to_each_grid*from_each_grid) -> mean over batch_size*to_each_grid'''
            loss_ent_latent = self.get_gamma_entropy_loss(gamma)*(self.num_grid**2)
        else:
            loss_ent_latent = self.zero_loss

        return loss_transition, loss_transition_each, loss_ent_latent
