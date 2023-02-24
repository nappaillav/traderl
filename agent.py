from dqn import DQN
from replaybuffer import Replaybuffer
import random
import torch 
import torch.nn as nn
import math 
from torch import optim
from replaybuffer import transition

class Agent:
    def __init__(self, obs_dim, act_dim, capacity=10000, device='cpu'):
        
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        self.inventory = []
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.device = device
        self.policy_net = DQN(obs_dim, act_dim).to(device)
        self.target_net = DQN(obs_dim, act_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = Replaybuffer(capacity)
        self.criterion = nn.SmoothL1Loss() # MSELoss
        self.steps_done = 0

    def act(self, obs):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END)* \
                                        math.exp(-1*self.steps_done/self.EPS_DECAY)

        # self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].view(1,1)
        else:
            self.steps_done += 1
            return torch.randint(0, self.act_dim - 1, (1,1)).to(self.device)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = transition(*zip(*transitions))

        # TODO [change the replay buffer]
        # Alternative: replay buffer could return 
        # obs : ndarray (N X obs_dim)
        # act : ndarray (N X act_dim)
        # reward : 
        # next_state:
        # Done : 

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        qval = reward_batch + self.GAMMA * next_state_values
        # print(qval.shape)
        # print(state_action_values.shape)
        # print('----')
        loss = self.criterion(qval.unsqueeze(1), state_action_values) 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 5) # variable

        self.optimizer.step()
    
    def target_update(self):
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.TAU + \
                (1-self.TAU) * target_state_dict[key]
        self.target_net.load_state_dict(target_state_dict)