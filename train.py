import numpy as np
import math
import torch
import pandas as pd 
from agent import Agent
from util import getState
# DATA

stock_name, window_size, episode_count = 'GSPC.csv', 32, 1000
data = pd.read_csv(stock_name)
# print(data.head())
market = list(data['Close'].values)
l_data = len(market)

# Get number of actions from gym action space
window_size, time_step = 32, 0
# Get the number of state observations
state = getState(market, time_step, window_size+1)
n_observations = len(state)
act_dim = 3
episode = 100
device = 'cpu'

agent = Agent(n_observations, act_dim)

for e in range(episode):
    print('Episode :{}'.format(e))
    obs = torch.tensor(getState(market, 0, window_size+1), dtype='float32', device=device).unsqueeze(0)
    total_profit = 0

    agent.inventory = []

    for t in range(l_data):
        action = agent.act(obs)
        
        if t == l_data:
            done = True
            next_state = None
            print('Total Profit : {}'.format(total_profit))
        else:
            done = False
            next_state = torch.tensor(getState(market, 0, window_size+1), dtype='float32', device=device).unsqueeze(0)
        
        reward = 0 # add slight negative reward when it holds a lot

        if action == 1:
            agent.inventory.append(data[t])
        
        elif action == 2 and len(agent.inventory()) > 0:
            bought_price = np.mean(agent.inventory())
            reward = max(data[t] -bought_price, 0)
            total_profit += data[t] - bought_price         
        done = True if t == l_data else False

        reward = torch.tensor([reward], device=device)
        
        agent.memory.push((obs, action, next_state, reward, done))

        obs = next_state
        
        if done:
            print('Total Profit : {}'.format(total_profit))
        
        agent.optimize_model()

        if t%2 == 0:
            agent.target_update()
