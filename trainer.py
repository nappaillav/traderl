import numpy as np
import math
import torch
import pandas as pd 
from agent import Agent
from util import getState
from reward_model import RewardModel

def Policytrainer(  stock:str,
                    window_size:int,
                    episode:int, 
                    obs_dim:int,
                    act_dim:int,
                    agent:agent.Agent,
                    use_HF:bool = False,
                    reward_model=None,
                    device:'str'='cpu',
                    ):
    if use_HF:
        # While using the reward model 
        model = RewardModel(obs_dim, act_dim).to(device)
        model.load_state_dict(torch.load(reward_model))

    data = pd.read_csv(stock)
    market = list(data['Close'].values)
    l_data = len(market)
    time_step = 0 # Start time step
    state = getState(market, time_step, window_size+1)

    for e in range(episode):
        print('Episode :{}'.format(e))
        obs = torch.tensor(getState(market, 0, window_size+1), dtype=torch.float32, device=device).unsqueeze(0)
        total_profit = 0

        agent.inventory = []

        for t in range(l_data):
            action = agent.act(obs)
            # print(t)
            if t == l_data-1:
                done = True
                next_state = None
                print('############### Total Profit : {} | Ends:{}'.format(total_profit, t))
            else:
                done = False
                next_state = torch.tensor(getState(market, t+1, window_size+1), dtype=torch.float32, device=device).unsqueeze(0)
            
            reward = 0 # add slight negative reward when it holds a lot

            if action == 1:
                # print('Buy at {}'.format(market[t]))
                agent.inventory.append(market[t])
            
            elif action == 2 and len(agent.inventory) > 0:
                # print('Sell at {}'.format(market[t]))
                # print(agent.inventory)
                bought_price = np.mean(agent.inventory)
                reward = max(market[t] - bought_price, 0)
                if t % 1000==0:
                print(round(reward,2))
                total_profit += market[t] - bought_price

                agent.inventory = []         
            done = True if t == l_data else False

            if use_HF:
                # add reward from reward model Learnt from Human feedback
                reward += getReward(model, obs, action)

            reward = torch.tensor([reward], device=device)
            
            agent.memory.push(obs, action, next_state, reward, done)

            obs = next_state
            
            if done:
                print('Total Profit : {}'.format(total_profit))
            
            agent.optimize_model()

            if t%2 == 0:
                agent.target_update()


def Policytester( stock:str,
            window_size:int,
            episode:int, 
            obs_dim:int,
            act_dim:int,
            agent:agent.Agent,
            device:'str'='cpu', 
    ):

    data = pd.read_csv(stock)
    market = list(data['Close'].values)
    l_data = len(market)
    time_step = 0 # Start time step
    state = getState(market, time_step, window_size+1)


    
    for t in range(l_data):
        action = agent.act(obs)

        if t == l_data-1:
            done = True
            next_state = None
            print('############### Total Profit : {} | Ends:{}'.format(total_profit, t))
        else:
            done = False
            next_state = torch.tensor(getState(market, t+1, window_size+1), dtype=torch.float32, device=device).unsqueeze(0)
        
        reward = 0 # add slight negative reward when it holds a lot

        if action == 1:
            # print('Buy at {}'.format(market[t]))
            agent.inventory.append(market[t])
        
        elif action == 2 and len(agent.inventory) > 0:
            # print('Sell at {}'.format(market[t]))
            # print(agent.inventory)
            bought_price = np.mean(agent.inventory)
            reward = max(market[t] - bought_price, 0)
            if t % 1000==0:
            print(round(reward,2))
            total_profit += market[t] - bought_price

            agent.inventory = []         
        done = True if t == l_data else False

        obs = next_state
        
        if done:
            print('Total Profit : {}'.format(total_profit)) 
        

def trainRewardModel(model, y_train, x_train,
          y_val, x_valid,
          learning_rate=0.001, epochs=300, print_out_interval=2
          stock_name='GSPC'):

    global criterion
    criterion = nn.MSELoss()  # we'll convert this to RMSE later
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    model.train()

    losses = []
    preds = []

    for i in range(epochs):
        i+=1 #Zero indexing trick to start the print out at epoch 1
        y_pred = model(x_train)
        preds.append(y_pred)
        loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
        losses.append(loss)
        
        if i%print_out_interval == 1:
            print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('='*80)
    print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
    print(f'Duration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    # Evaluate model
    with torch.no_grad():
        y_val = model(x_valid)
        loss = torch.sqrt(criterion(y_val, y_test))
    print(f'RMSE: {loss:.8f}')

    # Create empty list to store my results
    preds = []
    diffs = []
    actuals = []

    for i in range(len(categorical_valid)):
        diff = np.abs(y_val[i].item() - y_test[i].item())
        pred = y_val[i].item()
        actual = y_test[i].item()

        diffs.append(diff)
        preds.append(pred)
        actuals.append(actual)

    valid_results_dict = {
        'predictions': preds,
        'diffs': diffs,
        'actuals': actuals
    }

    # Save model
    torch.save(model.state_dict(), f'model_artifacts/{stock_name}_{epochs}.pt')
    # Return components to use later
    return model


def getReward(model, obs, act):
    model.eval()
    with torch.no_grad():
        y_val = model(obs, act.unsqueeze(0)).squeeze(0)
    return y_val.item()