import numpy as np
import math
import pandas as pd 

def instantiate_class(arguments):
    from importlib import import_module

    d = dict(arguments)
    classname = d["classname"]
    del d["classname"]
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c(**d)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
  """
  data 
  t : 0
  n : window size
  """
  d = t - n + 1
  block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
  # print(block)
  res = []
  for i in range(n-1):
    res.append(sigmoid(block[i + 1] - block[i]))
  out = np.array(res, dtype='float32')
  return out 

def readTrader(datset:str, window_size:int):
  dataset = []
  rewards = []

  data = pd.read_csv(datset)
  market = list(data['Close'].values)
  timestep = len(market)
  actions = data['Action'].values
  for t in range(timestep-1):
    state = getState(market, t, window_size)
    action = actions[t]
    # Trader action is same as the policy, reward is 1 else 0
    # Buy:1, Hold:0, Sell:-1

    for act in [1, 0, -1]:
      state_act = np.append(state, act)
      reward = 1 if action == act else 0
      dataset.append(state_act)
      rewards.append(reward)


    return np.array(dataset), np.array(rewards)

# if __name__ == 'main':
# data = pd.read_csv('GSPC.csv')
# # print(data.head())
# market = list(data['Close'].values)
# sample = getState(market, len(market)-1, 5+1)
# print(sample)

