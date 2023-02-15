import numpy as np
import math
import pandas as pd 


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
	print(block)
	res = []
	for i in range(n-1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res], dtype='float32')


# if __name__ == 'main':
data = pd.read_csv('GSPC.csv')
# print(data.head())
market = list(data['Close'].values)
sample = getState(market, len(market)-1, 5+1)
print(sample)

