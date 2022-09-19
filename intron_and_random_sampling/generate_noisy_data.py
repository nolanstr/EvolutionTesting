import numpy as np

n=25
x = np.random.normal(0, 10,size=n).reshape((-1,1))

y = np.sin(x * 5) + x**2 + 2.4*x + np.random.normal(0, 0.1, x.shape)

data = np.hstack((x,y))

np.save('noisy-data', data)
