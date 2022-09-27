import numpy as np

from true_model import create_true_model

true_model = create_true_model()
true_params = [5, 2.4]
import pdb;pdb.set_trace()
true_model.set_local_optimization_params(true_params)
print(true_model)

n=25
x = np.random.normal(0, 10,size=n).reshape((-1,1))
y = true_model.evaluate_equation_at(x) + np.random.normal(0, 0.1, x.shape)

data = np.hstack((x,y))

np.save('new_noisy_data', data)
