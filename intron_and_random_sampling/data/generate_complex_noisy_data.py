import numpy as np

from true_models import * 

true_model = create_complex_true_model()
true_params = [5, 2.4]
true_model.set_local_optimization_params(true_params)
print(true_model)

n=25
x = np.random.normal(0, 10,size=n).reshape((-1,1))
y = true_model.evaluate_equation_at(x) + np.random.normal(0, 0.1, x.shape)

data = np.hstack((x,y))

np.save('complex_noisy_data', data)
