import numpy as np

from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness_function import \
                                      BayesFitnessFunction
from true_model import create_true_model

true_model = create_true_model()
print(str(true_model))

data = np.load('new_noisy_data.npy')
x, y = data[:,0].reshape((-1,1)), data[:,1].reshape((-1,1))
training_data = ExplicitTrainingData(x, y)

fitness = ExplicitRegression(training_data=training_data)
local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
bff = BayesFitnessFunction(local_opt_fitness,
                            num_particles=600,
                            mcmc_steps=20,
                            num_multistarts=4)

n = 100
nmll_estimates = [bff(true_model) for _ in range(n)]
print(nmll_estimates)
print(f'NMLL of true model: {np.nanmean(nmll_estimates)}')
