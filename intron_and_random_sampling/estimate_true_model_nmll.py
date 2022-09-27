import numpy as np

from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness_function import \
                                      BayesFitnessFunction

def estimate_nmll_of_true_model(model, data_string, n=20):

    data = np.load(data_string)
    x, y = data[:,0].reshape((-1,1)), data[:,1].reshape((-1,1))
    training_data = ExplicitTrainingData(x, y)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    bff = BayesFitnessFunction(local_opt_fitness,
                                num_particles=600,
                                mcmc_steps=20,
                                num_multistarts=4)

    nmll_estimates = [bff(model) for _ in range(n)]

    print(f'nmll estimates: {nmll_estimates}')
    print(f'NMLL of true model: {np.nanmean(nmll_estimates)}')

    return np.nanmean(nmll_estimates)
