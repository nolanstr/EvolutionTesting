import numpy as np

from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.bayes_crowding import BayesCrowding
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness_function import \
                                      BayesFitnessFunction

POP_SIZE = 20
STACK_SIZE = 64
MAX_GEN = 500
FIT_THRESH = -np.inf
CHECK_FREQ = 10
MIN_GEN = 100


def execute_generational_steps():

    data = np.load('../data/simple_noisy_data.npy')
    x, y = data[:,0].reshape((-1,1)), data[:,1].reshape((-1,1))
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    component_generator.add_operator("exp")
    component_generator.add_operator("pow")
    component_generator.add_operator("sqrt")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    bff = BayesFitnessFunction(local_opt_fitness,
                                num_particles=600,
                                mcmc_steps=20,
                                num_multistarts=4)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(bff, redundant=True, multiprocess=8)

    selection_phase=BayesCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)

    opt_result = archipelago.evolve_until_convergence(max_generations=MAX_GEN,
                                                  fitness_threshold=FIT_THRESH,
                                        convergence_check_frequency=CHECK_FREQ,
                                              checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    execute_generational_steps()

