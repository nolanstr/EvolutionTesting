import numpy as np
import matplotlib.pyplot as plt
import glob

from bingo.evolutionary_optimizers.parallel_archipelago import \
        load_parallel_archipelago_from_file as lpaff

def get_all_pickles(DIR):

    filenames = glob.glob(DIR + '/*.pkl')
    pickles = [lpaff(filename) for filename in filenames]
    
    return pickles

def get_pickles_gens(pickles):
    
    return [pickle.generational_age for pickle in pickles]
    
def organize_pickles_and_gens(pickles, gens=False):
    
    if not gens:
        gens = get_pickles_gens(pickles)

    pickles = [pickles[i] for i in np.argsort(gens)]
    
    return pickles, np.sort(gens)

def get_pickle_fitness(pickle):

    population = pickle.island.population

    fitness = np.array([ind.fitness for ind in population])

    return fitness


def plot_dir_fitness_vs_gens(DIR, ax, color):
    
    pickles, gens = organize_pickles_and_gens(get_all_pickles(DIR))
    fitnesses = []

    for pickle in pickles:
        fitnesses.append(get_pickle_fitness(pickle))

    max_fitness = np.array([np.nanmax(fitness) for fitness in fitnesses])
    min_fitness = np.array([np.nanmin(fitness) for fitness in fitnesses])
    median_fitness = np.array([np.nanmedian(fitness) for fitness in fitnesses])

    ax.fill_between(gens, min_fitness, max_fitness, 
                                            color=color, alpha=0.5)
    ax.plot(gens, median_fitness, color=color, linestyle='--', label=f'{DIR} Fitness')
    ax.plot(gens, min_fitness, color=color)
    ax.plot(gens, max_fitness, color=color)


fig, ax = plt.subplots()
colors = [plt.cm.tab20c(i) for i in [0,4]]
DIRS = ['control', 'test']

for color, DIR in zip(colors, DIRS):
    plot_dir_fitness_vs_gens(DIR, ax, color)

true_nmll = -1.9273088728828391
ax.axhline(true_nmll, color='k', linestyle='--', label='true model NMLL estimate')
ax.set_xlabel('Generations')
ax.set_ylabel('Normalized Marginal Log-Likelihood')
plt.legend()

plt.show()

