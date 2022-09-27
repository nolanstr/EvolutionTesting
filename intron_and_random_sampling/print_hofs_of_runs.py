import numpy as np
import glob 

from bingo.evolutionary_optimizers.parallel_archipelago import \
                        load_parallel_archipelago_from_file as lpaff


def get_hof_from_dir(DIR):
    file_names = glob.glob(DIR+'/*.pkl')
    pickles = [lpaff(file_name) for file_name in file_names]
    gens = [pickle.generational_age for pickle in pickles]
    oldest_pickle = pickles[np.argmax(gens)]
    return oldest_pickle.island.hall_of_fame


DIRS = ['control', 'test']
hofs = [get_hof_from_dir(DIR) for DIR in DIRS]

for DIR, hof in zip(DIRS, hofs):
    print(f'{DIR} runs:')
    for ind in hof:
        print(f'model form: {str(ind)}\nfitness: {ind.fitness}')
    print('\n\n')
