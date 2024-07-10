import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

outfolderpath = Path.cwd() / 'hitting_sets_data'

universe_sizes = np.load(outfolderpath / 'universe_sizes.npy')
collection_sizes = np.load(outfolderpath / 'collection_sizes.npy')
hitting_set_probabilities = np.load(outfolderpath / 'hitting_set_probabilities.npy')
n_hittingsets = np.load(outfolderpath / 'n_hittingsets.npy')


sizes = np.unique(universe_sizes)
mean_probs = [np.mean(hitting_set_probabilities[universe_sizes == size]) for size in sizes]
std_probs = [np.std(hitting_set_probabilities[universe_sizes == size]) for size in sizes]

plt.errorbar(sizes, mean_probs, yerr=std_probs, label='Mean hitting set probs for random conflicts', color='g')
plt.plot(sizes, 1/2**sizes, color='b', label='Hitting set prob for max number of hitting sets')
plt.axhline(y=0.01, color='r', linestyle='--', label='Noise floor of 1%')

plt.xlabel('Universe Size')
plt.ylabel('Hitting Set Probability')
plt.yscale('log')
plt.legend()
plt.savefig(outfolderpath / 'hitting_set_probabilities.png')