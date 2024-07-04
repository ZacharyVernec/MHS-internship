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

plt.errorbar(sizes, mean_probs, yerr=std_probs)
plt.axhline(y=0.01, color='r', linestyle='--')

plt.xlabel('Universe Size')
plt.ylabel('Hitting Set Probability')
plt.yscale('log')
plt.savefig(outfolderpath / 'hitting_set_probabilities.png')