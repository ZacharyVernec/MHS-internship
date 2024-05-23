# Quantum imports
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from itertools import combinations, chain

# User imports
from test_helper import run_tests



# Algorithm functions

def run_algorithm(q_matrix, nqubits):
    # Convert the Q matrix to a binary quadratic model
    bqm = BinaryQuadraticModel.from_qubo(q_matrix)

    # Use the simulated annealing sampler for the problem
    sampler = SimulatedAnnealingSampler()

    # We run the simulated annealing sampler for N reads
    print("Annealing...")
    sampleset = sampler.sample(bqm, num_reads=N)

    return sampleset.data(['sample', 'num_occurrences'])

def get_set_from_sample(sample):
    """ Convert from sample format of algorithm result to frozenset of integers"""
    return frozenset(i for i, val in sample.items() if val == 1)


#------------------------------------------------------------

if __name__ == "__main__":
    N = 100000 #Number of reads
    run_tests(run_algorithm, get_set_from_sample, N)