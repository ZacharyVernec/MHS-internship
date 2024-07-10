# Quantum imports
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler #TODO move to dwave.samplers as neal is deprecated
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

def get_set_from_sample(sample, universe):
    """ 
    Convert from sample format of algorithm result to frozenset of integers in the universe

    Parameters:
    sample (dict of {int: bit}): sample from the algorithm, where keys are the elements of the universe

    Returns:
    (frozenset of ints): a set representing a candidate hitting set
    """
    max_universe = max(universe, default=-1) #default for empty universe: arbitrary as sample will be empty
    return frozenset(i for i, val in sample.items() if ((val == 1) and (i <= max_universe)))


#------------------------------------------------------------

if __name__ == "__main__":
    N = 100000 #Number of reads
    run_tests(run_algorithm, get_set_from_sample, N)