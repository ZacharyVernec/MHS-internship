# Standard imports
import time
from pathlib import Path
# User imports
from helper import parse_conflicts, parse_mhs, get_minimum_sets, is_hitting_set, construct_q_matrix
# Quantum imports
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from itertools import combinations, chain



# Params
N = 100000 #Number of reads

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

def main():
    # Get data

    # Prompt user
    filename = input("format name (eg 3-5-4; 3-5-9...): ")
    filepath = Path.cwd() / 'spectras' / (filename + '-conflicts.txt')

    # Get problem
    size, conflicts_collection = parse_conflicts(filepath)
    #size, conflicts_collection = (3, [{1, 2}, {2, 3}, {1, 3}])
    if conflicts_collection:
        print("Possible candidates:", conflicts_collection)
    else:
        print("No candidates found or file is empty.")
    #universe = set.union(*conflicts_collection)
    universe = set(i for i in range(1, size+1))
    nqubits = size
    print("number of qubits for this problem: ", nqubits)

    # Get solutions
    print("Minimal hitting sets:")
    filepath_mhs = Path.cwd() / 'spectras' / (filename + '-mhs.txt')
    minimal_hitting_sets = parse_mhs(filepath_mhs)
    #minimal_hitting_sets = [{1, 2}, {2, 3}, {1, 3}]
    print(minimal_hitting_sets)
    minimum_length, minimum_hitting_sets = get_minimum_sets(minimal_hitting_sets)


    # Build & run algorithm
    start_time = time.time()

    C = 10 # Penalty constant
    Q = construct_q_matrix(conflicts_collection, universe, C) # Construct the Q matrix for the QUBO problem
    #print("QUBO formulation:",Q)
    results = run_algorithm(Q, nqubits)
    
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Computation time: {computation_time:.2f}s")


    # Compile results

    # Count solution occurrences
    solution_counts = {}
    for sample, num_occurrences in results:
        # Convert sample to hitting set format
        hitting_set = get_set_from_sample(sample)
        # Count occurrences of each unique solution
        if hitting_set in solution_counts:
            solution_counts[hitting_set] += num_occurrences
        else:
            solution_counts[hitting_set] = num_occurrences

    # Compute statistics
    print("Computing stats...")
    freq_minimal = 0
    freq_minimum = 0
    freq_hitting = 0
    weighted_approx_ratio = 0
    print("Hitting set\tFrequency\tApproximation ratio")
    for hitting_set, count in solution_counts.items():
        freq = count/N
        approx_ratio = len(hitting_set)/minimum_length
        print(f"{set(hitting_set)}\t {freq*100:05.2f}%\t {approx_ratio:.2f}")
        if is_hitting_set(hitting_set, conflicts_collection):
            freq_hitting += freq
            weighted_approx_ratio += freq * approx_ratio
            if hitting_set in minimal_hitting_sets:
                freq_minimal += freq
            if hitting_set in minimum_hitting_sets:
                freq_minimum += freq

    print("Results: ")
    print(f"Ratio of hitting sets: {freq_hitting*100:05.2f}%")
    print(f"Ratio of minimal hitting sets: {freq_minimal*100:05.2f}%")
    print(f"Ratio of minimum hitting sets: {freq_minimum*100:05.2f}%")
    print(f"Weighted approximation ratio: {weighted_approx_ratio:.4f}")



#------------------------------------------------------------

if __name__ == "__main__":
    main()