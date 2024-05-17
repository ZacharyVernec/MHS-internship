from pathlib import Path
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
import numpy as np
from itertools import combinations, chain
import time


#Numer of reads
N = 100000


# First of all we make a function that given different subsets returns it in the adequate format.
def parse_conflicts(filepath):
    # get size
    filename = filepath.name #remove folders 
    size = int(filename.split('-')[0]) #take first number

    # parse conflicts
    collection = []
    with filepath.open('r') as f:
        for line in f:
            line = line.strip()[1:-1] # remove whitespace and curly braces
            subset = set(int(x[1:]) for x in line.split(',')) #split into items, remove c prefix
            collection.append(subset)

    return size, collection
def parse_mhs(filepath):
    collection = []
    with filepath.open('r') as f:
        str_tuples = f.read().strip()[2:-2].split('}, {') # gives list of e.g. 'c1, c3' strings
    if str_tuples == ['']: #empty
        collection.append[set()]
    else: #nonempty
        collection = []
        for tup in str_tuples:
            hitting_set = set(int(item[1:]) for item in tup.split(', '))
            collection.append(hitting_set)
    return collection


def get_minimum_sets(collection):
    argmins = []
    minimum = np.inf
    for subset in collection:
        if len(subset) < minimum:
            minimum = len(subset)
            argmins = [subset]
        elif len(subset) == minimum:
            argmins.append(subset)
    return minimum, argmins
def is_hitting_set(subset, collection):
    misses = [subset.isdisjoint(s) for s in collection]
    return not any(misses)



# Construct the Q matrix for the QUBO problem
def construct_q_matrix(collection, universe, C):
    Q = {}
    assert set.union(*collection) <= universe, "The collection contains elements outside the universe"
    # Linear terms: min sum(x_i)
    for i in universe:
        Q[(i, i)] = 1
    # Penalty terms: ensure each subset has at least one element
    for subset in collection:
        print(subset)
        for i in subset:
            print(i)
            Q[(i, i)] += -2 * C
            for j in subset:
                if i != j:
                    if (i, j) not in Q:
                        Q[(i, j)] = 0
                    Q[(i, j)] += C #TODO fix
    return Q



def main():
    # Prompt the user for the file name
    filename = input("format name (eg 3-5-4; 3-5-9...): ")
    filepath = Path.cwd() / 'spectras' / (filename + '-conflicts.txt')

    # Read the candidates from the file
    size, output_collection = parse_conflicts(filepath)
    #size, output_collection = (3, [{1, 2}, {2, 3}, {1, 3}])

    # Print the output
    if output_collection:
        print("Possible candidates:", output_collection)
    else:
        print("No candidates found or file is empty.")

    #------------------------------------------------------------

    # Read and print the minimal hitting sets from the file
    print("Minimal hitting sets:")
    filepath_mhs = Path.cwd() / 'spectras' / (filename + '-mhs.txt')
    minimal_hitting_sets = parse_mhs(filepath_mhs)
    #minimal_hitting_sets = [{1, 2}, {2, 3}, {1, 3}]
    print(minimal_hitting_sets)
    minimum_length, minimum_hitting_sets = get_minimum_sets(minimal_hitting_sets)



    #------------------------------------------------------------
    print("----------------------------------------------")

    # collection of sets for the problem
    collection = output_collection
    #universe = set.union(*collection)
    universe = set(i for i in range(1, size+1))

    # Penalty constant
    C = 10

    start_time = time.time()

    Q = construct_q_matrix(collection, universe, C) # Construct the Q matrix for the QUBO problem

    # Convert the Q matrix to a binary quadratic model
    bqm = BinaryQuadraticModel.from_qubo(Q)

    # Use the simulated annealing sampler for the problem
    sampler = SimulatedAnnealingSampler()

    # We run the simulated annealing sampler for N reads
    print("Simulated Annealing...")
    sampleset = sampler.sample(bqm, num_reads=N)

    # Initialize a dictionary to count solution occurrences
    solution_counts = {}

    # Iterate over all samples in the sampleset
    for sample, num_occurrences in sampleset.data(['sample', 'num_occurrences']):
        # Convert sample to hitting set format
        hitting_set = frozenset(i for i, val in sample.items() if val == 1)

        # Count occurrences of each unique solution
        if hitting_set in solution_counts:
            solution_counts[hitting_set] += num_occurrences
        else:
            solution_counts[hitting_set] = num_occurrences

    # Display all unique solutions and their frequencies, compare solution
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
        if is_hitting_set(hitting_set, output_collection):
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

    # End timing the computation
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Computation time: {computation_time:.2f}s")



#------------------------------------------------------------

if __name__ == "__main__":
    main()