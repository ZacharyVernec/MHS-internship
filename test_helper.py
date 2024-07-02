# Standard imports
import time
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
from math import sqrt

# Parse example files
def parse_conflicts(filepath):
    """
    Parse an i-j-k-conflicts.txt file into python objects

    Returns:
    universe (list of ints): sorted, representing all elements in the file
    collection (list of sets of ints): representing a collection of conflicts
    """

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
    universe = list(set.union(*collection))
    universe.sort()

    return universe, collection
def parse_mhs(filepath):
    """
    Parse an i-j-k-mhs.txt file into python objects

    Returns:
    collection (list of sets of ints): representing a collection of MHS conflicts
    """

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

# Set helpers
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
def construct_q_matrix(collection, universe, lambda_weight,beta_weight):
    """
    Construct the QUBO matrix for the Minimal Hitting Set Problem.
    
    Parameters:
    universe (list of ints): ordered list of elements in the universe.
    collection (list of sets): List of subsets of universe.
    lambda_weight (float): Weight for the hitting condition.
    beta_weight (float): Weight for avoiding unnecessary elements in larger sets.
    
    Returns:
    Q (dict of {(int,int): float}): QUBO matrix with keys being edges and values being weights.
    """
    
    # Initialize QUBO dictionary
    Q = {}

    # Ensure all subsets are within the universe
    assert all(subset.issubset(universe) for subset in collection), "The collection contains elements outside the universe"

    # Linear terms: min sum(x_i)
    for i in universe:
        Q[(i, i)] = 1

    # Penalty terms: ensure each subset has at least one element
    for subset in collection:
        for i in subset:
            Q[(i, i)] += -2 * lambda_weight
            for j in subset:
                if i != j:
                    if (i, j) not in Q:
                        Q[(i, j)] = 0
                    Q[(i, j)] += lambda_weight
    
    # Additional term to penalize unnecessary elements in larger sets (Second version of our QUBO formulation. They give better results!)
    for i in universe:
        for j in universe:
            if i != j:
                if (i, j) not in Q:
                    Q[(i, j)] = 0
                Q[(i, j)] -= beta_weight

    return Q

def compute_overlap(minimal_hitting_sets, solution_counts, N):
    """
    Compute the overlap between the exact solution and the one obtained by the QA simulation.

    Parameters:
    minimal_hitting_sets (list of sets): List of minimal hitting sets (exact solution)
    solution_counts (dict of {frozenset: int}): Dictionary of solution samples and their occurrences from the QA simulation.
    N (int): Number of reads

    Returns:
    overlap (float): The overlap between the exact solution and the QA simulation.
    """

    # Normalize the frequencies to probabilities
    total_count = sum(solution_counts.values())
    normalized_frequencies = {k: v / total_count for k, v in solution_counts.items()}

    # Construct phi_sol vector
    num_minimal_sets = len(minimal_hitting_sets)
    phi_sol = defaultdict(float)
    for i, mhs in enumerate(minimal_hitting_sets):
        phi_sol[frozenset(mhs)] = 1 / sqrt(num_minimal_sets)

    # Construct phi_QA vector
    phi_QA = defaultdict(float)
    for hs, prob in normalized_frequencies.items():
        for i, mhs in enumerate(minimal_hitting_sets):
            if hs == mhs:
                phi_QA[frozenset(mhs)] = sqrt(prob)

    # Compute overlap
    overlap = sum(phi_sol[k] * phi_QA[k] for k in phi_sol)
    return overlap




def run_tests(run_algorithm, get_set_from_sample, N):
    """Prompts user for file, runs algorithm, and prints results
    
    Arguments:
    run_algorithm -- a function that takes a QUBO matrix and returns a dict of {sample: occurrences}
    get_set_from_sample -- a function that takes an algorithm return sample, and returns a frozenset of integers (corresponding to universe elements)
    """


    # Get data

    # Prompt user
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('filename', metavar="filename (eg 3-5-4; 3-5-9...): ")
    args = parser.parse_args()
    filepath = Path.cwd() / 'spectras' / (args.filename + '-conflicts.txt')

    # Get problem
    universe, conflicts_collection = parse_conflicts(filepath)
    #universe, conflicts_collection = (3, [{1, 2}, {2, 3}, {1, 3}])
    if conflicts_collection:
        print("Possible candidates:", conflicts_collection)
    else:
        print("No candidates found or file is empty.")
    nqubits = len(universe)
    print("Number of qubits for this problem: ", nqubits)

    # Get solutions
    print("Minimal hitting sets:")
    filepath_mhs = Path.cwd() / 'spectras' / (args.filename + '-mhs.txt')
    minimal_hitting_sets = parse_mhs(filepath_mhs)
    #minimal_hitting_sets = [{1, 2}, {2, 3}, {1, 3}]
    print(minimal_hitting_sets)
    minimum_length, minimum_hitting_sets = get_minimum_sets(minimal_hitting_sets)


    # Build & run algorithm
    start_time = time.time()

    # Set weights
    lambda_weight = 5 # Weight for the hitting condition
    beta_weight = 0.0000000001 # weight to penalize unnecessary elements in larger sets

    Q = construct_q_matrix(conflicts_collection, universe, lambda_weight,beta_weight) # Construct the Q matrix for the QUBO problem
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
        hitting_set = get_set_from_sample(sample, universe)
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

    # Compute and print overlap
    overlap = compute_overlap(minimal_hitting_sets, solution_counts, N)
    print(f"Overlap: {overlap:.4f}")

    # Save data into a folder.
    output_folder = Path.cwd() / 'QA_results'
    output_folder.mkdir(exist_ok=True)
    results_file = output_folder / f"{nqubits}-qubit-results.txt"

    with results_file.open('a') as f:
        # Write header if file is empty
        if results_file.stat().st_size == 0:
            f.write("Filename\tRatio of hitting sets\tRatio of minimal hitting sets\tRatio of minimum hitting sets\tWeighted approximation ratio\tOverlap\n")
        # Write the summary line
        f.write(f"{args.filename}\t{freq_hitting*100:05.2f}%\t{freq_minimal*100:05.2f}%\t{freq_minimum*100:05.2f}%\t{weighted_approx_ratio:.4f}\t{overlap:.4f}\n")

    print(f"Results saved to {results_file}")
    print("------------------------------")