# Standard imports
from itertools import chain, combinations
from pathlib import Path
import time
import numpy as np

# User imports
from test_helper import parse_conflicts, is_hitting_set

def get_powerset(iterable):
    """ from https://stackoverflow.com/a/1482316 """
    "get_powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))
def get_hitting_sets(universe, collection):
    """
    Arguments:
        universe (list of ints) -- sorted, representing all elements in the file
        collection (list of sets of ints) -- representing a collection of conflicts
    Returns:
        all hitting sets (iterator of tuples of ints)
    """
    is_hitting_set_for_collection = lambda s: is_hitting_set(s, collection)
    return filter(is_hitting_set_for_collection, get_powerset(universe))


def main():
    # Compute for all test data
    universe_sizes = np.empty((8,8,10), dtype=int)
    collection_sizes = np.empty_like(universe_sizes)
    n_qubits = np.empty_like(universe_sizes)
    n_hittingsets = np.empty_like(universe_sizes)
    hitting_set_probabilities = np.empty_like(universe_sizes, dtype=float)
    
    for i in range(8):
        for j in range(8):
            for k in range(10):

                # Get data
                filepath = Path.cwd() / 'spectras' / (f'{i+3}-{j+3}-{k+1}-conflicts.txt')

                # Get problem
                universe, conflicts_collection = parse_conflicts(filepath)
                if conflicts_collection:
                    print("Possible candidates:", conflicts_collection)
                else:
                    print("No candidates found or file is empty.")
                nqubits = len(universe) + sum(len(s) for s in conflicts_collection) #TODO check where used downstream
                print("Number of qubits for this problem: ", nqubits)


                # Run algorithm
                start_time = time.time()
                
                hitting_sets = list(get_hitting_sets(universe, conflicts_collection))

                end_time = time.time()
                computation_time = end_time - start_time
                print(f"Computation time: {computation_time:.2f}s")


                # Compile results

                universe_sizes[i,j,k] = len(universe)
                collection_sizes[i,j,k] = len(conflicts_collection)
                n_qubits[i,j,k] = nqubits
                n_hittingsets[i,j,k] = len(hitting_sets)
                hitting_set_probabilities[i,j,k] = 1/len(hitting_sets)

    # Save results

    outfolderpath = Path.cwd() / 'hitting_sets_data'

    np.save(outfolderpath / 'universe_sizes', universe_sizes)
    np.save(outfolderpath / 'collection_sizes', collection_sizes)
    np.save(outfolderpath / 'n_qubits', n_qubits)
    np.save(outfolderpath / 'n_hittingsets', n_hittingsets)
    np.save(outfolderpath / 'hitting_set_probabilities', hitting_set_probabilities)


    # Show some statistics

    print(f"{i+3}-{j+3}-{k+1} conflicts:")
    print(f"Hitting sets: {np.mean(n_hittingsets)} ± {np.std(n_hittingsets)}")
    print(f"Hitting set probabilities: {np.mean(hitting_set_probabilities)} ± {np.std(hitting_set_probabilities)}")


if __name__ == "__main__":
    main()

    #TODO calculate amount of minimal HS, of non-minimal HS