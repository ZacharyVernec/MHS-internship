import numpy as np

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
        for i in subset:
            Q[(i, i)] += -2 * C
            for j in subset:
                if i != j:
                    if (i, j) not in Q:
                        Q[(i, j)] = 0
                    Q[(i, j)] += C #TODO fix
    return Q