# Standard imports
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# User imports
from helper import parse_conflicts, parse_mhs, get_minimum_sets, is_hitting_set, construct_q_matrix
# Quantum imports
from dimod.utilities import qubo_to_ising
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit import Aer, transpile, execute
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp



# Params
N = 1024 #Number of shots


# We discart the offset since it is a constant term that does not affect the optimization
# Now we have to build the hamiltonian with the Pauli gates. We know that the Ising Hamiltonian is given by:
# H = sum_i h_i Z_i + sum_ij J_ij Z_i Z_j
# So the linear terms are proportional to the Z gate and the quadratic terms are proportional to the ZZ gate. We can build the Hamiltonian using the Pauli gates
def ising_terms_to_pauli(H_C, nqubits):

    linear, quadratic, offset = H_C

    # Initialize lists for coefficients and Pauli strings
    coefficients = []
    pauli_strings = []

    # Handle linear terms
    for qubit, coeff in linear.items():
        # Add the coefficient
        coefficients.append(coeff)
        # Generate the Pauli string with Z at the correct position and I elsewhere
        pauli_string = ''.join(['Z' if i == qubit else 'I' for i in range(1, nqubits + 1)])
        pauli_strings.append(pauli_string)

    # Handle quadratic terms
    for (qubit1, qubit2), coeff in quadratic.items():
        adjusted_coeff = coeff
        # Add the coefficient
        coefficients.append(adjusted_coeff)
        # Generate the Pauli string with Zs at the correct positions
        pauli_string = ''.join(['Z' if i == qubit1 or i == qubit2 else 'I' for i in range(1, nqubits + 1)])
        pauli_strings.append(pauli_string)

    return coefficients, pauli_strings

def Hamiltonian(pauli_strings, coefficients):
    # Initialize a list to hold (Pauli string, coefficient) tuples
    pauli_list = []

    # Iterate over the Pauli strings and coefficients
    for i in range(len(pauli_strings)):
        # Append the tuple to the list
        pauli_list.append((pauli_strings[i], coefficients[i]))

    # Create the Hamiltonian from the list of tuples
    H = SparsePauliOp.from_list(pauli_list)

    return H

def expectation_value(params, ansatz, H):
    backend = Aer.get_backend('aer_simulator_statevector')
    quantum_instance = QuantumInstance(backend, seed_simulator=42, seed_transpiler=4)

    # Bind parameters to the ansatz
    param_dict = dict(zip(ansatz.parameters, params))
    bound_ansatz = ansatz.assign_parameters(param_dict)

    # Setup for expectation value computation
    op = ~StateFn(H) @ StateFn(bound_ansatz)
    expectation = PauliExpectation().convert(op)

    # Use CircuitSampler for the computation
    sampler = CircuitSampler(quantum_instance).convert(expectation)

    return sampler.eval().real  # Return the real part of the expectation value

def get_measurement_circuit(circuit):
    # Clone the optimized circuit to keep the original intact
    measurement_circuit = circuit.copy()
    # Add measurement to all qubits
    measurement_circuit.measure_all()
    return measurement_circuit

def execute_circuit(measurement_circuit):
    # Use Aer's qasm_simulator
    backend = Aer.get_backend('qasm_simulator')

    # Execute the circuit
    job = execute(measurement_circuit, backend, shots=N)
    result = job.result()

    # Get the counts of each bitstring
    counts = result.get_counts()
    return counts

def run_algorithm(q_matrix, nqubits):
    # To formulate the model into ising we use a function from qiskit that given the dictionary with the QUBO encoded it gives an output of the Ising Hamiltonian
    # x'Qx -> offset s'Js + h's 
    H_C = qubo_to_ising(q_matrix, offset=0.0)
    #print("ISING Hamiltonian",H_C)
    h, J, offset = H_C
    #print('offset:', offset)
    #print('linear terms:', h)
    #print('quadratic terms:', J)

    # We discart the offset since it is a constant term that does not affect the optimization
    # Now we have to build the hamiltonian with the Pauli gates. We know that the Ising Hamiltonian is given by:
    # H = sum_i h_i Z_i + sum_ij J_ij Z_i Z_j
    # So the linear terms are proportional to the Z gate and the quadratic terms are proportional to the ZZ gate. We can build the Hamiltonian using the Pauli gates

    # Convert the Ising Hamiltonian to a list of Pauli strings and coefficients
    coefficients, pauli_strings = ising_terms_to_pauli(H_C, nqubits)
    #print('Pauli strings:', pauli_strings)
    #print('Coefficients:', coefficients)

    # Now we build the hamiltonian:
    H = Hamiltonian(pauli_strings, coefficients)
    #print("Hamiltonian: ", H)


    ansatz = QAOAAnsatz(H, reps=10)

    # compatible H for qiskit
    compatible_H = PauliSumOp.from_list(H.to_list())

    algorithm_globals.random_seed = 42

    initial_params = np.random.rand(ansatz.num_parameters) * np.pi

    # Using COBYLA optimizer
    print("Optimizing...")
    objective_function = lambda params: expectation_value(params, ansatz, compatible_H)
    result = minimize(objective_function, initial_params, method='COBYLA')
    #print('Optimal parameters:', result.x)

    print("Getting expectation of optimized...")
    # Until here we have the circuit that evolves the state |beta,gamma>. We need to compute the expectation value
    optimized_params = result.x  # Optimized parameters from the previous step
    optimized_circuit = ansatz.bind_parameters(optimized_params)

    measurement_circuit = get_measurement_circuit(optimized_circuit)
    counts = execute_circuit(measurement_circuit)

    return counts.items()

def get_set_from_sample(sample):
    """ Convert from sample format of algorithm result to frozenset of integers"""
    # reverses order of sample since qiskit has samples as binary strings with least sig on right
    return frozenset(i+1 for i,c in enumerate(sample[::-1]) if c == '1')

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