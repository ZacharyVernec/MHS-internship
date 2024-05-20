# We start by importing the necessary libraries
from dimod.utilities import qubo_to_ising
import numpy as np
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_distribution

# SciPy minimizer routine
from scipy.optimize import minimize

# rustworkx graph library
import rustworkx as rx
from rustworkx.visualization import mpl_draw

from qiskit import Aer, transpile, execute
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp

from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from itertools import combinations

# User library
from helper import parse_conflicts, parse_mhs, get_minimum_sets, is_hitting_set, construct_q_matrix


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

    # collection of sets for the problem
    collection = output_collection
    #universe = set.union(*collection)
    universe = set(i for i in range(1, size+1))

    #------------------------------------------------------------
    print("----------------------------------------------")

    nqubits = size
    print("number of qubits for this problem: ",nqubits)

    # Penalty constant
    C = 10

    start_time = time.time()

    Q = construct_q_matrix(collection, universe, C) # Construct the Q matrix for the QUBO problem
    #print("QUBO formulation:",Q)

    #------------------------------------------------------

    # To formulate the model into ising we use a function from qiskit that given the dictionary with the QUBO encoded it gives an output of the Ising Hamiltonian
    # x'Qx -> offset s'Js + h's 
    H_C = qubo_to_ising(Q, offset=0.0)
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

    #now we plot the circuit
    ansatz.decompose(reps = 3).draw('mpl', style='iqx')

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

    # Initialize a dictionary to count solution occurrences
    solution_counts = {}

    for sample, num_occurrences in counts.items():
        # Convert sample to hitting set format
        hitting_set = frozenset(i+1 for i,c in enumerate(sample[::-1]) if c == '1') #reverse order because x_0 is least significant

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



if __name__ == "__main__":
    main()