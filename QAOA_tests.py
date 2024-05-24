# Standard imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np

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

# User imports
from test_helper import run_tests



# Quantum algorithm setup helper functions

def ising_terms_to_pauli(H_C, nqubits):

    # We discart the offset since it is a constant term that does not affect the optimization
    # Now we have to build the hamiltonian with the Pauli gates. We know that the Ising Hamiltonian is given by:
    # H = sum_i h_i Z_i + sum_ij J_ij Z_i Z_j
    # So the linear terms are proportional to the Z gate and the quadratic terms are proportional to the ZZ gate. We can build the Hamiltonian using the Pauli gates


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

# Algorithm functions

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

def get_set_from_sample(sample, universe):
    """ 
    Convert from sample format of algorithm result to frozenset of integers in the universe

    Parameters:
    sample (string): a bitstring of qubit results, where the least significant bit is the rightmost
    universe (list of ints): Ordered list of elements in the universe.

    Returns:
    (frozenset of ints): a set representing a candidate hitting set
    """
    return frozenset(universe[i] for i,c in enumerate(sample[::-1]) if c == '1')

#------------------------------------------------------------

if __name__ == "__main__":
    N = 1024 #Number of shots
    run_tests(run_algorithm, get_set_from_sample, N)