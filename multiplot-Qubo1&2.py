import numpy as np
import matplotlib.pyplot as plt
import os

def process_results(folder):
    # Initialize lists to hold results
    qubit_sizes = []
    mean_overlap_squares = []
    variance_overlap_squares = []

    # Loop through files
    for qubit_size in range(1, 11):
        filename = f"{folder}/{qubit_size}-qubit-results.txt"
        
        if os.path.isfile(filename):
            data = np.loadtxt(filename, skiprows=1, usecols=5, delimiter='\t')
            
            # Compute the square of the overlap values
            overlap_square = data ** 2
            
            qubit_sizes.append(qubit_size)
            mean_overlap_squares.append(np.mean(overlap_square))
            variance_overlap_squares.append(np.var(overlap_square))

    # Convert lists to numpy arrays for plotting
    qubit_sizes = np.array(qubit_sizes)
    mean_overlap_squares = np.array(mean_overlap_squares)
    variance_overlap_squares = np.array(variance_overlap_squares)

    return qubit_sizes, mean_overlap_squares, variance_overlap_squares

# Process results for both QUBO formulations
qubit_sizes_QUBO1, mean_overlap_squares_QUBO1, variance_overlap_squares_QUBO1 = process_results("QA_results_Qubo1")
qubit_sizes_QUBO2, mean_overlap_squares_QUBO2, variance_overlap_squares_QUBO2 = process_results("QA_results")

# Plotting
plt.figure(figsize=(12, 8))

# Plot for QUBO1
plt.errorbar(qubit_sizes_QUBO1, mean_overlap_squares_QUBO1, yerr=variance_overlap_squares_QUBO1, fmt='o', capsize=5, label='QUBO1 Mean value with variance error bars')

# Plot for QUBO2
plt.errorbar(qubit_sizes_QUBO2, mean_overlap_squares_QUBO2, yerr=variance_overlap_squares_QUBO2, fmt='s', capsize=5, label='QUBO2 Mean value with variance error bars')

plt.xlabel('Number of Qubits')
plt.ylabel(r'$|\langle \psi_{\mathrm{sol}} | \psi_{\mathrm{QA}} \rangle|^2$')
plt.title('QA Simulation Comparison')
plt.legend()
plt.grid(True)
plt.savefig('qubit_results_multiplot.png')
