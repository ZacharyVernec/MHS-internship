import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize lists to hold results
qubit_sizes = []
mean_overlap_squares = []
variance_overlap_squares = []

# Loop through files
for qubit_size in range(1, 11):
    filename = f"QA_results/{qubit_size}-qubit-results.txt"
    
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
error_bars = variance_overlap_squares  # Variance for error bars

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(qubit_sizes, mean_overlap_squares, yerr=error_bars, fmt='o', capsize=5, label='Mean value with variance error bars')
plt.xlabel('Number of Qubits')
plt.ylabel(r'$|\langle \psi_{\mathrm{sol}} | \psi_{\mathrm{QA}} \rangle|^2$')
plt.title('QA Simulation')
plt.legend()
plt.grid(True)
plt.savefig('qubit_results_plot-QUBO1.png')