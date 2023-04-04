# Written by Nathnael Kahassai

# Import necessary libraries
import numpy as np
from mirror import mirror_surface_model
from qvt import QVT
from mc import QVT_prioritized_Monte_Carlo_simulation, prioritize_subregions, QVT_multiple_Monte_Carlo_simulations

# Define the Monte Carlo simulation parameters
desired_accuracy = 0.007 # RMS error threshold
num_runs = 100 # Number of Monte Carlo simulation runs to perform
num_measurements = 100 # Number of measurements to take in each Monte Carlo simulation run
max_num_measurements = 100 # Maximum number of measurements to take in the multiple Monte Carlo simulations

# Define the grid spacing
dx = 1
dy = 1

# Define the x and y limits of the grid
x_min = -500.0
x_max = -300.0
y_min = 300.0
y_max = 500.0

# Generate the x and y coordinates of the grid
x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
X, Y = np.meshgrid(x, y)
print(np.shape(Y))

# Evaluate the mirror surface model at the grid points
Z = mirror_surface_model(X, Y)
print("Mirror Surface Generated")

# Define the maximum level of recursion for the QVT algorithm
max_level = 8
print(max_level,"levels of QVT recursion ")

# Identify subregions with high variance using QVT
subregions = QVT(X, Y, Z, max_level=max_level)
print(np.size(subregions), "subregions")

# Prioritize subregions for measurement
prioritized_subregions = prioritize_subregions(subregions)
print(np.size(prioritized_subregions), "prioritized subregions")

# Run the QVT-prioritized Monte Carlo simulation
min_num_measurements, RMS_errors = QVT_prioritized_Monte_Carlo_simulation(X, Y, Z, prioritized_subregions, desired_accuracy, num_runs, num_measurements)

# Print the estimated minimum number of measurements required to achieve the desired level of accuracy
print("Estimated minimum number of measurements required to achieve {} RMS error: {}".format(desired_accuracy, min_num_measurements))

# Print the list of RMS errors
print("List of RMS Errors", RMS_errors)

# Run multiple Monte Carlo simulations with varying numbers of measurements
mean_RMS_errors = QVT_multiple_Monte_Carlo_simulations(X, Y, Z, prioritized_subregions, desired_accuracy, num_runs, max_num_measurements)

# Print the mean RMS error for each number of measurements
print("Mean RMS Error for each Number of Measurements:", mean_RMS_errors)