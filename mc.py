# Monte Carlo Modules and Functions
# Written by Nathnael Kahassai

# Import necessary libraries
import numpy as np
from mirror import mirror_surface_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define a function to randomly select a set of measurement points from the prioritized subregions identified by QVT
def select_measurement_points(subregions, num_measurements):
    # Create an empty array to hold the measurement points
    measurement_points = np.empty((0, 3))

    # Iterate over the subregions, starting with the highest-variance subregions as identified by QVT
    for subregion in subregions[::-1]:
        # Extract the X, Y, and Z coordinates for the current subregion
        x_min, x_max = subregion[0]
        y_min, y_max = subregion[1]
        z_min, z_max = subregion[2]

        # Calculate the number of measurement points to take within the current subregion
        n_subregion_points = int(num_measurements * subregion[3] / subregions[-1][3])

        # Generate a set of random measurement points within the current subregion
        X_subregion = np.random.uniform(x_min, x_max, n_subregion_points)
        Y_subregion = np.random.uniform(y_min, y_max, n_subregion_points)
        Z_subregion = np.random.uniform(z_min, z_max, n_subregion_points)

        # Add the measurement points to the measurement_points array
        measurement_points_subregion = np.column_stack((X_subregion, Y_subregion, Z_subregion))
        measurement_points = np.vstack((measurement_points, measurement_points_subregion))

    return measurement_points, subregions[::-1]


# Prioritize the subregions by sorting them in descending order based on their variance
def prioritize_subregions(subregions):
    # Sort the subregions by variance in descending order
    sorted_subregions = sorted(subregions, key=lambda x: x[3], reverse=True)

    # Initialize an empty list to store the prioritized subregions
    prioritized_subregions = []

    # Iterate over the sorted subregions and add each one to the prioritized list
    for subregion in sorted_subregions:
        prioritized_subregions.append(subregion)

    return prioritized_subregions


# Calculate the RMS error between the measured and expected Z values for each set of measurement points selected in each Monte Carlo simulation run
def calculate_RMS_error(mirror_surface_model, measurement_points):
    # Calculate the expected Z values for the measurement points based on the mathematical model
    expected_Z = mirror_surface_model(measurement_points[:, 0], measurement_points[:, 1])

    # Calculate the RMS error between the measured and expected Z values
    RMS_error = np.sqrt(np.mean((measurement_points[:, 2] - expected_Z)**2))

    return RMS_error


# Analyze the Monte Carlo simulation results to estimate the minimum number of measurements required to achieve the desired level of accuracy
def analyze_Monte_Carlo_results(RMS_errors, accuracy_threshold):
    # Calculate the number of Monte Carlo simulation runs
    num_runs = len(RMS_errors)

    # Iterate over the Monte Carlo simulation runs, starting with the fewest measurement points
    for i in range(num_runs):
        # Check if the RMS error for the current number of measurement points is below the desired accuracy threshold
        if RMS_errors[i] < accuracy_threshold:
            # If it is, return the current number of measurement points as the minimum required
            return (i+1)

    # If the desired accuracy threshold was not achieved with the maximum number of measurement points, return -1 to indicate failure
    return -1


def get_mask(x, y, subregion):
    """
    Get a boolean mask indicating whether each point (x[i], y[i]) falls within the specified subregion
    """
    x_mask = (x >= subregion[0][0]) & (x <= subregion[0][1])
    y_mask = (y >= subregion[1][0]) & (y <= subregion[1][1])
    return x_mask & y_mask


def QVT_prioritized_Monte_Carlo_simulation(X, Y, Z, prioritized_subregions, desired_accuracy, num_runs, num_measurements):
    # Initialize empty list to store the RMS errors from each Monte Carlo simulation run
    RMS_errors = []

    # Iterate over the specified number of Monte Carlo simulation runs
    for i in range(num_runs):
        # Select a set of measurement points from the prioritized subregions identified by QVT
        measurement_points, _ = select_measurement_points(prioritized_subregions, num_measurements)

        # Calculate the RMS error between the measured and expected Z values for each set of measurement points selected in each Monte Carlo simulation run
        RMS_error = calculate_RMS_error(mirror_surface_model, measurement_points)

        # Add the RMS error from the current simulation run to the list of errors
        RMS_errors.append(RMS_error)

        # Print the progress of the Monte Carlo simulation
        if (i+1) % 100 == 0:
            print("Completed {} of {} Monte Carlo simulation runs".format(i+1, num_runs))

    # Analyze the Monte Carlo simulation results to estimate the minimum number of measurements required to achieve the desired level of accuracy
    min_num_measurements = analyze_Monte_Carlo_results(RMS_errors, desired_accuracy)

    # Generate 3D wireframe plot of the mirror surface with red dots at measurement locations
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, Z, color='black', linewidths=0.5)
    ax1.scatter(measurement_points[:, 0], measurement_points[:, 1], measurement_points[:, 2], color='red', s=1)

    # Set the plot title and axis labels
    ax1.set_title("Wireframe Plot of Mirror Surface with Measurement Locations")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Count the number of prioritized subregions and the number of measurement points
    num_subregions = len(prioritized_subregions)
    num_points = measurement_points.shape[0]

    # Print the number of prioritized subregions and measurement points
    print("Number of prioritized subregions: {}".format(num_subregions))
    print("Number of measurement points: {}".format(num_points))


    # Generate 2D scatter plot of the mirror surface with different colors indicating the different prioritized subregions
    ax2 = fig.add_subplot(122)
    for i, subregion in enumerate(prioritized_subregions):
        if subregion[3] >= np.percentile(np.array(prioritized_subregions)[:, 3], 50):
            ax2.scatter(X[get_mask(X, Y, subregion)], Y[get_mask(X, Y, subregion)], color='red', s=1, alpha=0.5)
        else:
            ax2.scatter(X[get_mask(X, Y, subregion)], Y[get_mask(X, Y, subregion)], color='purple', s=1, alpha=0.5)
        rect = patches.Rectangle((subregion[0][0], subregion[1][0]), subregion[0][1]-subregion[0][0], subregion[1][1]-subregion[1][0], linewidth=0.5, edgecolor='black', facecolor='none')
        ax2.add_patch(rect)

    # Set the plot title and axis labels
    ax2.set_title("Scatter Plot of Mirror Surface with Prioritized Subregions")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Show the plots
    plt.show()

    # Return the estimated minimum number of measurements required to achieve the desired level of accuracy
    return min_num_measurements, RMS_errors


def QVT_multiple_Monte_Carlo_simulations(X, Y, Z, prioritized_subregions, desired_accuracy, num_runs, max_num_measurements):
    # Initialize empty list to store the RMS errors from each Monte Carlo simulation run
    RMS_errors = []

    # Iterate over the specified number of Monte Carlo simulation runs
    for i in range(num_runs):
        # Select a subset of the prioritized subregions for measurement
        selected_subregions = prioritized_subregions[:max_num_measurements]

        # Initialize empty list to store the RMS errors for each number of measurements
        RMS_errors_by_num_measurements = []

        # Iterate over each number of measurements from 0 to the maximum number of measurements
        for num_measurements in range(max_num_measurements + 1):
            # Select the first num_measurements subregions for measurement
            selected_measurement_points, _ = select_measurement_points(selected_subregions[:num_measurements], num_measurements)

            # Calculate the RMS error between the measured and expected Z values
            RMS_error = calculate_RMS_error(mirror_surface_model, selected_measurement_points)

            # Add the RMS error to the list of errors for the current number of measurements
            RMS_errors_by_num_measurements.append(RMS_error)

        # Add the list of RMS errors for the current Monte Carlo simulation to the list of errors for all simulations
        RMS_errors.append(RMS_errors_by_num_measurements)

        # Print the progress of the Monte Carlo simulation
        if (i+1) % 10 == 0:
            print("Completed {} of {} Monte Carlo simulation runs".format(i+1, num_runs))

    # Calculate the mean RMS error for each number of measurements over all Monte Carlo simulations
    mean_RMS_errors = np.mean(RMS_errors, axis=0)

    # Generate plot of RMS errors as a function of number of measurements
    fig, ax = plt.subplots()
    ax.plot(range(max_num_measurements + 1), mean_RMS_errors)

    # Set the plot title and axis labels
    ax.set_title("RMS Error vs. Number of Measurements")
    ax.set_xlabel("Number of Measurements")
    ax.set_ylabel("RMS Error")

    # Show the plot
    plt.show()

    # Return the mean RMS error for each number of measurements
    return mean_RMS_errors