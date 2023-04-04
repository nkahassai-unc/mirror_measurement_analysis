# Written by Nathnael Kahassai

# Import necessary libraries
import numpy as np

# Quad Variance Tree Function
def QVT(X, Y, Z, max_level):
    # Calculate the global variance of the mirror surface
    global_variance = np.var(Z)

    # Initialize the subregions array to contain the entire mirror surface as a single subregion
    subregions = np.array([[(np.min(X), np.max(X)), (np.min(Y), np.max(Y)), (np.min(Z), np.max(Z)), global_variance]], dtype=object)

    # Iterate over the levels of the quadtree until the desired maximum level is reached
    for level in range(1, max_level + 1):
        # Initialize an empty list to store the new subregions at the current level
        new_subregions = []

        # Iterate over the subregions at the previous level
        for subregion in subregions:
            # Split the subregion into four equal sub-subregions in the x and y dimensions
            x_mid = (subregion[0][0] + subregion[0][1]) / 2
            y_mid = (subregion[1][0] + subregion[1][1]) / 2
            sub_subregions = [
                [(subregion[0][0], x_mid), (subregion[1][0], y_mid)],
                [(x_mid, subregion[0][1]), (subregion[1][0], y_mid)],
                [(subregion[0][0], x_mid), (y_mid, subregion[1][1])],
                [(x_mid, subregion[0][1]), (y_mid, subregion[1][1])]
            ]

            # Calculate the variance of the Z values within each sub-subregion
            sub_subregion_variances = []
            for sub_subregion in sub_subregions:
                mask = np.logical_and.reduce((X >= sub_subregion[0][0], X < sub_subregion[0][1], Y >= sub_subregion[1][0], Y < sub_subregion[1][1]))
                sub_subregion_Z = Z[mask]
                sub_subregion_variance = np.var(sub_subregion_Z)
                sub_subregion_variances.append(sub_subregion_variance)

            # Calculate the average variance of the sub-subregions
            avg_sub_subregion_variance = np.mean(sub_subregion_variances)

            # Calculate the ratio of the subregion's variance to the average variance of its sub-subregions
            variance_ratio = subregion[3] / avg_sub_subregion_variance

            # If the ratio is less than one, add the sub-subregions to the new subregions list
            if variance_ratio < 1:
                for sub_subregion in sub_subregions:
                    new_subregions.append([
                        sub_subregion[0],
                        sub_subregion[1],
                        (np.min(Z), np.max(Z)),
                        np.var(Z[np.logical_and.reduce((X >= sub_subregion[0][0], X < sub_subregion[0][1], Y >= sub_subregion[1][0], Y < sub_subregion[1][1]))])
                    ])

            # If the ratio is greater than or equal to one, add the current subregion to the new subregions list
            else:
                new_subregions.append(subregion)

        # Convert the new subregions list to a numpy array and overwrite the previous subregions array
        subregions = np.array(new_subregions, dtype=object)

    # Return the final subregions array
    return subregions