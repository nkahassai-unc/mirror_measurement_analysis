# Mirror Surface Model
# Written by Saianeesh Haridas and Nathnael Kahassai

# Import necessary libraries
import numpy as np


# Define the mathematical model for the mirror surface and calculate the height of the mirror surface at each (X, Y) coordinate using the mirror function with the provided coefficients
def mirror_surface_model(X, Y):
    """
    Define the mathematical model for the mirror surface
    
    @param X: x positions to calculate at
    @param Y: y positions to calculate at
    
    @return Z: z position of the mirror at each xy
    """
    # Mirror coefficients
    a_primary = np.array([
        [0., 0., -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601,],
        [0., 0., 0., 0., 0., 0., 0.],
        [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1.8083973, -0.603195, 0.2177414, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0.0394559, 0., 0., 0., 0., 0., 0.,]
    ])

    a_secondary = np.array([
        [0., 0., 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483,  0.0896645],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [-0.0250794, 0.0709672, 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    
    # Calculate the height of the mirror surface at each (X, Y) coordinate using the mirror function with the provided coefficients
    Rn = 3000.0
    Z_primary = np.zeros_like(X)
    for i in range(a_primary.shape[0]):
        for j in range(a_primary.shape[1]):
            Z_primary += a_primary[i, j] * (X / Rn) ** i * (Y / Rn) ** j
    Z_secondary = np.zeros_like(X)
    for i in range(a_secondary.shape[0]):
        for j in range(a_secondary.shape[1]):
            Z_secondary += a_secondary[i, j] * (X / Rn) ** i * (Y / Rn) ** j

    mirror_choice = 'primary'

    if mirror_choice == 'primary':
      zee = Z_primary
    else:
      zee = Z_secondary
    
    return zee