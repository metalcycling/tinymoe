# %% Modules

import numpy as np

# %% Functions

def polynomial(x, coeff):
    """
    Polynomial function defined by its coefficients
    """
    order = len(coeff) - 1
    coeff = coeff[:, np.newaxis]
    indices = np.arange(order + 1)[::-1][:, np.newaxis]
    powers = np.array([x ** i for i in indices])

    return np.sum(coeff * powers, axis=0)

def squared_distance(x, point, coeff, derivative=0):
    """
    Function for the squared-distance between a point and a polynomial
    """

    assert derivative <= 2, "ERROR: Only 0th, 1st, and 2nd derivatives are supported"

    order = len(coeff) - 1
    coeff = coeff[:, np.newaxis]
    indices = np.arange(order + 1)[::-1][:, np.newaxis]
    powers = np.array([x ** i for i in indices])

    if derivative == 0:
        value  = (x - point[0]) ** 2 
        value += (np.sum(coeff * powers, axis=0) - point[1]) ** 2

    elif derivative == 1:
        value  = 2.0 * (x - point[0])
        value += 2.0 * (np.sum(coeff * powers, axis=0) - point[1]) * np.sum(indices[:-1] * coeff[:-1] * powers[1:], axis=0)

    elif derivative == 2:
        value  = 2.0
        value += 2.0 * (np.sum(indices[:-1] * coeff[:-1] * powers[1:], axis=0) ** 2)
        value += 2.0 * (np.sum(coeff * powers, axis=0) - point[1]) * (np.sum(indices[:-2] * (indices[:-2] - 1.0) * coeff[:-2] * powers[2:], axis=0))

    return value

# %% End of script
