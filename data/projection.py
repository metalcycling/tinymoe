# %% Modules

import numpy as np
import scipy as sp
import scipy.optimize as spopt

from data.functions import polynomial, squared_distance

# %% Functions

def find_projection(point, coeff, guess=None):
    """
    Find the projection of 'point' onto the polynomial function defined by 'coeff'
    """

    zeroth_derivative = lambda x: squared_distance(x, point, coeff, derivative=0)
    first_derivative  = lambda x: squared_distance(x, point, coeff, derivative=1)
    second_derivative = lambda x: squared_distance(x, point, coeff, derivative=2)

    optimizer = spopt.minimize(zeroth_derivative, guess if guess else point[0], jac=first_derivative, hess=second_derivative, method="Newton-CG")

    x = optimizer.x[0]
    y = polynomial(x, coeff)[0]

    return x, y


# %% End of script
