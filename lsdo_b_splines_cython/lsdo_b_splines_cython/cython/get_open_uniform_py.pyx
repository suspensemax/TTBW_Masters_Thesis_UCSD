import numpy as np
cimport numpy as np

from lsdo_b_splines_cython.cython.get_open_uniform cimport get_open_uniform

def get_open_uniform(int order, int num_coefficients, np.ndarray[double] knot_vector):
  get_open_uniform(order, num_coefficients, &knot_vector[0])