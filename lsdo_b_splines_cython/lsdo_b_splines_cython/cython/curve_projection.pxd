from libc.stdlib cimport malloc, free

from lsdo_b_splines_cython.cython.get_open_uniform cimport get_open_uniform
from lsdo_b_splines_cython.cython.basis0 cimport get_basis0
from lsdo_b_splines_cython.cython.basis1 cimport get_basis1
from lsdo_b_splines_cython.cython.basis2 cimport get_basis2
from lsdo_b_splines_cython.cython.basis_matrix_curve cimport get_basis_curve_matrix


cdef compute_curve_projection(
    int order_u, int num_coefficients_u,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* u_vec, double* knot_vector,
    int n_guesses,
)