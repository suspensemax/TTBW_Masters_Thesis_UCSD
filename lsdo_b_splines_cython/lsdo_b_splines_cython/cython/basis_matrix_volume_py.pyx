import numpy as np
cimport numpy as np

from lsdo_b_splines_cython.cython.basis_matrix_volume cimport get_basis_volume_matrix


def get_basis_volume_matrix(
        int order_u, int num_coefficients_u, int u_der, np.ndarray[double] u_vec, np.ndarray[double] knot_vector_u,
        int order_v, int num_coefficients_v, int v_der, np.ndarray[double] v_vec, np.ndarray[double] knot_vector_v,
        int order_w, int num_coefficients_w, int w_der, np.ndarray[double] w_vec, np.ndarray[double] knot_vector_w,
        int num_points, 
        np.ndarray[double] data, np.ndarray[int] row_indices, np.ndarray[int] col_indices):
    get_basis_volume_matrix(
        order_u, num_coefficients_u, u_der, &u_vec[0], &knot_vector_u[0],
        order_v, num_coefficients_v, v_der, &v_vec[0], &knot_vector_v[0],
        order_w, num_coefficients_w, w_der, &w_vec[0], &knot_vector_w[0],
        num_points, &data[0], &row_indices[0], &col_indices[0],
    )
    