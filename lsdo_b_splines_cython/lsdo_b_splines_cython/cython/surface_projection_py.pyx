import numpy as np
cimport numpy as np

from lsdo_b_splines_cython.cython.surface_projection cimport compute_surface_projection


def compute_surface_projection(
    np.ndarray[int] surfs_order_u, np.ndarray[int] surfs_num_coefficients_u,
    np.ndarray[int] surfs_order_v, np.ndarray[int] surfs_num_coefficients_v,
    int num_points, int max_iter,
    np.ndarray[double] pts,  np.ndarray[double] cps,
    np.ndarray[double] knot_vector_u, np.ndarray[double] knot_vector_v,
    np.ndarray[double] u_vec, np.ndarray[double] v_vec,
    int guess_grid_n,
    np.ndarray[double] axis,
    np.ndarray[int] surfs_index,
    int num_surfs,
):
    compute_surface_projection(
        &surfs_order_u[0], &surfs_num_coefficients_u[0],
        &surfs_order_v[0], &surfs_num_coefficients_v[0],
        num_points, max_iter,
        &pts[0], &cps[0],
        &knot_vector_u[0], &knot_vector_v[0],
        &u_vec[0], &v_vec[0],
        guess_grid_n,
        &axis[0],
        &surfs_index[0],
        num_surfs,
        #&u_vec_initial[0], &v_vec_initial[0],
    )