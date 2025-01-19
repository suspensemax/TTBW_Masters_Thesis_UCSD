from functools import partial, partialmethod
from re import U
import numpy as np1
import jax.numpy as np
import numpy as onp
from jax import jit, jacfwd, lax
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
# from numba import jit

def eval_biot_savart(xcp0, xnode1, xnode2):
    """
    This function uses the Biot-Savart law to evaluate the induced velocities
    at control points (xcp), due to vortex line elements defined by locations xnode1, xnode2,
    with strengths gamma and lengths l0. The delta_visc parameter ensures velocity goes
    zero at the vortex line.

    ncp, nvor = number of control points / vortex line elements
    Inputs:
        xcp: (ncp, 3)
        xnode1, xnode2: (1, nvor, 3)
        gamma, l0: (nvor,)
        delta_visc: (1,)
    Returns
        u_gamma: (ncp, nvor, 3)
    """

    delta_visc=0.025
    l0 = np.ones(xnode1.shape[1]) * 1.

    xcp = xcp0.reshape(-1, 1, 3)  # xcp shape (ncp, 1, 3)
    # dim [1] of xcp is broadcast nvor times
    # dim [0] of xnode1/2 is broadcast ncp times
    r1 = xcp - xnode1  # r1 shape (ncp, nvor, 3)
    r2 = xcp - xnode2  # r2 shape (ncp, nvor, 3)

    r1_norm = np.sqrt(np.sum(r1 ** 2, axis=2))  # r1_norm shape = (ncp, nvl)
    r1_norm = r1_norm.reshape(r1_norm.shape + (1,))  # add 3rd dimension
    r2_norm = np.sqrt(np.sum(r2 ** 2, axis=2))  # r2_norm shape = (ncp, nvl)
    r2_norm = r2_norm.reshape(r2_norm.shape + (1,))  # add 3rd dimension

    cross_r1r2 = np.cross(r1, r2)
    dotr1r2 = np.sum(r1 * r2, axis=2)
    dotr1r2 = dotr1r2.reshape(dotr1r2.shape + (1,))  # add 3rd dimension
    r1r2 = r1_norm * r2_norm

    numer = (r1_norm + r2_norm) * cross_r1r2
    denom = 4 * math.pi * (r1r2 * (r1r2 + dotr1r2) )#+ (delta_visc * l0.reshape(1,-1,1)) ** 2)
    u_gamma = numer / denom
    # print('ugamma', u_gamma.shape)


    return u_gamma

if __name__ == "__main__":

    import numpy as onp
    from jax import grad
    from jax import jit
    from jax import jacfwd, jacrev

    def generate_simple_mesh(nx, ny):
        mesh = np.zeros((nx, ny, 3))
        mesh = mesh.at[:, :, 0].set(np.outer(np.arange(nx), np.ones(ny)))
        mesh = mesh.at[:, :, 1].set(np.outer(np.arange(ny), np.ones(nx)).T)
        
        return mesh

    n_wake_pts_chord = 2
    nc = 2
    ns = 3
    nc_v = 3
    ns_v = 4
    eval_pt_names = ['col']
    vortex_coords_names = ['vor']
    # eval_pt_shapes = [(nx, ny, 3)]
    # vortex_coords_shapes = [(nx, ny, 3)]

    eval_pt_shapes = [(nc, ns, 3), (nc, ns, 3)]
    vortex_coords_shapes = [(nc, ns, 3)]

    output_names = ['aic']



    vor_val = generate_simple_mesh(nc, ns)
    col_val = (0.25 * (vor_val[:-1, :-1, :] + vor_val[:-1, 1:, :] +
                      vor_val[1:, :-1, :] + vor_val[1:, 1:, :])).reshape(-1, 3)

    A = vor_val[:-1, :-1, :].reshape(1,-1, 3)
    B = vor_val[:-1, 1:, :].reshape(1,-1, 3)

    print(A.shape)
    print(col_val.shape)
    grad_col = jit(jacfwd(eval_biot_savart,argnums=[0,1,2]))
    grad_col_val = grad_col(col_val, A, B)
    print(grad_col_val[0].shape)

    # grad_A = jacfwd(eval_biot_savart,argnums=1)(col_val, A, B)
    # print(grad_A.shape)
    # grad_B = jacfwd(eval_biot_savart,argnums=2)(col_val, A, B)
    # print(grad_B.shape)
    import timeit
    print(timeit.timeit(lambda: jit(jacfwd(eval_biot_savart,argnums=[0,1,2])), number=1000))
    print(timeit.timeit(lambda: grad_col(col_val, A, B), number=1000))

    '''
    no_jit = jacfwd(eval_biot_savart,argnums=[0,1,2])
    print(timeit.timeit(lambda: no_jit(col_val, A, B), number=1000))
    '''
    
    # import pyvista as pv
    # x = vor_val[:, :, 0]
    # y = vor_val[:, :, 1]
    # z = vor_val[:, :, 2]
    # grid = pv.StructuredGrid(onp.array(x), onp.array(y), onp.array(z))
    # grid.plot(show_edges=True, line_width=3, color='w', show_scalar_bar=False)