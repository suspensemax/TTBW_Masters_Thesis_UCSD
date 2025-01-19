import numpy as np


class RadialBasisFunctions:
    def Gaussian(x_dist, eps=1.):
        return np.exp(-(eps*x_dist)**2)

    def BumpFunction(x_dist, eps=1.):
        # filter x_dist to get rid of x_dist >= 1/eps, this prevents overflow warnings
        x_dist_filt = np.where(x_dist < 1/eps, x_dist, 0.)
        f_mat = np.where(x_dist < 1/eps, np.exp(-1./(1.-(eps*x_dist_filt)**2)), 0.)
        return f_mat/np.exp(-1)  # normalize output so x_dist = 0 corresponds to f = 1
    
    def ThinPlateSpline(x_dist):
        return np.multiply(np.power(x_dist, 2), np.log(x_dist))

class NodalMap:
    def __init__(self, solid_nodal_mesh, fluid_nodal_mesh, RBF_width_par=np.inf, RBF_func=RadialBasisFunctions.Gaussian, column_scaling_vec=None):
        self.solid_nodal_mesh = solid_nodal_mesh
        self.fluid_nodal_mesh = fluid_nodal_mesh
        self.RBF_width_par = RBF_width_par
        self.RBF_func = RBF_func
        if column_scaling_vec is not None:
            self.column_scaling_vec = column_scaling_vec
        else:
            self.column_scaling_vec = np.ones((solid_nodal_mesh.shape[0],))

        self.map_shape = [self.fluid_nodal_mesh.shape[0], self.solid_nodal_mesh.shape[0]]
        self.distance_matrix = self.compute_distance_matrix()
        self.map = self.construct_map()
    
    def compute_distance_matrix(self):
        coord_dist_mat = np.zeros((self.map_shape + [3]))
        for i in range(3):
            # print('------------------------------------------------------------------------',self.fluid_nodal_mesh.shape)
            # print(self.solid_nodal_mesh[:, i])
            try:
                coord_dist_mat[:, :, i] = NodalMap.coord_diff(self.fluid_nodal_mesh.value[:, i], self.solid_nodal_mesh.value[:, i])
            except:
                coord_dist_mat[:, :, i] = NodalMap.coord_diff(self.fluid_nodal_mesh[:, i], self.solid_nodal_mesh[:, i])

        coord_dist_mat = NodalMap.compute_pairwise_Euclidean_distance(coord_dist_mat)
        return coord_dist_mat
    
    def construct_map(self):
        influence_coefficients = self.RBF_func(self.distance_matrix, eps=self.RBF_width_par)

        # influence_coefficients = np.multiply(influence_coefficients, influence_dist_below_max_mask)

        # include influence of column scaling
        influence_coefficients = np.einsum('ij,j->ij', influence_coefficients, self.column_scaling_vec)

        # sum influence coefficients in each row and normalize the coefficients with those row sums
        inf_coeffs_per_row = np.sum(influence_coefficients, axis=1)
        normalized_inf_coeff_map = np.divide(influence_coefficients, inf_coeffs_per_row[:, None])
        return normalized_inf_coeff_map

    @staticmethod
    def coord_diff(arr_1, arr_2):
        # subtracts arr_1 and arr_2 of different sizes in such a way that the result is a matrix of size (arr_1.shape[0], arr_2.shape[0])
        return np.subtract(arr_1[:, None], arr_2[None, :])

    @staticmethod
    def compute_pairwise_Euclidean_distance(coord_dist_mat):
        coord_dist_mat_sqrd = np.power(coord_dist_mat, 2)
        coord_dist_mat_summed = np.sum(coord_dist_mat_sqrd, axis=2)
        return np.sqrt(coord_dist_mat_summed)