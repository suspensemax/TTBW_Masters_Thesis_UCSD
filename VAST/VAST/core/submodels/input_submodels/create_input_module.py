from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class CreateACSatesModule(Model):
    def initialize(self):
        self.parameters.declare('v_inf', types=np.ndarray)
        self.parameters.declare('theta', types=np.ndarray)
        self.parameters.declare('num_nodes')

    def define(self):
        v_inf = self.parameters['v_inf']
        theta = self.parameters['theta']
        num_nodes = self.parameters['num_nodes']

        u_all = self.declare_variable('u', shape=(num_nodes, 1),val=v_inf)
        v_all = self.declare_variable('v', shape=(num_nodes, 1))
        w_all = self.declare_variable('w', shape=(num_nodes, 1))
        theta_all = self.declare_variable('theta', shape=(num_nodes, 1),val=theta)
        gamma_all = self.declare_variable('gamma', shape=(num_nodes, 1))
        psi_all = self.declare_variable('psi', shape=(num_nodes, 1))
        p_all = self.declare_variable('p', shape=(num_nodes, 1))
        q_all = self.declare_variable('q', shape=(num_nodes, 1))
        r_all = self.declare_variable('r', shape=(num_nodes, 1))
        x_all = self.declare_variable('x', shape=(num_nodes, 1))
        y_all = self.declare_variable('y', shape=(num_nodes, 1))
        z_all = self.declare_variable('z', shape=(num_nodes, 1))
        rho_all = self.declare_variable('density', shape=(num_nodes, 1))

        # print('theta',theta )

        dummy = self.register_output('dummy', u_all*theta)
        # a = self.re

