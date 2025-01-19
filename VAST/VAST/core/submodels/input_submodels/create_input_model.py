from csdl import Model
import numpy as np

class CreateACSatesModel(Model):
    '''
    Create ACS states from the ACS inputs

    parameters
    ----------
    u[num_nodes, 1] : csdl array
    v[num_nodes, 1] : csdl array
    w[num_nodes, 1] : csdl array
    p[num_nodes, 1] : csdl array
    q[num_nodes, 1] : csdl array
    r[num_nodes, 1] : csdl array
    phi[num_nodes, 1] : csdl array
    theta[num_nodes, 1] : csdl array
    psi[num_nodes, 1] : csdl array
    x[num_nodes, 1] : csdl array
    y[num_nodes, 1] : csdl array
    z[num_nodes, 1] : csdl array
    phiw[num_nodes, 1] : csdl array
    gamma[num_nodes, 1] : csdl array
    psiw[num_nodes, 1] : csdl array
    '''
    def initialize(self):
        self.parameters.declare('v_inf', types=np.ndarray)
        self.parameters.declare('theta', types=np.ndarray)
        self.parameters.declare('num_nodes')

    def define(self):
        v_inf = self.parameters['v_inf']
        theta = self.parameters['theta']
        num_nodes = self.parameters['num_nodes']

        u = self.create_input('u',val=v_inf)
        v = self.create_input('v',val=np.zeros((num_nodes, 1)))
        w = self.create_input('w',val=np.ones((num_nodes,1))*0)
        p = self.create_input('p',val=np.zeros((num_nodes, 1)))
        q = self.create_input('q',val=np.zeros((num_nodes, 1)))
        r = self.create_input('r',val=np.zeros((num_nodes, 1)))
        phi = self.create_input('phi',val=np.zeros((num_nodes, 1)))
        theta = self.create_input('theta',val=theta)
        psi = self.create_input('psi',val=np.zeros((num_nodes, 1)))
        x = self.create_input('x',val=np.zeros((num_nodes, 1)))
        y = self.create_input('y',val=np.zeros((num_nodes, 1)))
        z = self.create_input('z',val=np.ones((num_nodes, 1))*1000)
        phiw = self.create_input('phiw',val=np.zeros((num_nodes, 1)))
        gamma = self.create_input('gamma',val=np.zeros((num_nodes, 1)))
        psiw = self.create_input('psiw',val=np.zeros((num_nodes, 1)))