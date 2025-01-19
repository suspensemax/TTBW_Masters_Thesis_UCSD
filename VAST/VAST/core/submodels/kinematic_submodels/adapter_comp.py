# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
# from fluids import atmosphere as atmosphere
# from lsdo_atmos.atmosphere_model import AtmosphereModel


class AdapterComp(Model):
    """
    An adapter component that takes in 15 variables from CADDEE (not all are used), 
    and adaptes in to frame_vel(linear without rotation),
    rotational velocity, and air density rho.

    parameters
    ----------
    u[num_nodes,1] : csdl array
        vx of the body
    v[num_nodes,1] : csdl array
        vy of the body
    w[num_nodes,1] : csdl array
        vz of the body

    p[num_nodes,1] : csdl array
        omega_x of the body
    q[num_nodes,1] : csdl array
        omega_y of the body
    r[num_nodes,1] : csdl array
        omega_z of the body

    phi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: p=\dot{phi}
    theta[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: q=\dot{theta}
    psi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: r=\dot{psi}

    x[num_nodes,1] : csdl array
        omega_x of the body
    y[num_nodes,1] : csdl array
        omega_y of the body
    z[num_nodes,1] : csdl array
        omega_z of the body

    phiw[num_nodes,1] : csdl array
        omega_x of the body
    gamma[num_nodes,1] : csdl array
        omega_y of the body
    psiw[num_nodes,1] : csdl array
        omega_z of the body    

    collocation points

    Returns
    -------
    1. frame_vel[num_nodes,3] : csdl array
        inertia frame vel
    2. alpha[num_nodes,1] : csdl array
        AOA in rad
    3. v_inf_sq[num_nodes,1] : csdl array
        square of v_inf in rad
    4. beta[num_nodes,1] : csdl array
        sideslip angle in rad
    5. rho[num_nodes,1] : csdl array
        air density

    # s_panel [(num_pts_chord-1)* (num_pts_span-1)]: csdl array
    #     The panel areas.
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        num_nodes = surface_shapes[0][0]

        u = self.declare_variable('u', shape=(num_nodes, 1))
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))
        # wing_incidence_angle = csdl.expand(self.declare_variable('wing_incidence', shape=(1, )), shape=(num_nodes, 1))

        p = self.declare_variable('p', shape=(num_nodes, 1))
        q = self.declare_variable('q', shape=(num_nodes, 1))
        r = self.declare_variable('r', shape=(num_nodes, 1))

        phi = self.declare_variable('phi', shape=(num_nodes, 1))
        self.register_output('dummy_phi', phi*1)
        theta = self.declare_variable('theta', shape=(num_nodes, 1))
        psi = self.declare_variable('psi', shape=(num_nodes, 1))

        # x = self.declare_variable('x', shape=(num_nodes, 1))
        # y = self.declare_variable('y', shape=(num_nodes, 1))
        # z = self.declare_variable('z', shape=(num_nodes, 1))

        # phiw = self.declare_variable('phiw', shape=(num_nodes, 1))
        # gamma = self.declare_variable('gamma', shape=(num_nodes, 1), val=0)
        # psiw = self.declare_variable('psiw', shape=(num_nodes, 1), val=0)
        self.register_output('dummy_output', theta * psi)# * psiw)

        # self.print_var(u)

        ################################################################################
        # compute the output: 3. v_inf_sq (num_nodes,1)
        ################################################################################
        v_inf_sq = (u**2 + v**2 + w**2)
        v_inf = (u**2 + v**2 + w**2)**0.5
        self.register_output('v_inf_sq', v_inf_sq)

        ################################################################################
        # compute the output: 3. alpha (num_nodes,1)
        ################################################################################
        alpha = csdl.arctan(-w/ u) # theta - gamma
        self.register_output('alpha', alpha)
        # self.print_var(alpha)

        ################################################################################
        # compute the output: 4. beta (num_nodes,1)
        ################################################################################
        # beta = csdl.arccos(u/ v_inf) # psi + psiw
        beta = csdl.arcsin(v/ v_inf) # psi + psiw
        # we always assume v_inf > 0 here
        self.register_output('beta', beta)
        # self.print_var(beta)

        gamma = csdl.arctan(w/v)
        self.register_output('gamma', gamma)
        
        ################################################################################
        # create the output: 1. frame_vel (num_nodes,3)
        # TODO:fix this
        ################################################################################


        frame_vel = self.create_output('frame_vel', shape=(num_nodes, 3))

        frame_vel[:, 0] = -1 * u #-v_inf * csdl.cos(beta) * csdl.cos(alpha)
        frame_vel[:, 1] = 1 * v# v_inf * csdl.sin(beta) * csdl.cos(alpha)

        # frame_vel[:, 2] = -v_inf * csdl.cos(beta) * csdl.sin(alpha)
        frame_vel[:, 2] = 1 * w #-v_inf * csdl.sin(alpha)
        # self.print_var(frame_vel)
        ################################################################################
        # compute the output: 5. rho
        # TODO: replace this hard coding
        ################################################################################
        # h = 1000
        # atmosisa = atmosphere.ATMOSPHERE_1976(Z=h)
        # rho_val = atmosisa.rho

        # self.add(AtmosphereModel(
        #     shape=(num_nodes,1),
        # ),name='atmosphere_model')

        self.declare_variable('density', val=1.*np.ones((num_nodes,1)))

        # self.create_input('rho', val=(num_nodes, 1))


