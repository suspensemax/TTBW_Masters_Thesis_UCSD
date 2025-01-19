import m3l 
import csdl
import numpy as np
from lsdo_airfoil.utils.get_airfoil_model import get_airfoil_models
from lsdo_airfoil.core.airfoil_models import (
    ClModel, CdModel, CmModel, InverseCl, CpLowerModel, CpUpperModel, 
    DeltaStarLowerModel, DeltaStarUpperModel, ThetaLowerModel, ThetaUpperModel,
    EdgeVelocityLowerModel, EdgeVelocityUpperModel
)

X_min_numpy_prestall = np.array([
                -3.87714524e-03, -6.21114345e-03, -3.65010835e-03, -5.48448414e-04,
                1.04316720e-03, -3.44090629e-04,  -1.91351119e-03, -2.22159643e-03,
                -2.74770800e-03, -4.31647711e-03, -5.94483502e-03, -9.50526167e-03,
                -1.18035562e-02, -1.22448076e-02, -8.20550136e-03, -3.83688067e-03,
                -2.66918913e-04,  5.67624951e-03,  1.92390252e-02,  2.55834870e-02,
                3.14692594e-02,  3.43126804e-02,   3.81270386e-02,  4.34582904e-02,
                4.47864607e-02,  4.02273424e-02,   3.80498208e-02,  2.97566336e-02,
                2.03249976e-02,  1.10881981e-02,   4.26956685e-03, -5.45227900e-04,
                -8.00000000e+00,  1.00000000e+05,  0.00000000e+00]
            )


X_max_numpy_prestall = np.array([
                1.64971128e-03, 5.25048282e-03, 1.47131169e-02, 3.03167850e-02,
                4.73764949e-02, 6.15255609e-02, 7.35139325e-02, 8.21573734e-02,
                8.81158486e-02, 9.02919322e-02, 8.93072858e-02, 8.19384754e-02,
                7.00145736e-02, 5.29626682e-02, 3.25598940e-02, 1.39800459e-02,
                1.76265929e-02, 4.02436182e-02, 7.17813671e-02, 1.06165685e-01,
                1.40150547e-01, 1.67483926e-01, 1.88060194e-01, 2.04852015e-01,
                2.15405628e-01, 2.26217642e-01, 2.21330181e-01, 1.99092031e-01,
                1.54896125e-01, 1.00657888e-01, 4.71989214e-02, 1.23437112e-02,
                17, 8.00000000e+06, 6.00000000e-01]
                )


X_max_numpy_poststall = np.array([
                1.64971128e-03, 5.25048282e-03, 1.47131169e-02, 3.03167850e-02,
                4.73764949e-02, 6.15255609e-02, 7.35139325e-02, 8.21573734e-02,
                8.81158486e-02, 9.02919322e-02, 8.93072858e-02, 8.19384754e-02,
                7.00145736e-02, 5.29626682e-02, 3.25598940e-02, 1.39800459e-02,
                1.76265929e-02, 4.02436182e-02, 7.17813671e-02, 1.06165685e-01,
                1.40150547e-01, 1.67483926e-01, 1.88060194e-01, 2.04852015e-01,
                2.15405628e-01, 2.26217642e-01, 2.21330181e-01, 1.99092031e-01,
                1.54896125e-01, 1.00657888e-01, 4.71989214e-02, 1.23437112e-02,
                90., 8.00000000e+06, 6.00000000e-01]
                )

X_min_numpy_poststall = np.array([
                -3.87714524e-03, -6.21114345e-03, -3.65010835e-03, -5.48448414e-04,
                1.04316720e-03, -3.44090629e-04,  -1.91351119e-03, -2.22159643e-03,
                -2.74770800e-03, -4.31647711e-03, -5.94483502e-03, -9.50526167e-03,
                -1.18035562e-02, -1.22448076e-02, -8.20550136e-03, -3.83688067e-03,
                -2.66918913e-04,  5.67624951e-03,  1.92390252e-02,  2.55834870e-02,
                3.14692594e-02,  3.43126804e-02,   3.81270386e-02,  4.34582904e-02,
                4.47864607e-02,  4.02273424e-02,   3.80498208e-02,  2.97566336e-02,
                2.03249976e-02,  1.10881981e-02,   4.26956685e-03, -5.45227900e-04,
                -90.00000000e+00,  1.00000000e+05,  0.00000000e+00]
            )

class SMLAirfoil(m3l.ExplicitOperation):
    """
    Subsonic machine learning arifoil operation.

    Assembles all submodels to predict
    - Pressure coefficient
    - Displacement thickness 
    - Momentum thickness
    - Edge velocity
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='subsonic_airfoil_model')
        self.parameters.declare('num_nodes', types=int, default=1)

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.num_nodes = self.parameters['num_nodes']

    def compute(self):
        parent_model = csdl.Model()

        cp_model = CpModelCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(cp_model, 'cp_model', promotes=[])

        dstar_model = DeltaStarCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(dstar_model, 'dstar_model', promotes=[])

        theta_model = ThetaCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(theta_model, 'theta_model', promotes=[])

        edge_velocity_model = EdgeVelocityCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(edge_velocity_model, 'edge_velocity_model', promotes=[])
        
        cl_model = ClModelCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(cl_model, 'cl_model', promotes=[])

        cd_model = CdModelCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(cd_model, 'cd_model', promotes=[])

        cm_model = CmModelCSDL(
            num_nodes=self.num_nodes,
        )
        parent_model.add(cm_model, 'cm_model', promotes=[])

        parent_model.connect('mach_number', 'cp_model.mach_number')
        parent_model.connect('reynolds_number', 'cp_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'cp_model.angle_of_attack')
        parent_model.connect('control_points', 'cp_model.control_points')

        parent_model.connect('mach_number', 'dstar_model.mach_number')
        parent_model.connect('reynolds_number', 'dstar_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'dstar_model.angle_of_attack')
        parent_model.connect('control_points', 'dstar_model.control_points')

        parent_model.connect('mach_number', 'theta_model.mach_number')
        parent_model.connect('reynolds_number', 'theta_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'theta_model.angle_of_attack')
        parent_model.connect('control_points', 'theta_model.control_points')

        parent_model.connect('mach_number', 'edge_velocity_model.mach_number')
        parent_model.connect('reynolds_number', 'edge_velocity_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'edge_velocity_model.angle_of_attack')
        parent_model.connect('control_points', 'edge_velocity_model.control_points')

        parent_model.connect('mach_number', 'cl_model.mach_number')
        parent_model.connect('reynolds_number', 'cl_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'cl_model.angle_of_attack')
        parent_model.connect('control_points', 'cl_model.control_points')

        parent_model.connect('mach_number', 'cd_model.mach_number')
        parent_model.connect('reynolds_number', 'cd_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'cd_model.angle_of_attack')
        parent_model.connect('control_points', 'cd_model.control_points')

        parent_model.connect('mach_number', 'cm_model.mach_number')
        parent_model.connect('reynolds_number', 'cm_model.reynolds_number')
        parent_model.connect('angle_of_attack', 'cm_model.angle_of_attack')
        parent_model.connect('control_points', 'cm_model.control_points')

        return parent_model




class CpModelCSDL(csdl.Model):
    """
    Pressure coefficient CSDL model
    """
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(vector_valued_models=['Cp'])
        num_nodes = self.parameters['num_nodes']    
        control_points = self.declare_variable('control_points', shape=(32, 1))

        
        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall_cp',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall_cp',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs_cp', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_prestall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input_cp', scaled_inputs_prestall)

        cp_upper = csdl.custom(x, op=CpUpperModel(
                neural_net=neural_net_dict['Cp_upper'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('cp_upper', cp_upper)

        cp_lower = csdl.custom(x, op=CpLowerModel(
                neural_net=neural_net_dict['Cp_lower'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('cp_lower', cp_lower)


class DeltaStarCSDL(csdl.Model):
    """
    Displacement thickness CSDL model
    """
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(vector_valued_models=['delta_star'])
        num_nodes = self.parameters['num_nodes']    
        control_points = self.declare_variable('control_points', shape=(32, 1))

        
        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall_dstar',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall_dstar',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs_dstar', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_prestall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input_dstar', scaled_inputs_prestall)

        dstar_upper = csdl.custom(x, op=DeltaStarUpperModel(
                neural_net=neural_net_dict['delta_star_upper'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('dstar_upper', dstar_upper)

        dstar_lower = csdl.custom(x, op=DeltaStarLowerModel(
                neural_net=neural_net_dict['delta_star_lower'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('dstar_lower', dstar_lower)


class ThetaCSDL(csdl.Model):
    """
    Momentum thickness csdl model
    """
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(vector_valued_models=['theta'])
        num_nodes = self.parameters['num_nodes']    
        control_points = self.declare_variable('control_points', shape=(32, 1))

        
        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall_theta',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall_theta',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs_theta', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_prestall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input_theta', scaled_inputs_prestall)

        theta_upper = csdl.custom(x, op=ThetaUpperModel(
                neural_net=neural_net_dict['theta_upper'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('theta_upper', theta_upper)

        theta_lower = csdl.custom(x, op=ThetaLowerModel(
                neural_net=neural_net_dict['theta_lower'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('theta_lower', theta_lower)


class EdgeVelocityCSDL(csdl.Model):
    """
    Edge velocity thickness csdl model
    """
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(vector_valued_models=['Ue'])
        num_nodes = self.parameters['num_nodes']    
        control_points = self.declare_variable('control_points', shape=(32, 1))

        
        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall_edge_velocity',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall_edge_velocity',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs_edge_velocity', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_prestall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input_edge_velocity', scaled_inputs_prestall)

        edge_velocity_upper = csdl.custom(x, op=EdgeVelocityUpperModel(
                neural_net=neural_net_dict['Ue_upper'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('edge_velocity_upper', edge_velocity_upper)

        edge_velocity_lower = csdl.custom(x, op=EdgeVelocityLowerModel(
                neural_net=neural_net_dict['Ue_lower'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('edge_velocity_lower', edge_velocity_lower)


class ClModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(scaler_valued_models=['Cl'])
        num_nodes = self.parameters['num_nodes']
        control_points = self.declare_variable('control_points', shape=(32, 1))

        X_min_poststall = csdl.expand(self.declare_variable(
            name='X_min_poststall',    
            val=X_min_numpy_poststall
        ), (num_nodes, 35), 'i->ji')


        X_max_poststall = csdl.expand(self.declare_variable(
            name='X_max_poststall',
            val=X_max_numpy_poststall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_poststall = (inputs - X_min_poststall) / (X_max_poststall - X_min_poststall)
        x_extrap = self.register_output('neural_net_input_extrap', scaled_inputs_poststall)

        cl = csdl.custom(x_extrap, op=ClModel(
                neural_net=neural_net_dict['Cl'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('Cl', cl)



class CdModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(scaler_valued_models=['Cd'])
        num_nodes = self.parameters['num_nodes']
        control_points = self.declare_variable('control_points', shape=(32, 1))

        X_min_poststall = csdl.expand(self.declare_variable(
            name='X_min_poststall',    
            val=X_min_numpy_poststall
        ), (num_nodes, 35), 'i->ji')


        X_max_poststall = csdl.expand(self.declare_variable(
            name='X_max_poststall',
            val=X_max_numpy_poststall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_poststall = (inputs - X_min_poststall) / (X_max_poststall - X_min_poststall)
        x_extrap = self.register_output('neural_net_input_extrap', scaled_inputs_poststall)

        cd = csdl.custom(x_extrap, op=CdModel(
                neural_net=neural_net_dict['Cd'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('Cd', cd)

class CmModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        
        neural_net_dict = get_airfoil_models(scaler_valued_models=['Cm'])
        num_nodes = self.parameters['num_nodes']
        control_points = self.declare_variable('control_points', shape=(32, 1))

        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 

        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))

        scaled_inputs_poststall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input', scaled_inputs_poststall)

        cm = csdl.custom(x, op=CmModel(
                neural_net=neural_net_dict['Cm'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('Cm', cm)