from lsdo_modules.module.module import Module
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import numpy as np
from python_csdl_backend import Simulator
from csdl import GraphRepresentation


# Create csdl modules
class RadiusCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('scaling_factor')

    def define(self):
        s = self.parameters['scaling_factor']
        disk_area = self.register_module_input('disk_area', shape=(1, ))

        radius = (s * disk_area / np.pi)**0.5
        self.register_module_output('radius', radius, importance=1)

class DoubleNestedDummy(ModuleCSDL):
    def define(self):
        dd = self.register_module_input('dummy_input_2', val=10)
        self.register_module_output('dummy_output_2', dd**2, importance=1)

class NestedDummy(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('something')
    def define(self):
        something = self.parameters['something']
        d = self.register_module_input('dummy_input', val=20)
        self.register_module_output('dummy_output', d**2, importance=1)

        double_nested_dummy = DoubleNestedDummy()
        self.add_module(double_nested_dummy, 'triple_nested_dummy_module')

class DummyCSDL(ModuleCSDL):
    def define(self):
        T = self.register_module_input('thrust', shape=(1, ))
        self.register_module_output('eta', 0.9*T/T, importance=1)

        dummy_module =  NestedDummy(
            something='something'
        )
        self.add_module(dummy_module, 'double_nested_dummy_module')

class ThrustCSDL(ModuleCSDL):
    def initialize(self): 
        self.parameters.declare('rho') 

    def define(self):
        rho = self.parameters['rho']

        R = self.register_module_input('radius', shape=(1, ))
        C_T = self.register_module_input('C_T', shape=(1, ))
        rpm = self.register_module_input('rpm', shape=(1, ))
        
        n = rpm / 60
        D = 2 * R

        T = C_T * rho * n**2 * D**4
        self.register_module_output('thrust', T, importance=1)

        dummy_model = DummyCSDL()
        self.add_module(dummy_model, 'nested_dummy_module')



class ThrustComputationCSDL(ModuleCSDL):
    def initialize(self): 
        self.parameters.declare('scaling_factor')
        self.parameters.declare('rho') 

    def define(self):
        scaling_factor = self.parameters['scaling_factor']
        rho = self.parameters['rho']

        # self.register_module_input('disk_area', shape=(1, ))
        radius_module_csdl = RadiusCSDL(
            module=self.module,
            scaling_factor=scaling_factor,
        )
        self.add_module(radius_module_csdl, 'radius_module')

        # C_T = self.register_module_input('C_T', shape=(1, ))
        # rpm = self.register_module_input('rpm', shape=(1, ))
        thrust_module_csdl = ThrustCSDL(
            module=self.module,
            sub_modules=self.sub_modules,
            rho=rho,
            name='thrust_module',
        )
        self.add_module(thrust_module_csdl, 'thrust_module')

        thrust = self.register_module_input('thrust',  shape=(1, ))
        self.register_module_output('T', thrust**0.9)

        test_input = self.register_module_input('new_input')


# define your module (pure python object)
class ThrustModule(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def assemble_csdl(self):
        csdl_model = ThrustComputationCSDL(
            module=self,
            scaling_factor=1.5,
            rho=1.2,
            name='thrust_example',
        )
        graph = GraphRepresentation(csdl_model)
        return csdl_model



# set up the simulation 
thrust_module = ThrustModule()
thrust_module.set_module_input('disk_area', val=4.)
# thrust_module.set_module_input('C_T', val=0.25)
thrust_module.set_module_input('rpm', val=1200)
thrust_module.set_module_input('new_input', val=12)

thrust_module_csdl = thrust_module.assemble_csdl()
# print('\n')
# print('module_inputs',thrust_module_csdl.module_inputs)
# print('module_outputs',thrust_module_csdl.module_outputs)
# print('sub_modules', thrust_module_csdl.sub_modules['radius_module']['inputs'])
# print('sub_modules', thrust_module_csdl.sub_modules['thrust module']['inputs'])
# print('sub_module_inputs',thrust_module_csdl.sub_modules[1].module_inputs)
# print('module_inputs', thrust_module_csdl.module_outputs)
# graph = GraphRepresentation(thrust_module_csdl)

# thrust_module_csdl.visualize_implementation(importance=5)
# exit()
# print(graph.module)
sim = Simulator(thrust_module_csdl)
print('module_inputs',thrust_module_csdl.sub_modules)
# exit()
sim.run()
# print(thrust_module_csdl.module.promoted_vars)
print('module_outputs',thrust_module_csdl.module_outputs)
print('Promoted vars', thrust_module_csdl.promoted_vars)
print('thrust', sim['thrust'])

