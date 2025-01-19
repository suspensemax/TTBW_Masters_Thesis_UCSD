import m3l 
import csdl
from csdl import Model
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
# from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.TBWminuswingCSDL import TBWminuswingCSDL
from caddee.utils.helper_classes import MassProperties


class TBWMassProperties_m3l(m3l.ExplicitOperation):
    def initialize(self, kwargs): 
        self.parameters.declare('name', types=str)
        self.parameters.declare('exclude_wing', types=bool, default=False)
        self.parameters.declare('full_wing', types=bool, default=False)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        exclude_wing = self.parameters['exclude_wing']
        full_wing = self.parameters['full_wing']
        if exclude_wing==True:
            csdl_model = TBWMassPropertiesMinusWingCSDL()
        else:    
            if full_wing==True:
                csdl_model = TBWMassPropertiesFullCSDL()
            else: 
                csdl_model = TBWMassPropertiesOnlyWingCSDL()
        return csdl_model
    
    def evaluate(self):
        self.arguments = {}
        # if design_condition:
        #     self.name = f"{design_condition.parameters['name']}_tbw_weight"
        # else:
        #     self.name = 'tbw_weight'
        mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable(name='cg_vector', shape=(3,), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3, 3), operation=self)

        outputs = MassProperties(
            mass=mass,
            cg_vector=cg_vector,
            inertia_tensor=inertia_tensor,
        )
        return outputs


class TBWMassPropertiesMinusWingCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('name', default='TBW_minus_wing_sizing', types=str)

    def define(self):
        shape = (1,)
        
        # Unit conversions
        ft_2_to_m_2 = 1 / 10.764
        ft2m = 0.3048
        ktas_to_m_s = 1 / 1.944
        lbs_to_kg =  1 / 2.205
        lbs_sq_ft_to_kg_sq_m = 0.0421401101
        # lbs_sq_ft_to_kg_sq_m = 1.

        wing_mass = self.declare_variable('wing_mass', shape=shape, val=16740.*lbs_to_kg)
        wing_strut_mass = self.declare_variable('wing_strut_mass', shape=shape, val=3680.*lbs_to_kg)

        total_mass = self.declare_variable('total_mass', shape=shape, val=150000.*lbs_to_kg)
        total_mass_without_wing_strut = total_mass - (wing_mass + wing_strut_mass)        
        self.register_output(name = 'mass', var = total_mass_without_wing_strut)

        dummy = self.declare_variable('dummy_variable', shape=shape, val=0.)

        # Ixx = 412459.845 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_xx_a = self.declare_variable('moment_of_inertia_xx_a', shape=shape, val=423971.193*lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xx_without_wing = moment_of_inertia_xx_a + dummy 
        self.register_output(name = 'Ixx', var = moment_of_inertia_xx_without_wing)
        
        # Iyy = 13379933.707 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_yy_a = self.declare_variable('moment_of_inertia_yy', shape=shape, val=13598247.66* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yy_without_wing = moment_of_inertia_yy_a + dummy
        self.register_output(name = 'Iyy', var = moment_of_inertia_yy_without_wing)

        moment_of_inertia_zz_a = self.declare_variable('moment_of_inertia_zz', shape=shape, val=13464420.96 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_zz_without_wing = moment_of_inertia_zz_a + dummy
        self.register_output(name = 'Izz', var = moment_of_inertia_zz_without_wing)

        moment_of_inertia_xy_a = self.declare_variable('moment_of_inertia_xy', shape=shape, val=-2651.385 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xy_without_wing = moment_of_inertia_xy_a + dummy
        self.register_output(name = 'Ixy', var = moment_of_inertia_xy_without_wing)

        moment_of_inertia_yz_a = self.declare_variable('moment_of_inertia_yz', shape=shape, val=2097.03 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yz_without_wing = moment_of_inertia_yz_a + dummy
        self.register_output(name = 'Iyz', var = moment_of_inertia_yz_without_wing)

        moment_of_inertia_xz_a = self.declare_variable('moment_of_inertia_xz', shape=shape, val=592625.625 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xz_without_wing = moment_of_inertia_xz_a + dummy
        self.register_output(name = 'Ixz', var = moment_of_inertia_xz_without_wing)

        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(moment_of_inertia_xx_without_wing, (1, 1)) 
        inertia_tensor[1, 1] = csdl.reshape(moment_of_inertia_yy_without_wing, (1, 1)) 
        inertia_tensor[2, 2] = csdl.reshape(moment_of_inertia_zz_without_wing, (1, 1)) 
        inertia_tensor[0, 2] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[2, 0] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[1, 2] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[2, 1] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[0, 1] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        inertia_tensor[1, 0] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        
        cg_x_a = self.declare_variable('center_of_gravity_x', shape=shape, val=58.595*ft2m)
        # cg_x = 57.505*ft2m
        cg_x = cg_x_a + dummy
        # cg_y = 0.
        cg_y_a = self.declare_variable('center_of_gravity_y', shape=shape, val=0.*ft2m)
        cg_y = cg_y_a + dummy
        # cg_z = 0.253*ft2m
        cg_z_a = self.declare_variable('center_of_gravity_z', shape=shape, val=5.715*ft2m)
        cg_z = cg_z_a + dummy

        self.register_output(
            name='cgx', 
            var=cg_x)
        self.register_output(
            name='cgy',  
            var=cg_y)
        self.register_output(
            name='cgz', 
            var=cg_z)
        
        cg_vector = self.create_output('cg_vector', shape=(3, ), val=0)
        cg_vector[0] = cg_x
        cg_vector[1] = cg_y
        cg_vector[2] = cg_z
        

class TBWMassPropertiesFullCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('name', default='TBW_minus_wing_sizing', types=str)

    def define(self):
        shape = (1,)
        
        # Unit conversions
        ft_2_to_m_2 = 1 / 10.764
        ft2m = 0.3048
        ktas_to_m_s = 1 / 1.944
        lbs_to_kg =  1 / 2.205
        lbs_sq_ft_to_kg_sq_m = 0.0421401101
        # lbs_sq_ft_to_kg_sq_m = 1.

        wing_mass = self.declare_variable('wing_mass', shape=shape, val=16740.*lbs_to_kg)
        wing_strut_mass = self.declare_variable('wing_strut_mass', shape=shape, val=3680.*lbs_to_kg)

        total_mass_a = self.declare_variable('total_mass', shape=shape, val=150000.*lbs_to_kg)
        dummy = self.declare_variable('dummy_variable', shape=shape, val=0.)
        # total_mass_without_wing_strut = total_mass - (wing_mass + wing_strut_mass)  
        total_mass = total_mass_a + dummy
        print('1')      
        self.register_output(name = 'mass', var = total_mass)
        print('2')


        # Ixx = 412459.845 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_xx_a = self.declare_variable('moment_of_inertia_xx_a', shape=shape, val=1917490.407*lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xx_without_wing = moment_of_inertia_xx_a + dummy 
        self.register_output(name = 'Ixx', var = moment_of_inertia_xx_without_wing)
        
        # Iyy = 13379933.707 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_yy_a = self.declare_variable('moment_of_inertia_yy', shape=shape, val= 13654998.682* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yy_without_wing = moment_of_inertia_yy_a + dummy
        self.register_output(name = 'Iyy', var = moment_of_inertia_yy_without_wing)

        moment_of_inertia_zz_a = self.declare_variable('moment_of_inertia_zz', shape=shape, val=4963769.686 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_zz_without_wing = moment_of_inertia_zz_a + dummy
        self.register_output(name = 'Izz', var = moment_of_inertia_zz_without_wing)

        moment_of_inertia_xy_a = self.declare_variable('moment_of_inertia_xy', shape=shape, val= -2160.340 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xy_without_wing = moment_of_inertia_xy_a + dummy
        self.register_output(name = 'Ixy', var = moment_of_inertia_xy_without_wing)

        moment_of_inertia_yz_a = self.declare_variable('moment_of_inertia_yz', shape=shape, val= 1506.523* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yz_without_wing = moment_of_inertia_yz_a + dummy
        self.register_output(name = 'Iyz', var = moment_of_inertia_yz_without_wing)

        moment_of_inertia_xz_a = self.declare_variable('moment_of_inertia_xz', shape=shape, val=596126.454* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xz_without_wing = moment_of_inertia_xz_a + dummy
        self.register_output(name = 'Ixz', var = moment_of_inertia_xz_without_wing)

        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(moment_of_inertia_xx_without_wing, (1, 1)) 
        inertia_tensor[1, 1] = csdl.reshape(moment_of_inertia_yy_without_wing, (1, 1)) 
        inertia_tensor[2, 2] = csdl.reshape(moment_of_inertia_zz_without_wing, (1, 1)) 
        inertia_tensor[0, 2] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[2, 0] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[1, 2] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[2, 1] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[0, 1] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        inertia_tensor[1, 0] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        
        cg_x_a = self.declare_variable('center_of_gravity_x', shape=shape, val=58.595*ft2m)
        # cg_x = 57.505*ft2m
        cg_x = cg_x_a + dummy
        # cg_y = 0.
        cg_y_a = self.declare_variable('center_of_gravity_y', shape=shape, val=0.*ft2m)
        cg_y = cg_y_a + dummy
        # cg_z = 0.253*ft2m
        cg_z_a = self.declare_variable('center_of_gravity_z', shape=shape, val=5.715*ft2m)
        cg_z = cg_z_a + dummy

        self.register_output(
            name='cgx', 
            var=cg_x)
        self.register_output(
            name='cgy',  
            var=cg_y)
        self.register_output(
            name='cgz', 
            var=cg_z)
        
        cg_vector = self.create_output('cg_vector', shape=(3, ), val=0)
        cg_vector[0] = cg_x
        cg_vector[1] = cg_y
        cg_vector[2] = cg_z
        

class TBWMassPropertiesOnlyWingCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('name', default='TBW_minus_wing_sizing', types=str)

    def define(self):
        shape = (1,)
        
        # Unit conversions
        ft_2_to_m_2 = 1 / 10.764
        ft2m = 0.3048
        ktas_to_m_s = 1 / 1.944
        lbs_to_kg =  1 / 2.205
        lbs_sq_ft_to_kg_sq_m = 0.0421401101
        # lbs_sq_ft_to_kg_sq_m = 1.

        wing_mass = self.declare_variable('wing_mass', shape=shape, val=16740.*lbs_to_kg)
        wing_strut_mass = self.declare_variable('wing_strut_mass', shape=shape, val=3680.*lbs_to_kg)

        total_mass = self.declare_variable('total_mass', shape=shape, val=150000.*lbs_to_kg)
        total_mass_without_wing_strut = total_mass - (wing_mass + wing_strut_mass)   
        total_mass_only_wing_strut = (wing_mass + wing_strut_mass)     
        self.register_output(name = 'mass', var = total_mass_only_wing_strut)

        dummy = self.declare_variable('dummy_variable', shape=shape, val=0.)

        # Ixx = 412459.845 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_xx_a = self.declare_variable('moment_of_inertia_xx_a', shape=shape, val=1447357.906*lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xx_without_wing = moment_of_inertia_xx_a + dummy 
        self.register_output(name = 'Ixx', var = moment_of_inertia_xx_without_wing)
        
        # Iyy = 13379933.707 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_yy_a = self.declare_variable('moment_of_inertia_yy', shape=shape, val=23602.589* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yy_without_wing = moment_of_inertia_yy_a + dummy
        self.register_output(name = 'Iyy', var = moment_of_inertia_yy_without_wing)

        moment_of_inertia_zz_a = self.declare_variable('moment_of_inertia_zz', shape=shape, val=1466652.226 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_zz_without_wing = moment_of_inertia_zz_a + dummy
        self.register_output(name = 'Izz', var = moment_of_inertia_zz_without_wing)

        moment_of_inertia_xy_a = self.declare_variable('moment_of_inertia_xy', shape=shape, val=-37.361 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xy_without_wing = moment_of_inertia_xy_a + dummy
        self.register_output(name = 'Ixy', var = moment_of_inertia_xy_without_wing)

        moment_of_inertia_yz_a = self.declare_variable('moment_of_inertia_yz', shape=shape, val=153.697* lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yz_without_wing = moment_of_inertia_yz_a + dummy
        self.register_output(name = 'Iyz', var = moment_of_inertia_yz_without_wing)

        moment_of_inertia_xz_a = self.declare_variable('moment_of_inertia_xz', shape=shape, val=-1902.288 * lbs_sq_ft_to_kg_sq_m)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xz_without_wing = moment_of_inertia_xz_a + dummy
        self.register_output(name = 'Ixz', var = moment_of_inertia_xz_without_wing)

        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(moment_of_inertia_xx_without_wing, (1, 1)) 
        inertia_tensor[1, 1] = csdl.reshape(moment_of_inertia_yy_without_wing, (1, 1)) 
        inertia_tensor[2, 2] = csdl.reshape(moment_of_inertia_zz_without_wing, (1, 1)) 
        inertia_tensor[0, 2] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[2, 0] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[1, 2] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[2, 1] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[0, 1] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        inertia_tensor[1, 0] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        
        cg_x_a = self.declare_variable('center_of_gravity_x', shape=shape, val=58.595*ft2m)
        # cg_x = 57.505*ft2m
        cg_x = cg_x_a + dummy
        # cg_y = 0.
        cg_y_a = self.declare_variable('center_of_gravity_y', shape=shape, val=0.*ft2m)
        cg_y = cg_y_a + dummy
        # cg_z = 0.253*ft2m
        cg_z_a = self.declare_variable('center_of_gravity_z', shape=shape, val=5.715*ft2m)
        cg_z = cg_z_a + dummy

        self.register_output(
            name='cgx', 
            var=cg_x)
        self.register_output(
            name='cgy',  
            var=cg_y)
        self.register_output(
            name='cgz', 
            var=cg_z)
        
        cg_vector = self.create_output('cg_vector', shape=(3, ), val=0)
        cg_vector[0] = cg_x
        cg_vector[1] = cg_y
        cg_vector[2] = cg_z
        
# # Module imports
# import numpy as np
# import caddee.api as cd
# import m3l
# from python_csdl_backend import Simulator
# # from modopt.scipy_library import SLSQP
# # from modopt.snopt_library import SNOPT
# # from modopt.csdl_library import CSDLProblem
# from lsdo_rotor import BEMParameters, evaluate_multiple_BEM_models, BEM
# # PittPeters, PittPetersParameters, evaluate_multiple_pitt_peters_models
# from VAST import FluidProblem, VASTFluidSover, VASTNodalForces
# # from lsdo_acoustics import Acoustics, evaluate_multiple_acoustic_models
# # from lsdo_motor import evaluate_multiple_motor_sizing_models, evaluate_multiple_motor_analysis_models, MotorAnalysis, MotorSizing
# from aframe import BeamMassModel, EBBeam, EBBeamForces
# from caddee.utils.helper_functions.geometry_helpers import  make_vlm_camber_mesh
# import time 
# import pickle


# from examples.advanced_examples.ex_tbw_geometry_setup_trial_1 import (wing_meshes, htail_meshes,  
#                                                                 left_strut_meshes, right_strut_meshes, 
#                                                                 left_jury_meshes, right_jury_meshes, 
#                                                                 vtail_meshes, fuesleage_mesh, 
#                                                                 S_ref, h_tail_area, v_tail_area, jury_area, strut_area,
#                                                                 fuselage_length, drag_comp_list, wing_AR, system_model, 
#                                                                 wing_box_beam_mesh, left_strut_box_beam_mesh, right_strut_box_beam_mesh,
#                                                                 left_jury_box_beam_mesh, right_jury_box_beam_mesh, num_wing_beam,
#                                                                 num_strut_beam, num_jury_beam)

# t2 = time.time()
# t3 = time.time()
# caddee = cd.CADDEE()

# if __name__ == '__main__':
#     # tbw_mass_model = TBWMassPropertiesMinusWingCSDL()
#     # import python_csdl_backend
#     # sim = python_csdl_backend.Simulator(tbw_mass_model)
#     # sim.run()
#     # print('inertia_tensor: ', sim['inertia_tensor'])
#     # print('mass:', sim['mass'])
#     # print('cg_vector: ', sim['cg_vector'])

#     # from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l
#     tbw_wt = TBWMassProperties_m3l(
#     name='TBW_Mass_Properties',
#     exclude_wing = True,
#     full_wing = False,
#     )
#     tbw_mass_properties= tbw_wt.evaluate()
#     system_model.register_output(tbw_mass_properties)

#     caddee_csdl_model = system_model.assemble_csdl()
#     sim = Simulator(caddee_csdl_model,analytics=True)
#     sim.run()
#     cd.print_caddee_outputs(system_model, sim, compact_print=True)