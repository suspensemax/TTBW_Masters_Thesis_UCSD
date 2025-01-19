
# Module imports
import numpy as np
import m3l
from python_csdl_backend import Simulator
# from modopt.scipy_library import SLSQP
# from modopt.external_libraries.snopt import snoptc
from modopt import SNOPT
# from modopt.snopt_library import SNOPT
from modopt import CSDLProblem
# from modopt.csdl_library import CSDLProblem
# from lsdo_rotor import BEMParameters, evaluate_multiple_BEM_models, BEM
# PittPeters, PittPetersParameters, evaluate_multiple_pitt_peters_models
from VAST import FluidProblem, VASTFluidSover, VASTNodalForces
# from lsdo_acoustics import Acoustics, evaluate_multiple_acoustic_models
# from lsdo_motor import evaluate_multiple_motor_sizing_models, evaluate_multiple_motor_analysis_models, MotorAnalysis, MotorSizing
from aframe import BeamMassModel, EBBeam, EBBeamForces
# from caddee.utils.helper_functions.geometry_helpers import  make_vlm_camber_mesh
import time 
import pickle
import pandas as pd
import caddee.api as cd

from examples.advanced_examples.Mission_Analysis.Climb.tbw_geometry_trial_1_climb import (wing_meshes, htail_meshes, wing_span_dv, wing_root_chord_dv, 
                                                                        wing_mid_chord_left_dv, wing_tip_chord_left_dv,  
                                                                left_strut_meshes, right_strut_meshes, 
                                                                left_jury_meshes, right_jury_meshes, 
                                                                vtail_meshes, S_ref_area, system_model, wing_AR, 
                                                                h_tail_area, jury_area, strut_area, angle_beta,
                                                                wing_box_beam_mesh, left_strut_box_beam_mesh, right_strut_box_beam_mesh,
                                                                left_jury_box_beam_mesh, right_jury_box_beam_mesh, num_wing_beam,
                                                                num_strut_beam, num_jury_beam)


t2 = time.time()
t3 = time.time()
caddee = cd.CADDEE()

# region tbw area
from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea
tbw_area_model_plus_1_point_0g = tbwArea(
    name='TBW_area_plus_1_point_0g',
    counter = 'plus_1_point_0_g',
)
tbw_area_outputs_plus_1_point_0g = tbw_area_model_plus_1_point_0g.evaluate(wing_span_dv = wing_span_dv, 
                                                                           wing_root_chord_dv = wing_root_chord_dv, 
                                                                           wing_mid_chord_left_dv = wing_mid_chord_left_dv,
                                                                           wing_tip_chord_left_dv = wing_tip_chord_left_dv,
                                                                           area_wing = S_ref_area,
                                                                           strut_area = strut_area,
                                                                           AR_wing= wing_AR, 
                                                                           )

system_model.register_output(tbw_area_outputs_plus_1_point_0g)
# endregion

# region tbw weights
from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

tbw_wt = TBWMassProperties_m3l(
name='TBW_Mass_Properties',
exclude_wing = True,
full_wing = False,
geometry_units='ft',
)
tbw_mass_properties= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
system_model.register_output(tbw_mass_properties)
# endregion

# region Beam sizing 
# create the aframe dictionaries:
joints, bounds, beams = {}, {}, {}

youngs_modulus = 73.1E9 #73.1Gpa - Aluminum 2024-T4
poissons_ratio = 0.33
# density = 6200.  # SI - only wing mass - 5820
# density = 3450.  # SI - only wing mass - 5820
# density = 2780.  # SI - only wing mass - 5820
density = 4150.  # SI - only wing mass - 5820

# density_wing = 5800.
# density_strut_jury = 3000.

density_wing = density
density_strut_jury = density

beams['wing_beam'] = {'E': youngs_modulus,
                        'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                        'rho': density_wing, 'cs': 'box',
                        'nodes': list(range(num_wing_beam))}
beams['right_strut_beam'] = {'E': youngs_modulus,
                        'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                        'rho': density_strut_jury, 'cs': 'box',
                        'nodes': list(range(num_strut_beam))}
beams['left_strut_beam'] = {'E': youngs_modulus,
                                'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                                'rho': density_strut_jury, 'cs': 'box',
                                'nodes': list(range(num_strut_beam))}
beams['right_jury_beam'] = {'E': youngs_modulus,
                                'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                                'rho': density_strut_jury, 'cs': 'box',
                                'nodes': list(range(num_jury_beam))}
beams['left_jury_beam'] = {'E': youngs_modulus,
                            'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                            'rho': density_strut_jury, 'cs': 'box',  
                            'nodes': list(range(num_jury_beam))}


bounds['wing_root'] = {'beam': 'wing_beam', 'node': 10, 'fdim': [1, 1, 1, 1, 1, 1]}
bounds['left_strut_root'] = {'beam': 'left_strut_beam', 'node': 0, 'fdim': [1, 1, 1, 1, 1, 1]}
bounds['right_strut_root'] = {'beam': 'right_strut_beam', 'node': 0, 'fdim': [1, 1, 1, 1, 1, 1]}

joints['wing_leftstrut'] = {'beams': ['wing_beam','left_strut_beam'],'nodes': [16, 8]}
joints['wing_rightstrut'] = {'beams': ['wing_beam', 'right_strut_beam'], 'nodes': [4, 8]}
joints['wing_leftjury'] = {'beams': ['wing_beam', 'left_jury_beam'], 'nodes': [14, 0]}
joints['wing_rightjury'] = {'beams': ['wing_beam', 'right_jury_beam'], 'nodes': [6, 0]}
joints['leftstrut_leftjury'] = {'beams': ['left_strut_beam', 'left_jury_beam'], 'nodes': [4, 2]}
joints['rightstrut_rightjury'] = {'beams': ['right_strut_beam', 'right_jury_beam'], 'nodes': [4, 2]}

inches_to_units = 0.0254
# inches_to_units = 1/12
ft_to_m = 0.3048

sizing = False

wing_beam_ttop_inches_skin = np.array([0.15, 0.25, 0.50, 0.625, 0.6875, 0.480, 0.230, 0.238, 0.240, 0.245, 
                                    0.40, 0.245, 0.240, 0.238, 0.230, 0.480, 0.6875, 0.625, 0.5, 0.25, 0.15]) # incorrect??
wing_beam_ttop_metres_skin = wing_beam_ttop_inches_skin * inches_to_units * 0.93

wing_beam_tbot_inches_skin = np.array([0.15, 0.25, 0.50, 0.625, 0.6875, 0.480, 0.230, 0.238, 0.240, 0.245, 
                                    0.40, 0.245, 0.240, 0.238, 0.230, 0.480, 0.6875, 0.625, 0.5, 0.25, 0.15]) # incorrect??
wing_beam_tbot_metres_skin = wing_beam_tbot_inches_skin * inches_to_units * 0.8
# wing_beam_tbot_metres_skin = wing_beam_tbot_inches_skin * inches_to_units

# wing_beam_tweb_metres = 0.4 * np.ones(num_wing_beam) * inches_to_units 
wing_beam_tweb_metres = 0.3 * np.ones(num_wing_beam) * inches_to_units 
# wing_beam_tweb_metres = 0.40 * np.ones(num_wing_beam) * inches_to_units 
# wing_beam_tweb_metres = wing_beam_ttop_metres_skin * 0.75
# wing_beam_tweb_metres = wing_beam_ttop_metres_skin
# wing_beam_ttop_metres_skin = wing_beam_tweb_metres
# wing_beam_tbot_metres_skin = wing_beam_tweb_metres

# wing_beam_t_top = system_model.create_input(name='wing_beam_ttop' ,val=wing_beam_ttop_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127 # skin thickness
wing_beam_t_top = system_model.create_input(name='wing_beam_ttop' ,val=wing_beam_ttop_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127 # skin thickness
wing_beam_t_bot = system_model.create_input(name='wing_beam_tbot' ,val=wing_beam_tbot_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
wing_beam_tweb = system_model.create_input(name='wing_beam_tweb' ,val=wing_beam_tweb_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

# strut_beam_inches_skin = np.array([0.490, 0.425, 0.365, 0.330, 0.290, 0.3125, 0.345, 0.390, 0.425]) # w/o constraint
strut_beam_inches_skin = np.array([0.490, 0.425, 0.365, 0.330, 0.3125, 0.290, 0.270, 0.240, 0.220]) #with constraint

# n_values = num_strut_beam
# expanded_strut_beam_inches_skin = np.interp(
#     np.linspace(0, len(strut_beam_inches_skin) - 1, n_values),
#     np.arange(len(strut_beam_inches_skin)),
#     strut_beam_inches_skin
# )

strut_beam_metres_skin = strut_beam_inches_skin * inches_to_units
# strut_beam_metres_skin = expanded_strut_beam_inches_skin * inches_to_units
# strut_beam_metres = 0.45 * np.ones(num_strut_beam) * inches_to_units 
strut_beam_metres = 0.30 * np.ones(num_strut_beam) * inches_to_units 
# strut_beam_metres = strut_beam_metres_skin * 0.75

left_strut_beam_t_top = system_model.create_input(name='left_strut_beam_ttop' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127
left_strut_beam_t_bot = system_model.create_input(name='left_strut_beam_tbot' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
left_strut_beam_tweb = system_model.create_input(name='left_strut_beam_tweb' ,val=strut_beam_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

right_strut_beam_t_top = system_model.create_input(name='right_strut_beam_ttop' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127
right_strut_beam_t_bot = system_model.create_input(name='right_strut_beam_tbot' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
right_strut_beam_tweb = system_model.create_input(name='right_strut_beam_tweb' ,val=strut_beam_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

jury_beam_metres = 0.15 * inches_to_units * np.ones((num_jury_beam, ))

left_jury_beam_t_top = system_model.create_input(name='left_jury_beam_ttop' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3) # lower=0.00127
left_jury_beam_t_bot = system_model.create_input(name='left_jury_beam_tbot' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
left_jury_beam_tweb = system_model.create_input(name='left_jury_beam_tweb' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
right_jury_beam_t_top = system_model.create_input(name='right_jury_beam_ttop' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3) # lower=0.00127
right_jury_beam_t_bot = system_model.create_input(name='right_jury_beam_tbot' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
right_jury_beam_tweb = system_model.create_input(name='right_jury_beam_tweb' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)


beam_mass_model = BeamMassModel(
beams=beams,
name='beam_mass_model',
mesh_units = 'ft',
)

# wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes],
#                                     width=[wing_box_beam_mesh.width], height=[wing_box_beam_mesh.height], 
#                                     t_top=[wing_beam_t_top], t_bot=[wing_beam_t_bot] ,t_web=[wing_beam_tweb])

# wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes * ft_to_m, left_strut_box_beam_mesh.beam_nodes * ft_to_m, right_strut_box_beam_mesh.beam_nodes * ft_to_m, left_jury_box_beam_mesh.beam_nodes * ft_to_m, right_jury_box_beam_mesh.beam_nodes * ft_to_m],
#                                     width=[wing_box_beam_mesh.width * ft_to_m * ft_to_m, left_strut_box_beam_mesh.width * ft_to_m, right_strut_box_beam_mesh.width * ft_to_m, left_jury_box_beam_mesh.width * ft_to_m, right_jury_box_beam_mesh.width * ft_to_m], 
#                                     height=[wing_box_beam_mesh.height * ft_to_m, left_strut_box_beam_mesh.height * ft_to_m, right_strut_box_beam_mesh.height * ft_to_m, left_jury_box_beam_mesh.height * ft_to_m, right_jury_box_beam_mesh.height * ft_to_m], 
#                                     t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
#                                     t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
#                                     t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb])

wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes, left_strut_box_beam_mesh.beam_nodes, right_strut_box_beam_mesh.beam_nodes, left_jury_box_beam_mesh.beam_nodes, right_jury_box_beam_mesh.beam_nodes],
                                    width=[wing_box_beam_mesh.width, left_strut_box_beam_mesh.width, right_strut_box_beam_mesh.width, left_jury_box_beam_mesh.width, right_jury_box_beam_mesh.width], 
                                    height=[wing_box_beam_mesh.height, left_strut_box_beam_mesh.height, right_strut_box_beam_mesh.height, left_jury_box_beam_mesh.height, right_jury_box_beam_mesh.height], 
                                    t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
                                    t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
                                    t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb])

system_model.register_output(wing_beam_mass_props)

total_mass_props_model = cd.TotalMassPropertiesM3L(
name=f"total_mass_properties_model_tbw"
)

total_mass_props = total_mass_props_model.evaluate(component_mass_properties=[wing_beam_mass_props, tbw_mass_properties])
system_model.register_output(total_mass_props)
# system_model.add_objective(total_mass_props.mass, scaler=1e-5)
# system_model.add_constraint(total_mass_props.mass, scaler=1e-5)
# endregion

climb = False

if climb:
    climb_condition = cd.ClimbCondition(
        name='steady_climb',
        num_nodes=1,
    )

    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    climb_hi = system_model.create_input('climb_initial_altitude', val=450)
    climb_hf = system_model.create_input('climb_final_altitude', val=13106)
    climb_M = system_model.create_input('climb_M', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    climb_pitch = system_model.create_input('climb_pitch', val=np.deg2rad(-0.95), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=10)
    climb_flight_path_angle = system_model.create_input(name='climb_flight_path_angle', val=np.deg2rad(3.8))

    # ac sates + atmos
    climb_ac_states, climb_atmos = climb_condition.evaluate(mach_number=climb_M, pitch_angle=climb_pitch, flight_path_angle=climb_flight_path_angle,
                                                            initial_altitude=climb_hi, final_altitude=climb_hf)

    system_model.register_output(climb_ac_states)
    system_model.register_output(climb_atmos)


    # region VAST solver
    vlm_model = VASTFluidSover(
    name='plus_1g_vlm_model',
    surface_names=[
        'wing_mesh_plus_1_point_0g',
        'htail_mesh_plus_1_point_0g',
        'right_strut_mesh_plus_1_point_0g',
        'left_strut_mesh_plus_1_point_0g',
        # 'right_jury_mesh_plus_1_point_0g',
        # 'left_jury_mesh_plus_1_point_0g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],s
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    cl0=[0.7, 0., 0., 0.]
    # cl0=[1.15, 0., 0., 0.]
    # cl0=[0.8, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0., 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    plus_1_point_0_g_elevator = system_model.create_input('plus_1_point_0_g_elevator', val=np.deg2rad(-0.75), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    vlm_outputs_plus_1_point_0g = vlm_model.evaluate(
    atmosphere=climb_atmos,
    ac_states=climb_ac_states,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, plus_1_point_0_g_elevator, None, None],
    # deflections=[None, plus_1_point_0_g_elevator],
    # deflections=[None],
    # wing_AR=wing_AR,
    wing_AR=tbw_area_outputs_plus_1_point_0g.wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(vlm_outputs_plus_1_point_0g)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    throttle_plus_1_point_0g = system_model.create_input('throttle_plus_1_point_0g', val = 1.5, shape=(1, ), dv_flag=True, lower=0., upper=2.5)
    tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_1_point_0g',
        counter = 'plus_1_point_0g',
    )
    tbw_left_propulsion_outputs_plus_1_point_0g = tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = throttle_plus_1_point_0g)
    system_model.register_output(tbw_left_propulsion_outputs_plus_1_point_0g)
    # endregion

    # region Viscous drag
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model # only cd_wave
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model # cd_wave + cd_interference
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 iport Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model
    # system_model.register_output('S_area',S_ref.value)

    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 ms
    tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_plus_1_point_0g',
            geometry_units='ft',
            counter = 'plus_1_point_0g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    wing_sweep_angle = system_model.create_input(name='wing_sweep_angle', shape=(1,), val=np.deg2rad(14.06),)
    chord_value = system_model.create_input('chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = climb_atmos, ac_states=climb_ac_states, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = chord_value, vlm_outputs = vlm_outputs_plus_1_point_0g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = wing_sweep_angle, mach_number = climb_M, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2 & tbw_viscous_drag_model_sweep_3
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_viscous_drag_outputs_plus_1_point_0g)
    # endregion

    from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

    tbw_L_over_D_plus_1_point_0g = Tbw_L_over_D_Model(
        name = 'tbw_L_over_D_final_plus_1_point_0g',
        counter = 'plus_1_point_0g', 
    )
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_plus_1_point_0g.forces)
    tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = vlm_outputs_plus_1_point_0g, tbw_viscous_drag = tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_L_over_D_outputs_plus_1_point_0g)

    plus_1_point_0g_trim_variables = climb_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g, tbw_left_propulsion_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g],
    ac_states=climb_ac_states,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(plus_1_point_0g_trim_variables)
            
    # system_model.add_constraint(plus_1_point_0g_trim_variables.accelerations, equals=0.)
    system_model.add_objective(plus_1_point_0g_trim_variables.accelerations, scaler = 1E5)
    # system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
    # system_model.add_constraint(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)

    # from caddee.utils.aircraft_models.tbw.tbw_range import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_range_2 import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_fuel_burn import Tbw_fuel_burn_Model
    from caddee.utils.aircraft_models.tbw.tbw_fuel_burn_climb import Tbw_fuel_burn_Model

    tbw_fuel_burn_plus_1_point_0g = Tbw_fuel_burn_Model(
        name = 'tbw_fuel_burn_final',
        counter = 'plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g', 
    )
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props.mass, thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props, thrust = tbw_left_propulsion_outputs_plus_1_point_0g)
    tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, 
                                                                                   wing_beam_mass_props = wing_beam_mass_props.mass, 
                                                                           tbw_mass_properties = tbw_mass_properties.mass, 
                                                                           thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust, 
                                                                           ac_states=climb_ac_states,)
    
    system_model.register_output(tbw_fuel_burn_outputs_plus_1_point_0g)

    # system_model.add_objective(tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1e-3)
    counter = 3
    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = f'Forward_tbw_trim_{counter}_climb_baseline_1g', analytics=True)
    sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)


    print('Total Forces: ', sim['steady_climb_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_climb_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_climb_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_climb_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_climb_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_climb_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_climb_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['climb_pitch'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_climb_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_climb_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_climb_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_climb_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('fuel_burned', sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('final_weight', sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.final_weight'])
    # exit()
    prob = CSDLProblem(problem_name=f'mission_segment_climb_{counter}', simulator=sim)
    snopt_options = {
        'Major iterations': 100, 
        # 'Major iterations': 400, 
        'Major optimality': 1e-4, 
        'Major feasibility': 1e-4, 
        'Iteration limit' :100000,
        }
    optimizer = SNOPT(prob, solver_options=snopt_options)
    # optimizer = SLSQP(prob, maxiter=400, ftol=1E-5)
    optimizer.solve()
    optimizer.print_results()
    print('Total Forces: ', sim['steady_climb_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_climb_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_climb_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_climb_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_climb_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_climb_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_climb_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['climb_pitch'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_climb_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_climb_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_climb_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_climb_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('fuel_burned', sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('final_weight', sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.final_weight'])

descent = False

if descent:
    descent_condition = cd.ClimbCondition(
        name='steady_descent',
        num_nodes=1,
    )

    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    descent_hi = system_model.create_input('descent_initial_altitude', val=13106)
    descent_hf = system_model.create_input('descent_final_altitude', val=450)
    descent_M = system_model.create_input('descent_M', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    descent_pitch = system_model.create_input('descent_pitch', val=np.deg2rad(-10), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=10)
    descent_flight_path_angle = system_model.create_input(name='descent_flight_path_angle', val=np.deg2rad(-3))

    # ac sates + atmos
    descent_ac_states, descent_atmos = descent_condition.evaluate(mach_number=descent_M, pitch_angle=descent_pitch, flight_path_angle=descent_flight_path_angle,
                                                            initial_altitude=descent_hi, final_altitude=descent_hf)

    system_model.register_output(descent_ac_states)
    system_model.register_output(descent_atmos)


    # region VAST solver
    descent_vlm_model = VASTFluidSover(
    name='descent_plus_1g_vlm_model',
    surface_names=[
        'descent_wing_mesh_plus_1_point_0g',
        'descent_htail_mesh_plus_1_point_0g',
        'descent_right_strut_mesh_plus_1_point_0g',
        'descent_left_strut_mesh_plus_1_point_0g',
        # 'right_jury_mesh_plus_1_point_0g',
        # 'left_jury_mesh_plus_1_point_0g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],s
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    # cl0=[0.7, 0., 0., 0.]
    # cl0=[1.15, 0., 0., 0.]
    cl0=[0.89, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0., 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    descent_plus_1_point_0_g_elevator = system_model.create_input('descent_plus_1_point_0_g_elevator', val=np.deg2rad(-7), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    descent_vlm_outputs_plus_1_point_0g = descent_vlm_model.evaluate(
    atmosphere=descent_atmos,
    ac_states=descent_ac_states,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, descent_plus_1_point_0_g_elevator, None, None],
    # deflections=[None, plus_1_point_0_g_elevator],
    # deflections=[None],
    # wing_AR=wing_AR,
    wing_AR=tbw_area_outputs_plus_1_point_0g.wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(descent_vlm_outputs_plus_1_point_0g)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    descent_throttle_plus_1_point_0g = system_model.create_input('descent_throttle_plus_1_point_0g', val = 0.2, shape=(1, ), dv_flag=True, lower=-1., upper=2.5)
    descent_tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_1_point_0g',
        counter = 'descent_plus_1_point_0g',
    )
    descent_tbw_left_propulsion_outputs_plus_1_point_0g = descent_tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = descent_throttle_plus_1_point_0g)
    system_model.register_output(descent_tbw_left_propulsion_outputs_plus_1_point_0g)
    # endregion

    # region Viscous drag
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model # only cd_wave
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model # cd_wave + cd_interference
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 iport Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model
    # system_model.register_output('S_area',S_ref.value)

    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 ms
    descent_tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_plus_1_point_0g',
            geometry_units='ft',
            counter = 'descent_plus_1_point_0g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    descent_wing_sweep_angle = system_model.create_input(name='descent_wing_sweep_angle', shape=(1,), val=np.deg2rad(14.06),)
    descent_chord_value = system_model.create_input('descent_chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    descent_tbw_viscous_drag_outputs_plus_1_point_0g = descent_tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = descent_atmos, ac_states=descent_ac_states, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = descent_chord_value, vlm_outputs = descent_vlm_outputs_plus_1_point_0g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = descent_wing_sweep_angle, mach_number = descent_M, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2 & tbw_viscous_drag_model_sweep_3
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(descent_tbw_viscous_drag_outputs_plus_1_point_0g)
    # endregion

    from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

    descent_tbw_L_over_D_plus_1_point_0g = Tbw_L_over_D_Model(
        name = 'tbw_L_over_D_final_plus_1_point_0g',
        counter = 'descent_plus_1_point_0g', 
    )
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_plus_1_point_0g.forces)
    descent_tbw_L_over_D_outputs_plus_1_point_0g = descent_tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = descent_vlm_outputs_plus_1_point_0g, tbw_viscous_drag = descent_tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(descent_tbw_L_over_D_outputs_plus_1_point_0g)

    descent_plus_1_point_0g_trim_variables = descent_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[descent_vlm_outputs_plus_1_point_0g, descent_tbw_left_propulsion_outputs_plus_1_point_0g, descent_tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g],
    ac_states=descent_ac_states,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(descent_plus_1_point_0g_trim_variables)
            
    # system_model.add_constraint(plus_1_point_0g_trim_variables.accelerations, equals=0.)
    system_model.add_objective(descent_plus_1_point_0g_trim_variables.accelerations, scaler = 5)
    # system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
    # system_model.add_constraint(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)

    # from caddee.utils.aircraft_models.tbw.tbw_range import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_range_2 import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_fuel_burn import Tbw_fuel_burn_Model
    from caddee.utils.aircraft_models.tbw.tbw_fuel_burn_climb import Tbw_fuel_burn_Model

    descent_tbw_fuel_burn_plus_1_point_0g = Tbw_fuel_burn_Model(
        name = 'tbw_fuel_burn_final',
        counter = 'descent_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g', 
    )
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props.mass, thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props, thrust = tbw_left_propulsion_outputs_plus_1_point_0g)
    descent_tbw_fuel_burn_outputs_plus_1_point_0g = descent_tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = descent_tbw_L_over_D_outputs_plus_1_point_0g, 
                                                                                   wing_beam_mass_props = wing_beam_mass_props.mass, 
                                                                           tbw_mass_properties = tbw_mass_properties.mass, 
                                                                           thrust = descent_tbw_left_propulsion_outputs_plus_1_point_0g.thrust, 
                                                                           ac_states=descent_ac_states,)
    
    system_model.register_output(descent_tbw_fuel_burn_outputs_plus_1_point_0g)

    # system_model.add_objective(tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1e-3)
    counter = 4
    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = f'Forward_tbw_trim_{counter}_descent_baseline_1g', analytics=True)
    sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)


    print('Total Forces: ', sim['steady_descent_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_descent_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_descent_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_descent_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_descent_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_descent_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_descent_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['descent_pitch'])
    print('tbw elevator deflection', sim['descent_plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_descent_total_forces_moments_model.descent_plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_descent_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_descent_total_forces_moments_model.descent_plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_descent_total_forces_moments_model.descent_plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['descent_plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['descent_plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('fuel_burned', sim['descent_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('final_weight', sim['descent_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.final_weight'])
    exit()
    prob = CSDLProblem(problem_name=f'mission_segment_descent_{counter}', simulator=sim)
    snopt_options = {
        'Major iterations': 100, 
        # 'Major iterations': 400, 
        'Major optimality': 1e-4, 
        'Major feasibility': 1e-4, 
        'Iteration limit' :100000,
        }
    optimizer = SNOPT(prob, solver_options=snopt_options)
    # optimizer = SLSQP(prob, maxiter=400, ftol=1E-5)
    optimizer.solve()
    optimizer.print_results()
    print('Total Forces: ', sim['steady_descent_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_descent_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_descent_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_descent_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_descent_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_descent_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_descent_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['descent_pitch'])
    print('tbw elevator deflection', sim['descent_plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_descent_total_forces_moments_model.descent_plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_descent_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_descent_total_forces_moments_model.descent_plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_descent_total_forces_moments_model.descent_plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['descent_plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['descent_plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['descent_plus_1g_vlm_model.descent_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['descent_plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('fuel_burned', sim['descent_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('final_weight', sim['descent_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.final_weight'])

climb = False

if climb:
    climb_condition = cd.ClimbCondition(
        name='steady_climb',
        num_nodes=1,
    )

    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    climb_hi = system_model.create_input('climb_initial_altitude', val=450)
    climb_hf = system_model.create_input('climb_final_altitude', val=13106)
    climb_M = system_model.create_input('climb_M', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    # climb_pitch = system_model.create_input('climb_pitch', val=np.deg2rad(-0.95), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=10)
    climb_pitch = system_model.create_input('climb_pitch', val=np.deg2rad(-0.95), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))
    climb_flight_path_angle = system_model.create_input(name='climb_flight_path_angle', val=np.deg2rad(3.8)) #3.8

    # ac sates + atmos
    climb_ac_states, climb_atmos = climb_condition.evaluate(mach_number=climb_M, pitch_angle=climb_pitch, flight_path_angle=climb_flight_path_angle,
                                                            initial_altitude=climb_hi, final_altitude=climb_hf)

    system_model.register_output(climb_ac_states)
    system_model.register_output(climb_atmos)


    # region VAST solver
    climb_vlm_model = VASTFluidSover(
    name='climb_plus_1g_vlm_model',
    surface_names=[
        'climb_wing_mesh_plus_1_point_0g',
        'climb_htail_mesh_plus_1_point_0g',
        'climb_right_strut_mesh_plus_1_point_0g',
        'climb_left_strut_mesh_plus_1_point_0g',
        # 'right_jury_mesh_plus_1_point_0g',
        # 'left_jury_mesh_plus_1_point_0g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],s
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    # cl0=[0.7, 0., 0., 0.]
    # cl0=[1.15, 0., 0., 0.]
    cl0=[0.7, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0., 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    # climb_plus_1_point_0_g_elevator = system_model.create_input('climb_plus_1_point_0_g_elevator', val=np.deg2rad(1.5), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))
    climb_plus_1_point_0_g_elevator = system_model.create_input('climb_plus_1_point_0_g_elevator', val=np.deg2rad(-0.5), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    climb_vlm_outputs_plus_1_point_0g = climb_vlm_model.evaluate(
    atmosphere=climb_atmos,
    ac_states=climb_ac_states,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, climb_plus_1_point_0_g_elevator, None, None],
    # deflections=[None, plus_1_point_0_g_elevator],
    # deflections=[None],
    # wing_AR=wing_AR,
    wing_AR=tbw_area_outputs_plus_1_point_0g.wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(climb_vlm_outputs_plus_1_point_0g)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    climb_throttle_plus_1_point_0g = system_model.create_input('climb_throttle_plus_1_point_0g', val = 0.7, shape=(1, ), dv_flag=True, lower=-1., upper=2.5)
    climb_tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_1_point_0g',
        counter = 'climb_plus_1_point_0g',
    )
    climb_tbw_left_propulsion_outputs_plus_1_point_0g = climb_tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = climb_throttle_plus_1_point_0g)
    system_model.register_output(climb_tbw_left_propulsion_outputs_plus_1_point_0g)
    # endregion

    # region Viscous drag
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model # only cd_wave
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model # cd_wave + cd_interference
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 iport Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model
    # system_model.register_output('S_area',S_ref.value)

    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 ms
    climb_tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_plus_1_point_0g',
            geometry_units='ft',
            counter = 'climb_plus_1_point_0g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    climb_wing_sweep_angle = system_model.create_input(name='climb_wing_sweep_angle', shape=(1,), val=np.deg2rad(14.06),)
    climb_chord_value = system_model.create_input('climb_chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    climb_tbw_viscous_drag_outputs_plus_1_point_0g = climb_tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = climb_atmos, ac_states=climb_ac_states, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = climb_chord_value, vlm_outputs = climb_vlm_outputs_plus_1_point_0g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = climb_wing_sweep_angle, mach_number = climb_M, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2 & tbw_viscous_drag_model_sweep_3
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(climb_tbw_viscous_drag_outputs_plus_1_point_0g)
    # endregion

    from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

    climb_tbw_L_over_D_plus_1_point_0g = Tbw_L_over_D_Model(
        name = 'tbw_L_over_D_final_plus_1_point_0g',
        counter = 'climb_plus_1_point_0g', 
    )
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_plus_1_point_0g.forces)
    climb_tbw_L_over_D_outputs_plus_1_point_0g = climb_tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = climb_vlm_outputs_plus_1_point_0g, tbw_viscous_drag = climb_tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(climb_tbw_L_over_D_outputs_plus_1_point_0g)

    climb_plus_1_point_0g_trim_variables = climb_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[climb_vlm_outputs_plus_1_point_0g, climb_tbw_left_propulsion_outputs_plus_1_point_0g, climb_tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g],
    ac_states=climb_ac_states,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(climb_plus_1_point_0g_trim_variables)
            
    system_model.add_constraint(climb_plus_1_point_0g_trim_variables.accelerations, equals=0., scaler = 5) #100
    # system_model.add_constraint(climb_plus_1_point_0g_trim_variables.accelerations, equals=0., scaler = 5)
    # system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
    # system_model.add_constraint(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)

    # from caddee.utils.aircraft_models.tbw.tbw_range import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_range_2 import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_fuel_burn import Tbw_fuel_burn_Model
    from caddee.utils.aircraft_models.tbw.tbw_fuel_burn_climb import Tbw_fuel_burn_Model

    climb_tbw_fuel_burn_plus_1_point_0g = Tbw_fuel_burn_Model(
        name = 'tbw_fuel_burn_final',
        counter = 'climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g', 
    )
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props.mass, thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props, thrust = tbw_left_propulsion_outputs_plus_1_point_0g)
    climb_tbw_fuel_burn_outputs_plus_1_point_0g = climb_tbw_fuel_burn_plus_1_point_0g.evaluate(total_drag = climb_tbw_L_over_D_outputs_plus_1_point_0g, 
                                                                                   wing_beam_mass_props = wing_beam_mass_props.mass, 
                                                                           tbw_mass_properties = tbw_mass_properties.mass, 
                                                                           thrust = climb_tbw_left_propulsion_outputs_plus_1_point_0g.thrust, 
                                                                           ac_states=climb_ac_states, climb_flight_path_angle = climb_flight_path_angle,
                                                                           climb_hf = climb_hf, climb_hi = climb_hi,)
    
    system_model.register_output(climb_tbw_fuel_burn_outputs_plus_1_point_0g)
    # system_model.add_constraint(climb_tbw_fuel_burn_outputs_plus_1_point_0g.time,lower = 10,upper = 30, scaler= 1.e-2)
    # system_model.add_objective(climb_tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1.e-2)
    system_model.add_objective(climb_tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1.e-3)
    counter = '4d'
    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = f'Forward_tbw_trim_{counter}_climb_baseline_1g', analytics=True)
    sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)

    prob = CSDLProblem(problem_name=f'mission_segment_climb_{counter}', simulator=sim)
    snopt_options = {
        'Major iterations': 500, 
        # 'Major iterations': 400, 
        'Major optimality': 1e-4, 
        'Major feasibility': 1e-4, 
        'Iteration limit' :100000,
        }
    optimizer = SNOPT(prob, solver_options=snopt_options)
    # optimizer = SLSQP(prob, maxiter=400, ftol=1E-5)
    optimizer.solve()
    optimizer.print_results()
    print('Total Forces: ', sim['steady_climb_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_climb_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_climb_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_climb_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_climb_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_climb_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_climb_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['climb_pitch'])
    print('tbw elevator deflection', sim['climb_plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_climb_total_forces_moments_model.climb_plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_climb_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_climb_total_forces_moments_model.climb_plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_climb_total_forces_moments_model.climb_plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['climb_plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['climb_plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('total_drag', sim['climb_plus_1_point_0g_tbw_L_over_D_model.total_drag_vlm_vsicous'])
    print('VLM induced_drag_for_each_surface', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('ROC_climb_m3l', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.ROC_climb_m3l'])
    print('fuel_burned (kg)', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('time_to_climb (hr)', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.time_to_climb_climb_m3l'])

    descent_condition = cd.ClimbCondition(
        name='steady_descent',
        num_nodes=1,
    )

    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    descent_hi = system_model.create_input('descent_initial_altitude', val=13106)
    descent_hf = system_model.create_input('descent_final_altitude', val=450)
    descent_M = system_model.create_input('descent_M', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    # descent_pitch = system_model.create_input('descent_pitch', val=np.deg2rad(-0.95), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=10)
    descent_pitch = system_model.create_input('descent_pitch', val=np.deg2rad(-0.95), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))
    descent_flight_path_angle = system_model.create_input(name='climb_flight_path_angle', val=np.deg2rad(3.8)) #3.8

    # ac sates + atmos
    climb_ac_states, climb_atmos = climb_condition.evaluate(mach_number=climb_M, pitch_angle=climb_pitch, flight_path_angle=climb_flight_path_angle,
                                                            initial_altitude=climb_hi, final_altitude=climb_hf)

    system_model.register_output(climb_ac_states)
    system_model.register_output(climb_atmos)


    # region VAST solver
    climb_vlm_model = VASTFluidSover(
    name='climb_plus_1g_vlm_model',
    surface_names=[
        'climb_wing_mesh_plus_1_point_0g',
        'climb_htail_mesh_plus_1_point_0g',
        'climb_right_strut_mesh_plus_1_point_0g',
        'climb_left_strut_mesh_plus_1_point_0g',
        # 'right_jury_mesh_plus_1_point_0g',
        # 'left_jury_mesh_plus_1_point_0g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],s
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    # cl0=[0.7, 0., 0., 0.]
    # cl0=[1.15, 0., 0., 0.]
    cl0=[0.7, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0., 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    # climb_plus_1_point_0_g_elevator = system_model.create_input('climb_plus_1_point_0_g_elevator', val=np.deg2rad(1.5), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))
    climb_plus_1_point_0_g_elevator = system_model.create_input('climb_plus_1_point_0_g_elevator', val=np.deg2rad(-0.5), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    climb_vlm_outputs_plus_1_point_0g = climb_vlm_model.evaluate(
    atmosphere=climb_atmos,
    ac_states=climb_ac_states,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, climb_plus_1_point_0_g_elevator, None, None],
    # deflections=[None, plus_1_point_0_g_elevator],
    # deflections=[None],
    # wing_AR=wing_AR,
    wing_AR=tbw_area_outputs_plus_1_point_0g.wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(climb_vlm_outputs_plus_1_point_0g)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    climb_throttle_plus_1_point_0g = system_model.create_input('climb_throttle_plus_1_point_0g', val = 0.7, shape=(1, ), dv_flag=True, lower=-1., upper=2.5)
    climb_tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_1_point_0g',
        counter = 'climb_plus_1_point_0g',
    )
    climb_tbw_left_propulsion_outputs_plus_1_point_0g = climb_tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = climb_throttle_plus_1_point_0g)
    system_model.register_output(climb_tbw_left_propulsion_outputs_plus_1_point_0g)
    # endregion

    # region Viscous drag
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model # only cd_wave
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model # cd_wave + cd_interference
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 iport Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model
    # system_model.register_output('S_area',S_ref.value)

    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 ms
    climb_tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_plus_1_point_0g',
            geometry_units='ft',
            counter = 'climb_plus_1_point_0g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    climb_wing_sweep_angle = system_model.create_input(name='climb_wing_sweep_angle', shape=(1,), val=np.deg2rad(14.06),)
    climb_chord_value = system_model.create_input('climb_chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    climb_tbw_viscous_drag_outputs_plus_1_point_0g = climb_tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = climb_atmos, ac_states=climb_ac_states, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = climb_chord_value, vlm_outputs = climb_vlm_outputs_plus_1_point_0g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = climb_wing_sweep_angle, mach_number = climb_M, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2 & tbw_viscous_drag_model_sweep_3
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(climb_tbw_viscous_drag_outputs_plus_1_point_0g)
    # endregion

    from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

    climb_tbw_L_over_D_plus_1_point_0g = Tbw_L_over_D_Model(
        name = 'tbw_L_over_D_final_plus_1_point_0g',
        counter = 'climb_plus_1_point_0g', 
    )
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_plus_1_point_0g.forces)
    climb_tbw_L_over_D_outputs_plus_1_point_0g = climb_tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = climb_vlm_outputs_plus_1_point_0g, tbw_viscous_drag = climb_tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(climb_tbw_L_over_D_outputs_plus_1_point_0g)

    climb_plus_1_point_0g_trim_variables = climb_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[climb_vlm_outputs_plus_1_point_0g, climb_tbw_left_propulsion_outputs_plus_1_point_0g, climb_tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_plus_1_point_0g],
    ac_states=climb_ac_states,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(climb_plus_1_point_0g_trim_variables)
            
    system_model.add_constraint(climb_plus_1_point_0g_trim_variables.accelerations, equals=0., scaler = 5) #100
    # system_model.add_constraint(climb_plus_1_point_0g_trim_variables.accelerations, equals=0., scaler = 5)
    # system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
    # system_model.add_constraint(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)

    # from caddee.utils.aircraft_models.tbw.tbw_range import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_range_2 import Tbw_range_Model
    # from caddee.utils.aircraft_models.tbw.tbw_fuel_burn import Tbw_fuel_burn_Model
    from caddee.utils.aircraft_models.tbw.tbw_fuel_burn_climb import Tbw_fuel_burn_Model

    climb_tbw_fuel_burn_plus_1_point_0g = Tbw_fuel_burn_Model(
        name = 'tbw_fuel_burn_final',
        counter = 'climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g', 
    )
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props.mass, thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    # tbw_fuel_burn_outputs_plus_1_point_0g = tbw_fuel_burn_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props, thrust = tbw_left_propulsion_outputs_plus_1_point_0g)
    climb_tbw_fuel_burn_outputs_plus_1_point_0g = climb_tbw_fuel_burn_plus_1_point_0g.evaluate(total_drag = climb_tbw_L_over_D_outputs_plus_1_point_0g, 
                                                                                   wing_beam_mass_props = wing_beam_mass_props.mass, 
                                                                           tbw_mass_properties = tbw_mass_properties.mass, 
                                                                           thrust = climb_tbw_left_propulsion_outputs_plus_1_point_0g.thrust, 
                                                                           ac_states=climb_ac_states, climb_flight_path_angle = climb_flight_path_angle,
                                                                           climb_hf = climb_hf, climb_hi = climb_hi,)
    
    system_model.register_output(climb_tbw_fuel_burn_outputs_plus_1_point_0g)
    # system_model.add_constraint(climb_tbw_fuel_burn_outputs_plus_1_point_0g.time,lower = 10,upper = 30, scaler= 1.e-2)
    # system_model.add_objective(climb_tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1.e-2)
    system_model.add_objective(climb_tbw_fuel_burn_outputs_plus_1_point_0g.fuel_burn, scaler = 1.e-3)
    counter = '4d'
    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = f'Forward_tbw_trim_{counter}_climb_baseline_1g', analytics=True)
    sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)

    prob = CSDLProblem(problem_name=f'mission_segment_climb_{counter}', simulator=sim)
    snopt_options = {
        'Major iterations': 500, 
        # 'Major iterations': 400, 
        'Major optimality': 1e-4, 
        'Major feasibility': 1e-4, 
        'Iteration limit' :100000,
        }
    optimizer = SNOPT(prob, solver_options=snopt_options)
    # optimizer = SLSQP(prob, maxiter=400, ftol=1E-5)
    optimizer.solve()
    optimizer.print_results()
    print('Total Forces: ', sim['steady_climb_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['steady_climb_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['steady_climb_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('beam mass', sim['beam_mass_model.mass'])
    print('total mass', sim['steady_climb_total_mass_properties_model.total_mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['steady_climb_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['steady_climb_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['steady_climb_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['climb_pitch'])
    print('tbw elevator deflection', sim['climb_plus_1_point_0_g_elevator'])
    print('VLM F', sim['steady_climb_total_forces_moments_model.climb_plus_1g_vlm_model.F'])
    print('Inertial F', sim['steady_climb_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['steady_climb_total_forces_moments_model.climb_plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['steady_climb_total_forces_moments_model.climb_plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['climb_plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['climb_plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('total_drag', sim['climb_plus_1_point_0g_tbw_L_over_D_model.total_drag_vlm_vsicous'])
    print('VLM induced_drag_for_each_surface', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('qBar_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['climb_plus_1g_vlm_model.climb_plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['climb_plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Area from my model', sim['plus_1_point_0_g_tbw_area_model.wing_area_value'])
    print('ROC_climb_m3l', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.ROC_climb_m3l'])
    print('fuel_burned (kg)', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.fuel_burn_climb_m3l'])
    print('time_to_climb (hr)', sim['climb_plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_fuel_burn_model.time_to_climb_climb_m3l'])