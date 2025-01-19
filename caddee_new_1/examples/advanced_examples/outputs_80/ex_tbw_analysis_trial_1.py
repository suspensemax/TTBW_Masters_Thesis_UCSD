# Module imports
import numpy as np
import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
# from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
from lsdo_rotor import BEMParameters, evaluate_multiple_BEM_models, BEM
# PittPeters, PittPetersParameters, evaluate_multiple_pitt_peters_models
from VAST import FluidProblem, VASTFluidSover, VASTNodalForces
# from lsdo_acoustics import Acoustics, evaluate_multiple_acoustic_models
# from lsdo_motor import evaluate_multiple_motor_sizing_models, evaluate_multiple_motor_analysis_models, MotorAnalysis, MotorSizing
from aframe import BeamMassModel, EBBeam, EBBeamForces
from caddee.utils.helper_functions.geometry_helpers import  make_vlm_camber_mesh
import time 
import pickle
import csdl

# # from examples.advanced_examples.outputs_51.ex_tbw_geometry_trial_1 import (wing_meshes, htail_meshes, wing,  
# #                                                                 left_strut_meshes, right_strut_meshes, wing_span_dv, wing_mid_chord_left_dv, 
# #                                                                 wing_tip_chord_left_dv, wing_root_chord_dv, S_ref_area, wing_AR, h_tail_area, strut_area,
# #                                                                 left_jury_meshes, right_jury_meshes, angle_beta,
# #                                                                 vtail_meshes, system_model, geometry) # span and chord working? 

# from examples.advanced_examples.outputs_51.ex_tbw_geometry_trial_1 import (wing_meshes, htail_meshes, wing,  
#                                                                 wing_twist_coefficients, wing_sweep_coefficients, 
#                                                                 left_strut_sweep_coefficients, right_strut_sweep_coefficients, 
#                                                                 left_jury_translation_x_coefficients, right_jury_translation_x_coefficients,
#                                                                 left_strut_meshes, right_strut_meshes, wing_span_dv, wing_mid_chord_left_dv, 
#                                                                 wing_tip_chord_left_dv, wing_root_chord_dv, S_ref_area, wing_AR, h_tail_area, strut_area,
#                                                                 left_jury_meshes, right_jury_meshes, wing_sweep_angle,angle_beta,
#                                                                 vtail_meshes, system_model, geometry) # span and chord and twist and sweep working? 

# t2 = time.time()
# t3 = time.time()
# caddee = cd.CADDEE()

# from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea
# tbw_area_model_plus_1_point_0g = tbwArea(
#     name='TBW_area_plus_1_point_0g',
#     counter = 'plus_1_point_0_g',
# )
# tbw_area_outputs_plus_1_point_0g = tbw_area_model_plus_1_point_0g.evaluate(wing_span_dv = wing_span_dv, 
#                                                                            wing_root_chord_dv = wing_root_chord_dv, 
#                                                                            wing_mid_chord_left_dv = wing_mid_chord_left_dv,
#                                                                            wing_tip_chord_left_dv = wing_tip_chord_left_dv,
#                                                                            area_wing = S_ref_area,
#                                                                            strut_area = strut_area,
#                                                                            AR_wing= wing_AR, 
#                                                                            )

# system_model.register_output(tbw_area_outputs_plus_1_point_0g)
# # endregion

# # region tbw weights
# # from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l
# # wing_ref_area = 1478 # 1477.109999845 ft^2 = 137.2280094 m^2
# # # wing_ref_area = S_ref.value  
# # area_value = system_model.create_input('area_plus_1_point_0g', val=wing_ref_area)

# # tbw_wt = TBWMassProperties_m3l(
# # name='TBW_Mass_Properties',
# # exclude_wing = False,
# # full_wing = True,
# # geometry_units='ft',
# # )
# # tbw_mass_properties= tbw_wt.evaluate(area = area_value)
# # system_model.register_output(tbw_mass_properties)

# from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

# tbw_wt = TBWMassProperties_m3l(
# name='TBW_Mass_Properties',
# exclude_wing = False,
# full_wing = True,
# geometry_units='ft',
# )
# tbw_mass_properties= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
# system_model.register_output(tbw_mass_properties)
# # endregion


full_geo_dv = False # working Span and chord 

if full_geo_dv:

    from examples.advanced_examples.outputs_76.ex_tbw_geometry_trial_1 import (wing_meshes, htail_meshes, wing,  
                                                                    left_strut_meshes, right_strut_meshes, wing_span_dv, wing_mid_chord_left_dv, 
                                                                    wing_tip_chord_left_dv, wing_root_chord_dv, S_ref_area, wing_AR, h_tail_area, strut_area,
                                                                    left_jury_meshes, right_jury_meshes, angle_beta,
                                                                    vtail_meshes, system_model, geometry) # span and chord working? 


    t2 = time.time()
    t3 = time.time()
    caddee = cd.CADDEE()

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
    # from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l
    # wing_ref_area = 1478 # 1477.109999845 ft^2 = 137.2280094 m^2
    # # wing_ref_area = S_ref.value  
    # area_value = system_model.create_input('area_plus_1_point_0g', val=wing_ref_area)

    # tbw_wt = TBWMassProperties_m3l(
    # name='TBW_Mass_Properties',
    # exclude_wing = False,
    # full_wing = True,
    # geometry_units='ft',
    # )
    # tbw_mass_properties= tbw_wt.evaluate(area = area_value)
    # system_model.register_output(tbw_mass_properties)

    from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

    tbw_wt = TBWMassProperties_m3l(
    name='TBW_Mass_Properties',
    exclude_wing = False,
    full_wing = True,
    geometry_units='ft',
    )
    tbw_mass_properties= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
    system_model.register_output(tbw_mass_properties)
    # endregion

    sizing_1_point_0g_condition = cd.CruiseCondition(
    name='plus_1_point_0g_sizing',
    stability_flag=False,
    )

    h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=43000) # in 'ft'
    M_1_point_0g = system_model.create_input('mach_1_point_0g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=23380342.065) # in 'ft'
    r_1_point_0g = system_model.create_input('range_1_point_0g', val=12006064000) # in 'm'
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    theta_1_point_0g = system_model.create_input('pitch_angle_1_point_0g', val=np.deg2rad(0.), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # ac sates + atmos
    ac_states_1_point_0g, atmos_1_point_0g = sizing_1_point_0g_condition.evaluate(mach_number=M_1_point_0g, pitch_angle=theta_1_point_0g, cruise_range=r_1_point_0g, altitude=h_1_point_0g)
    system_model.register_output(ac_states_1_point_0g)
    system_model.register_output(atmos_1_point_0g)

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
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    # cl0=[0.72, 0., 0., 0.]
    cl0=[0.63, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0.55, 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    plus_1_point_0_g_elevator = system_model.create_input('plus_1_point_0_g_elevator', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    vlm_outputs = vlm_model.evaluate(
    atmosphere=atmos_1_point_0g,
    ac_states=ac_states_1_point_0g,
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
    system_model.register_output(vlm_outputs)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    throttle_plus_1_point_0g = system_model.create_input('throttle_plus_1_point_0g', val = 0.5, shape=(1, ), dv_flag=True, lower=0., upper=1.)
    tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_1_point_0g',
        counter = 'plus_1_point_0g',
    )
    tbw_left_propulsion_outputs_plus_1_point_0g = tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = throttle_plus_1_point_0g)
    system_model.register_output(tbw_left_propulsion_outputs_plus_1_point_0g)
    # endregion

    # region Viscous drag
    # # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # # system_model.register_output('S_area',S_ref.value)
    # wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
    # tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
    #         name = 'tbw_viscous_drag_plus_1_point_0g',
    #         geometry_units='ft',
    #         counter = 'plus_1_point_0g', 
    # )
    # # wing_ref_area = S_ref.value  
    # # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    # chord_value = system_model.create_input('chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,)
    # # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    # #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    # system_model.register_output(tbw_viscous_drag_outputs_plus_1_point_0g)

    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 import Tbw_Viscous_Drag_Model
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
    wing_sweep_angle = system_model.create_input(name='wing_sweep_angle', shape=(1,), val=np.deg2rad(12.6),)
    chord_value = system_model.create_input('chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = chord_value, vlm_outputs = vlm_outputs, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2
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
    tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = vlm_outputs, tbw_viscous_drag = tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_L_over_D_outputs_plus_1_point_0g)

    plus_1_point_0g_trim_variables = sizing_1_point_0g_condition.assemble_trim_residual(
    mass_properties=[tbw_mass_properties],
    # mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[vlm_outputs, tbw_left_propulsion_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs],
    ac_states=ac_states_1_point_0g,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(plus_1_point_0g_trim_variables)
            
    system_model.add_constraint(plus_1_point_0g_trim_variables.accelerations, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.du_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dv_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dw_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dp_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dq_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dr_dt, equals=0.)
    # system_model.add_objective(plus_1_point_0g_trim_variables.accelerations, scaler=1e-3)
    # system_model.add_objective(vlm_outputs.L_over_D, scaler = -1)
    system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)

    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = 'Forward_geo_dv_full_connection', analytics=True)
    sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)

    print('Total Forces: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('du_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['pitch_angle_1_point_0g'])
    print('VLM F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['plus_1_point_0g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('VLM wing_area_vlm', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.panel_area_wing_mesh_plus_1_point_0g'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('VLM wing_AR_vlm', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.wing_AR_vlm'])
    print('VLM total_lift', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])
    print('VLM total_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_drag'])
    # exit()
    prob = CSDLProblem(problem_name='tbw_trim_geo_dv_full_connections', simulator=sim)
    optimizer = SLSQP(prob, maxiter=100, ftol=1E-5)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-7)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-6)
    optimizer.solve()
    optimizer.print_results()

    print('Total Forces: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['pitch_angle_1_point_0g'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('VLM F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['plus_1_point_0g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('VLM total_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_drag'])
    print('qBar_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Wing Span', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_span_dv'])
    print('wing tip chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_tip_chord_left_dv'])
    print('wing root chord', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_root_chord_dv'])
    print('wing mid chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_mid_chord_left_dv'])
    print('Wing Fuselage Connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_fuselage_connection'])
    print('gearpod left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_left_strut_connection'])
    print('gearpod right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_right_strut_connection'])
    print('wing left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_left_strut_connection'])
    print('wing right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_right_strut_connection'])
    print('wing_AR', sim['plus_1_point_0_g_tbw_area_model.wing_AR'])
    print('wing_AR_vlm', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.wing_AR_vlm'])

twist_sweep_geo_dv = True # span and chord and twist and sweep working

if twist_sweep_geo_dv:

    from examples.advanced_examples.outputs_80.ex_tbw_geometry_trial_1 import (wing_meshes, htail_meshes, wing,  
                                                                    wing_twist_coefficients, wing_sweep_coefficients, 
                                                                    left_strut_sweep_coefficients, right_strut_sweep_coefficients, 
                                                                    left_jury_translation_x_coefficients, right_jury_translation_x_coefficients,
                                                                    left_strut_meshes, right_strut_meshes, wing_span_dv, wing_mid_chord_left_dv, 
                                                                    wing_tip_chord_left_dv, wing_root_chord_dv, S_ref_area, wing_AR, h_tail_area, strut_area,
                                                                    left_jury_meshes, right_jury_meshes, wing_sweep_angle,angle_beta,
                                                                    vtail_meshes, system_model, geometry) # span and chord and twist and sweep working? 

    t2 = time.time()
    t3 = time.time()
    caddee = cd.CADDEE()

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
    # from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l
    # wing_ref_area = 1478 # 1477.109999845 ft^2 = 137.2280094 m^2
    # # wing_ref_area = S_ref.value  
    # area_value = system_model.create_input('area_plus_1_point_0g', val=wing_ref_area)

    # tbw_wt = TBWMassProperties_m3l(
    # name='TBW_Mass_Properties',
    # exclude_wing = False,
    # full_wing = True,
    # geometry_units='ft',
    # )
    # tbw_mass_properties= tbw_wt.evaluate(area = area_value)
    # system_model.register_output(tbw_mass_properties)

    from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

    tbw_wt = TBWMassProperties_m3l(
    name='TBW_Mass_Properties',
    exclude_wing = False,
    full_wing = True,
    geometry_units='ft',
    )
    tbw_mass_properties= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
    system_model.register_output(tbw_mass_properties)
    # endregion




    sizing_1_point_0g_condition = cd.CruiseCondition(
    name='plus_1_point_0g_sizing',
    stability_flag=False,
    )

    h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
    # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=43000) # in 'ft'
    M_1_point_0g = system_model.create_input('mach_1_point_0g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=23380342.065) # in 'ft'
    r_1_point_0g = system_model.create_input('range_1_point_0g', val=12006064000) # in 'm'
    # r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in 'nautical miles'
    theta_1_point_0g = system_model.create_input('pitch_angle_1_point_0g', val=np.deg2rad(0.), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # ac sates + atmosFalse
    ac_states_1_point_0g, atmos_1_point_0g = sizing_1_point_0g_condition.evaluate(mach_number=M_1_point_0g, pitch_angle=theta_1_point_0g, cruise_range=r_1_point_0g, altitude=h_1_point_0g)
    system_model.register_output(ac_states_1_point_0g)
    system_model.register_output(atmos_1_point_0g)

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
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # OutputsVLM = None,
    # mesh_unit='m',
    cl0=[0.73, 0., 0., 0.]
    # cl0=[0.63, 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0.55, 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    plus_1_point_0_g_elevator = system_model.create_input('plus_1_point_0_g_elevator', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    vlm_outputs = vlm_model.evaluate(
    atmosphere=atmos_1_point_0g,
    ac_states=ac_states_1_point_0g,
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
    system_model.register_output(vlm_outputs)
    # endregion

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    throttle_plus_1_point_0g = system_model.create_input('throttle_plus_1_point_0g', val = 0.5, shape=(1, ), dv_flag=True, lower=0., upper=1.)
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
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_1 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_51_sweep_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 import Tbw_Viscous_Drag_Model
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
    # wing_sweep_angle = system_model.create_input(name='wing_sweep_angle', shape=(1,), val=np.deg2rad(12.6),)
    chord_value = system_model.create_input('chord_plus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, ) # tbw_viscous_drag_try_model_sweep_2
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                             tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                             chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                             h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                             sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g,) # tbw_viscous_drag_model_51_sweep_1
    tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = chord_value, vlm_outputs = vlm_outputs, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = wing_sweep_angle, mach_number = M_1_point_0g, angle_beta = angle_beta,) # tbw_viscous_drag_model_51_sweep_2
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
    tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = vlm_outputs, tbw_viscous_drag = tbw_viscous_drag_outputs_plus_1_point_0g)
    # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_L_over_D_outputs_plus_1_point_0g)

    plus_1_point_0g_trim_variables = sizing_1_point_0g_condition.assemble_trim_residual(
    mass_properties=[tbw_mass_properties],
    # mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[vlm_outputs, tbw_left_propulsion_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs, tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs],
    ac_states=ac_states_1_point_0g,
    load_factor=1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(plus_1_point_0g_trim_variables)
            
    system_model.add_constraint(plus_1_point_0g_trim_variables.accelerations, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.du_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dv_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dw_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dp_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dq_dt, equals=0.)
    # system_model.add_constraint(plus_1_point_0g_trim_variables.dr_dt, equals=0.)
    # system_model.add_objective(plus_1_point_0g_trim_variables.accelerations, scaler=1e-3)
    # system_model.add_objective(vlm_outputs.L_over_D, scaler = -1)
    # system_model.add_constraint(tbw_viscous_drag_outputs_plus_1_point_0g.cf_wave_1, equals = 0, scaler = 1.e-4)
    # system_model.add_constraint(tbw_viscous_drag_outputs_plus_1_point_0g.cf_wave_1, scaler = 1.e2)
    system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
    
    caddee_csdl_model = system_model.assemble_csdl()
    sim = Simulator(caddee_csdl_model,name = 'Forward_geo_dv_full_connection', analytics=True)
    sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)

    print('Total Forces: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('area fraction weights',sim['TBW_Mass_Properties.area_fraction_weights'])
    print('strut area fraction weights',sim['TBW_Mass_Properties.strut_area_fraction_weights'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('du_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['pitch_angle_1_point_0g'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('VLM F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['plus_1_point_0g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('VLM cd_induced', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('VLM total_lift', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])
    print('VLM total_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_drag'])
    # print('wing sweep angle radians', sim[wing_sweep_angle.operation.name + "." + wing_sweep_angle.name])
    print('wing sweep angle radians', sim[wing_sweep_angle.name])
    print('angle_beta', sim[angle_beta.name])
    # exit()
    prob = CSDLProblem(problem_name='tbw_trim_geo_dv_full_connections', simulator=sim)
    optimizer = SLSQP(prob, maxiter=100, ftol=1E-5)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-7)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-6)
    optimizer.solve()
    optimizer.print_results()

    print('Total Forces: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('area fraction weights',sim['TBW_Mass_Properties.area_fraction_weights'])
    print('strut area fraction weights',sim['TBW_Mass_Properties.strut_area_fraction_weights'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['pitch_angle_1_point_0g'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('VLM F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['plus_1_point_0g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM total_CD', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('VLM total_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_drag'])
    print('qBar_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('VLM cd_induced_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('Wing Span', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_span_dv'])
    print('wing tip chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_tip_chord_left_dv'])
    print('wing root chord', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_root_chord_dv'])
    print('wing mid chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_mid_chord_left_dv'])
    print('Wing Fuselage Connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_fuselage_connection'])
    print('gearpod left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_left_strut_connection'])
    print('gearpod right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_right_strut_connection'])
    print('wing left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_left_strut_connection'])
    print('wing right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_right_strut_connection'])
    print('wing_AR', sim['plus_1_point_0_g_tbw_area_model.wing_AR'])
    print('wing_twist_coefficients', sim[wing_twist_coefficients.name])
    print('wing sweep angle radians', sim[wing_sweep_angle.name])
    # print('wing_sweep_coefficients', sim[wing_sweep_coefficients.name])
    # print('left_strut_sweep_coefficients', sim[left_strut_sweep_coefficients.operation.name + "." + left_strut_sweep_coefficients.name])
    # print('left_strut_sweep_coefficients', sim[right_strut_sweep_coefficients.operation.name + "." + right_strut_sweep_coefficients.name])
    # print('left_jury_sweep_coefficients', sim[left_jury_translation_x_coefficients.operation.name + "." + left_jury_translation_x_coefficients.name])
    # print('right_jury_sweep_coefficients', sim[right_jury_translation_x_coefficients.operation.name + "." + right_jury_translation_x_coefficients.name])
    # print('wing sweep angle radians', sim[wing_sweep_angle.operation.name + "." + wing_sweep_angle.name])
