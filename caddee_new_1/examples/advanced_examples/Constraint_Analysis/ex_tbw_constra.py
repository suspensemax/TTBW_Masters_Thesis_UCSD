"Working - Trim (all) -> VLM (3 DOF)"

# Module imports
import numpy as np
import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from VAST import FluidProblem, VASTFluidSover
import time 
import pickle
import csdl

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline



st = time.time()
caddee = cd.CADDEE()

from examples.advanced_examples.Constraint_Analysis.tbw_geometry_constraint import (wing_meshes, htail_meshes, wing_span_dv, wing_root_chord_dv, 
                                                                        wing_mid_chord_left_dv, wing_tip_chord_left_dv,    
                                                                left_strut_meshes, right_strut_meshes, 
                                                                left_jury_meshes, right_jury_meshes, 
                                                                vtail_meshes, S_ref_area, system_model, wing_AR, 
                                                                h_tail_area, jury_area, strut_area, angle_beta,
                                                                wing_box_beam_mesh, left_strut_box_beam_mesh, right_strut_box_beam_mesh,
                                                                left_jury_box_beam_mesh, right_jury_box_beam_mesh, num_wing_beam,
                                                                num_strut_beam, num_jury_beam)



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
# region Flight Condition
h_tbw = system_model.create_input('h_tbw', val=13106.) # in 'm'
M_tbw = system_model.create_input('M_tbw', val=0.70)
r_tbw = system_model.create_input('r_tbw', val=6000000.) # in 'm'
# r_tbw = system_model.create_input('r_tbw', val=6482000.) # in 'm'
theta_tbw = system_model.create_input('theta_tbw', val=0, dv_flag=False, lower=np.deg2rad(-20), upper=np.deg2rad(20))

sizing_condition = cd.CruiseCondition(
    name='sizing_condition',
    stability_flag=False,
    )
ac_states_1_point_0g, atmos_1_point_0g = sizing_condition.evaluate(
    mach_number=M_tbw, pitch_angle=theta_tbw, 
    cruise_range=r_tbw, 
    altitude=h_tbw
    )
system_model.register_output(ac_states_1_point_0g)
system_model.register_output(atmos_1_point_0g)
# endregion

# -- Analysis --

# region Weights
from examples.advanced_examples.Constraint_Analysis.caddeemodel_ouu import TBW_Mass_Properties
# wing_ref_area = 1477.109999845
# area_value = system_model.create_input('area_value', val=wing_ref_area)
lbs_to_kg =  1 / 2.205
fuselage_mass = system_model.create_input('fuselage_mass', val=120000.*lbs_to_kg)
tail_mass = system_model.create_input('tail_mass', val=3000.*lbs_to_kg)
payload_mass = system_model.create_input('payload_mass', val=3580.*lbs_to_kg)

tbw_mass_properties = TBW_Mass_Properties(
    name='tbw_mass_properties',
    # exclude_wing = False,
    # full_wing = True,
    geometry_units='ft'
    )
tbw_mass_outputs= tbw_mass_properties.evaluate(
    area = tbw_area_outputs_plus_1_point_0g.wing_area,
    fuselage_mass=fuselage_mass,
    tail_mass=tail_mass,
    payload_mass=payload_mass,
    tbwarea = tbw_area_outputs_plus_1_point_0g, 
    wing_span_dv = wing_span_dv,
    )
system_model.register_output(tbw_mass_outputs)
# endregion


# region tbw weights
from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

tbw_wt = TBWMassProperties_m3l(
name='TBW_Mass_Properties',
exclude_wing = True,
full_wing = False,
geometry_units='ft',
)
tbw_mass_properties_1= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
system_model.register_output(tbw_mass_properties_1)
# endregion


# # region Prop: CSDLcycle
# engpos = system_model.create_input('engpos', val=4.)
# throttle = system_model.create_input('throttle', val = 0.2434, shape=(1, ), dv_flag=False, lower=0., upper=1.)
# Inlet_W = system_model.create_input('Inlet_W', val=170., shape=(1,), dv_flag=True, lower=110., upper=180.)

# from pycycle_caddee import CSDLcycle
# tbw_propulsion = CSDLcycle(
#     name='tbw_propulsion',
#     num_nodes=1,
#     )
# tbw_propulsion_outputs = tbw_propulsion.evaluate(
#     # ac_states = ac_states,
#     # atmosphere = atmos,
#     eng_pos = engpos,
#     throttle = throttle,
#     Inlet_W = Inlet_W,
#     # mix_ratio = mix_ratio,
#     # PR = PR,
#     )
# system_model.register_output(tbw_propulsion_outputs)
# # endregion

# region Propulsion Solvers
from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
throttle_plus_1_point_0g = system_model.create_input('throttle_plus_1_point_0g', val = 0.23, shape=(1, ), dv_flag=True, lower=0., upper=1.)
tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
    name='TBW_Propulsion_plus_1_point_0g',
    counter = 'plus_1_point_0g',
)
tbw_left_propulsion_outputs_plus_1_point_0g = tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = throttle_plus_1_point_0g)
system_model.register_output(tbw_left_propulsion_outputs_plus_1_point_0g)
SFC_tbw = system_model.create_input(name = 'SFC_tbw', val = 1.3014*(10**(-4)), shape=(1,))
# endregion



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
cl0=[0.72, 0., 0., 0.]
# cl0=[0., 0., 0., 0.]
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
eval_pt=tbw_mass_properties_1.cg_vector,
)
system_model.register_output(vlm_outputs)
# endregion

# region Aero: Random Drag
# from caddeemodel_ouu import TBW_Random_Drag_Model
from examples.advanced_examples.Constraint_Analysis.caddeemodel_ouu import TBW_Random_Drag_Model
    
# wing_ref_area = S_ref.value  
# area_value = system_model.create_input('area_tbw', val=wing_ref_area)
wing_ref_chord = 110.286 
chord_value = system_model.create_input('chord_tbw', val=wing_ref_chord)
# random
wing_viscous_cf = system_model.create_input('wing_viscous_cf', val=0.0135)
wing_interference_cf = system_model.create_input('wing_interference_cf', val=0.0015)
wing_wave_cf = system_model.create_input('wing_wave_cf', val=0.0005)

tbw_random_drag_model = TBW_Random_Drag_Model(
    name = 'tbw_random_drag_model',
    geometry_units='ft',
    counter = 'normal', 
    )
tbw_random_drag_model_outputs = tbw_random_drag_model.evaluate(
    atmos = atmos_1_point_0g, ac_states = ac_states_1_point_0g, 
    area = tbw_area_outputs_plus_1_point_0g.wing_area, 
    chord = chord_value, vlm_outputs = vlm_outputs,
    wing_viscous_cf = wing_viscous_cf,
    wing_interference_cf = wing_interference_cf,
    wing_wave_cf = wing_wave_cf,
    )
system_model.register_output(tbw_random_drag_model_outputs)
# endregion

# -- Design --

# region constraints analysis
# from caddeemodel_ouu import TBW_Constraints_Analysis_Model
from examples.advanced_examples.Constraint_Analysis.caddeemodel_ouu import TBW_Constraints_Analysis_Model

tbw_constraints_model = TBW_Constraints_Analysis_Model(
    name = 'tbw_constraints_model',
    )
tbw_constraints_model_outputs = tbw_constraints_model.evaluate(
    atmos = atmos_1_point_0g, 
    # ac_states = ac_states_tbw,
    tbwarea = tbw_area_outputs_plus_1_point_0g, 
    tbw_mass = tbw_mass_outputs, tbw_prop = tbw_left_propulsion_outputs_plus_1_point_0g,
    vlm_outputs = vlm_outputs, tbw_random_drag = tbw_random_drag_model_outputs,
    wing_span_dv = wing_span_dv, V = M_tbw, 
    SFC_tbw= SFC_tbw,
    )
system_model.register_output(tbw_constraints_model_outputs)
# endregion

# -- Optimzation --

# region postprocess
system_model.add_constraint(tbw_constraints_model_outputs.Constraint_Stall, upper = 0.)
# system_model.add_constraint(tbw_constraints_model_outputs.Constraint_Climb, lower = 0., upper = 1.)
# system_model.add_constraint(tbw_constraints_model_outputs.Constraint_Maneuver, lower = 0., upper = 1.)
system_model.add_constraint(tbw_constraints_model_outputs.Constraint_Maneuver, equals=0.)
# system_model.add_objective(tbw_constraints_model_outputs.Constraint_Stall, scaler = 1)
# system_model.add_objective(tbw_constraints_model_outputs.Constraint_Climb, scaler = 1)
# system_model.add_objective(tbw_constraints_model_outputs.Constraint_Maneuver, scaler = -1)

system_model.add_objective(tbw_constraints_model_outputs.Constr_Obj, scaler = 1)

caddee_csdl_model = system_model.assemble_csdl()
sim = Simulator(caddee_csdl_model,name = 'TBW_Constraints_Analysis', analytics=True)
sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])
# sim['vlm_model.vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])

# region optimization 
sim.run()
cd.print_caddee_outputs(system_model, sim, compact_print=True)
# exit()
prob = CSDLProblem(problem_name='tbw_constraints_analysis', simulator=sim)
optimizer = SLSQP(prob, maxiter=200, ftol=1E-5)
optimizer.solve()
optimizer.print_results()
# endregion

# region print
print("----------------- design variables -----------------")
print('wing_span_dv: ', sim['wing_span_dv'])
print('wing_tip_chord_left_dv: ', sim['wing_tip_chord_left_dv'])
print('wing_mid_chord_left_dv: ', sim['wing_mid_chord_left_dv'])
print('wing_root_chord_dv: ', sim['wing_root_chord_dv'])

print('Inlet_W: ', sim['tbw_propulsion.Inlet_W'])

# print("----------------- propulsion -----------------")
# print('TSFC: ', sim['tbw_propulsion.TSFC'])
# print('Thurst: ', sim['tbw_propulsion.Thurst'])

print("----------------- acstates & atmoshpere -----------------")
print('u: ', sim['sizing_condition.u'])
print('v: ', sim['sizing_condition.v'])
print('w: ', sim['sizing_condition.w'])
print('density: ', sim['sizing_condition.atmosphere_model.density'])

print("----------------- constraints -----------------")
print('W_S: ', sim['tbw_constraints_model.W_S'])
print('T_W: ', sim['tbw_constraints_model.T_W'])
print('Constraint_Stall: ', sim['tbw_constraints_model.Constraint_Stall'])
print('Constraint_Climb: ', sim['tbw_constraints_model.Constraint_Climb'])
print('Constraint_Maneuver: ', sim['tbw_constraints_model.Constraint_Maneuver'])
print('objective: ', sim['tbw_constraints_model.Constr_Obj'])

print('wing_area: ', sim['nomral_tbw_area_model.wing_area_value'])
print('S: ', sim['tbw_constraints_model.area'])
print('S_Me: ', sim['tbw_constraints_model.S_Me'])
print('T: ', sim['tbw_constraints_model.thrust'])
print('W: ', sim['tbw_constraints_model.weight'])
print('vlm_drag: ', sim['tbw_constraints_model.vlm_drag'])
print('ran_drag: ', sim['tbw_constraints_model.ran_drag'])
print('rho: ', sim['tbw_constraints_model.rho'])
print('V: ', sim['tbw_constraints_model.V'])
print('span: ', sim['tbw_constraints_model.span'])
print('CD0: ', sim['tbw_constraints_model.CD0'])
print('AR: ', sim['tbw_constraints_model.AR'])

print('C1: ', sim['tbw_constraints_model.C1'])
print('A2: ', sim['tbw_constraints_model.A2'])
print('B2: ', sim['tbw_constraints_model.B2'])
print('C2: ', sim['tbw_constraints_model.C2'])
print('A3: ', sim['tbw_constraints_model.A3'])
print('B3: ', sim['tbw_constraints_model.B3'])

print('T_W_Climb: ', sim['tbw_constraints_model.T_W_Climb'])
print('T_W_Maneuver: ', sim['tbw_constraints_model.T_W_Maneuver'])

print("time", time.time() - st)
# endregion

# region plot
m = 50
RV1 = np.reshape(np.linspace(140., 200., m), (m,1)) # span
RV2 = np.reshape(np.linspace(3.37, 4.13, m), (m,1)) # chord
RV3 = np.reshape(np.linspace(8.70, 10.68, m), (m,1)) # chord
RV4 = np.reshape(np.linspace(9.64, 12.79, m), (m,1)) # chord
RV5 = np.reshape(np.linspace(110., 180., m), (m,1)) # inlet W
Results_vector = np.zeros((m, 4))
S_vector = np.zeros((m, 1))
Results_C1 = 0.

for i in range(m):
    sim['wing_span_dv'] = RV1[i]
    sim['wing_tip_chord_left_dv'] = RV2[i]
    sim['wing_mid_chord_left_dv'] = RV3[i]
    sim['wing_root_chord_dv'] = RV4[i]
    sim['Inlet_W'] = RV5[i]

    sim.run()

    Results_vector[i,0] = sim['tbw_constraints_model.W_S']
    Results_vector[i,1] = sim['tbw_constraints_model.T_W']
    Results_C1 = sim['tbw_constraints_model.C1'] # stall speed
    Results_vector[i,2] = sim['tbw_constraints_model.T_W_Climb']
    Results_vector[i,3] = sim['tbw_constraints_model.T_W_Maneuver']

    S_vector[i,0] = sim['tbw_constraints_model.S_Me']

# plt.figure(1)
# plt.plot(S_vector[:,0], Results_vector[:,0], color='r', linewidth=1.0, label='W_S')
# plt.xlabel('S')
# plt.ylabel('W/S')

# plt.figure(2)
# plt.plot(S_vector[:,0], Results_vector[:,1], color='b', linewidth=1.0, label='T_W')
# plt.xlabel('S')
# plt.ylabel('T/W')

plt.figure(3)
plt.axvline(x = Results_C1, color = 'b', label='Stall Speed')
# plt.plot(Results_vector[:,0], Results_vector[:,2], color='r', linewidth=1.0, label='Climb')
plt.plot(Results_vector[:,0], Results_vector[:,3], color='g', linewidth=1.0, label='Maneuver')
# plt.plot(421.46, 0.248, 'o', label='Optimal Design')
# plt.plot(421.4513080111, 0.24837067, 'o', label='Optimal Design')
# plt.plot(413.33367588, 0.2583025, 'o', label='Optimal Design')
plt.plot(271.36568976, 0.43945891, 'o', label='Optimal Design')

# plt.annotate('421.46, 0.248', xy=(421.46, 0.248), 
#              xytext=(375, 0.23),
#             #  arrowprops=dict(facecolor='black', shrink=0.01),
#              )
# plt.text(421.46, 0.248, str('421.46, 0.248'))
plt.xlabel('W/S')
plt.ylabel('T/W')
plt.legend()
plt.show()
# endregion
