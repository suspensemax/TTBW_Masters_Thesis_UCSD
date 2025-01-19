'''Example 2 : Description of example 2'''
import numpy as np
import caddee.api as cd 
import lsdo_geo as lg
import m3l
from python_csdl_backend import Simulator
from caddee import IMPORTS_FILES_FOLDER
import array_mapper as am
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from lsdo_rotor.core.pitt_peters.pitt_peters_m3l import PittPeters, PittPetersMesh


caddee = cd.CADDEE()

# Import representation and the geometry from another file for brevity
from examples.advanced_examples.TC2_problem.ex_tc2_geometry_setup import lpc_rep, lpc_param, wing_camber_surface, htail_camber_surface, \
    wing_vlm_mesh_name, htail_vlm_mesh_name, \
    pp_disk_in_plane_x, pp_disk_in_plane_y, pp_disk_origin, pp_disk, \
    rlo_in_plane_x, rlo_in_plane_y, rlo_origin, rlo_disk, \
    rli_in_plane_x, rli_in_plane_y, rli_origin, rli_disk, \
    rri_in_plane_x, rri_in_plane_y, rri_origin, rri_disk, \
    rro_in_plane_x, rro_in_plane_y, rro_origin, rro_disk, \
    flo_in_plane_x, flo_in_plane_y, flo_origin, flo_disk, \
    fli_in_plane_x, fli_in_plane_y, fli_origin, fli_disk, \
    fri_in_plane_x, fri_in_plane_y, fri_origin, fri_disk, \
    fro_in_plane_x, fro_in_plane_y, fro_origin, fro_disk \

# set system representation and parameterization
caddee.system_representation = lpc_rep
caddee.system_parameterization = lpc_param

# system model
caddee.system_model = system_model = cd.SystemModel()

system_m3l_model = m3l.Model()

# region sizing
# Battery sizing
battery_component = cd.Component(name='battery')
simple_battery_sizing = cd.SimpleBatterySizingM3L(component=battery_component)

simple_battery_sizing.set_module_input('battery_mass', val=800, dv_flag=False, lower=600, scaler=1e-3)
simple_battery_sizing.set_module_input('battery_position', val=np.array([3.4, 0, 0.5]))
simple_battery_sizing.set_module_input('battery_energy_density', val=400)

battery_mass, cg_battery, I_battery = simple_battery_sizing.evaluate()
system_m3l_model.register_output(battery_mass)
system_m3l_model.register_output(cg_battery)
system_m3l_model.register_output(I_battery)


# M4 regressions
m4_regression = cd.M4RegressionsM3L(exclude_wing=False)

mass_m4, cg_m4, I_m4 = m4_regression.evaluate(battery_mass=battery_mass)
system_m3l_model.register_output(mass_m4)
system_m3l_model.register_output(cg_m4)
system_m3l_model.register_output(I_m4)

total_mass_properties = cd.TotalMassPropertiesM3L()
total_mass, total_cg, total_inertia = total_mass_properties.evaluate(battery_mass, mass_m4, cg_battery, cg_m4, I_battery, I_m4)

system_m3l_model.register_output(total_mass)
system_m3l_model.register_output(total_cg)
system_m3l_model.register_output(total_inertia)
# endregion

# region BEM meshes

airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

pusher_bem_mesh = BEMMesh(
    airfoil='NACA_4412', 
    num_blades=4,
    num_radial=25,
    num_tangential=1,
    mesh_units='ft',
    use_airfoil_ml=False,
    airfoil_polar=None,
)
bem_mesh_lift = BEMMesh(
    num_blades=2,
    num_radial=25,
    num_tangential=25,
    airfoil='NACA_4412',
    use_airfoil_ml=False,
    mesh_units='ft',
    airfoil_polar=None,
)

pitt_peters_mesh_lift = PittPetersMesh(
    num_blades=2,
    num_radial=25,
    num_tangential=25,
    airfoil='NACA_4412',
    use_airfoil_ml=False,
    mesh_units='ft',
)
# endregion

design_scenario = cd.DesignScenario(name='quasi_steady_transition')

# # region qst 1
# qst_1 = cd.CruiseCondition(name='qst_1')
# qst_1.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_1.set_module_input('pitch_angle', val=0-0.0134037)
# qst_1.set_module_input('mach_number', val=0.00029412)
# qst_1.set_module_input('altitude', val=300)
# qst_1.set_module_input(name='range', val=20)
# qst_1.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_1_ac_states = qst_1.evaluate_ac_states()


# # vlm_model = VASTFluidSover(
# #     surface_names=[
# #         f"{wing_vlm_mesh_name}_qst_1",
# #         f"{htail_vlm_mesh_name}_qst_1",
# #     ],
# #     surface_shapes=[
# #         (1, ) + wing_camber_surface.evaluate().shape[1:],
# #         (1, ) + htail_camber_surface.evaluate().shape[1:],
# #     ],
# #     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
# #     mesh_unit='ft',
# #     cl0=[0.43, 0]
# # )

# # forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)
# # system_m3l_model.register_output(vlm_forces, design_condition=qst_1)
# # system_m3l_model.register_output(vlm_moments, design_condition=qst_1)

# # pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# # pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=0, upper=2000, scaler=1e-3)
# # pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# rlo_bem_model = BEM(disk_prefix='qst_1_rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=bem_mesh_lift)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# rli_bem_model = BEM(disk_prefix='qst_1_rli_disk', blade_prefix='rli', component=rli_disk, mesh=bem_mesh_lift)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# rri_bem_model = BEM(disk_prefix='qst_1_rri_disk', blade_prefix='rri', component=rri_disk, mesh=bem_mesh_lift)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# rro_bem_model = BEM(disk_prefix='qst_1_rro_disk', blade_prefix='rro', component=rro_disk, mesh=bem_mesh_lift)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# flo_bem_model = BEM(disk_prefix='qst_1_flo_disk', blade_prefix='flo', component=flo_disk, mesh=bem_mesh_lift)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# fli_bem_model = BEM(disk_prefix='qst_1_fli_disk', blade_prefix='fli', component=fli_disk, mesh=bem_mesh_lift)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# fri_bem_model = BEM(disk_prefix='qst_1_fri_disk', blade_prefix='fri', component=fri_disk, mesh=bem_mesh_lift)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# fro_bem_model = BEM(disk_prefix='qst_1_fro_disk', blade_prefix='fro', component=fro_disk, mesh=bem_mesh_lift)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_1_ac_states, design_condition=qst_1)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_1_ac_states, 
#     design_condition=qst_1
# )
# system_m3l_model.register_output(inertial_forces, qst_1)
# system_m3l_model.register_output(inertial_moments, qst_1)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     rlo_bem_forces, 
#     rlo_bem_moments, 
#     rli_bem_forces, 
#     rli_bem_moments,
#     rri_bem_forces, 
#     rri_bem_moments, 
#     rro_bem_forces, 
#     rro_bem_moments,  
#     flo_bem_forces, 
#     flo_bem_moments, 
#     fli_bem_forces, 
#     fli_bem_moments,
#     fri_bem_forces, 
#     fri_bem_moments, 
#     fro_bem_forces, 
#     fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     # pp_bem_forces,
#     # pp_bem_moments,
#     # vlm_forces,
#     # vlm_moments,
#     design_condition=qst_1,
# )
# system_m3l_model.register_output(total_forces,qst_1)
# system_m3l_model.register_output(total_moments, qst_1)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_1_ac_states,
#     design_condition=qst_1,
# )
# system_m3l_model.register_output(trim_residual, qst_1)
# # endregion

# # region qst 2
# qst_2 = cd.CruiseCondition(name='qst_2')
# qst_2.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_2.set_module_input('pitch_angle', val=-0.04973228)
# qst_2.set_module_input('mach_number', val=0.06489461)
# qst_2.set_module_input('altitude', val=300)
# qst_2.set_module_input(name='range', val=20)
# qst_2.set_module_input(name='observer_location', val=np.array([0, 0, 0]))


# qst_2_ac_states = qst_2.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_2",
#         f"{htail_vlm_mesh_name}_qst_2",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_2)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_2)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=50, dv_flag=True, lower=800, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# rlo_bem_model = PittPeters(disk_prefix='qst_2_rlo_disk',  blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# rli_bem_model = PittPeters(disk_prefix='qst_2_rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# rri_bem_model = PittPeters(disk_prefix='qst_2_rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# rro_bem_model = PittPeters(disk_prefix='qst_2_rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# flo_bem_model = PittPeters(disk_prefix='qst_2_flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# fli_bem_model = PittPeters(disk_prefix='qst_2_fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# fri_bem_model = PittPeters(disk_prefix='qst_2_fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# fro_bem_model = PittPeters(disk_prefix='qst_2_fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_2_ac_states, design_condition=qst_2)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_2_ac_states, 
#     design_condition=qst_2
# )
# system_m3l_model.register_output(inertial_forces, qst_2)
# system_m3l_model.register_output(inertial_moments, qst_2)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     rlo_bem_forces, 
#     rlo_bem_moments, 
#     rli_bem_forces, 
#     rli_bem_moments,
#     rri_bem_forces, 
#     rri_bem_moments, 
#     rro_bem_forces, 
#     rro_bem_moments,  
#     flo_bem_forces, 
#     flo_bem_moments, 
#     fli_bem_forces, 
#     fli_bem_moments,
#     fri_bem_forces, 
#     fri_bem_moments, 
#     fro_bem_forces, 
#     fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_2,
# )
# system_m3l_model.register_output(total_forces,qst_2)
# system_m3l_model.register_output(total_moments, qst_2)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_2_ac_states,
#     design_condition=qst_2,
# )
# system_m3l_model.register_output(trim_residual, qst_2)
# # endregion

# # region qst 3
# qst_3 = cd.CruiseCondition(name='qst_3')
# qst_3.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_3.set_module_input('pitch_angle', val=0.16195989)
# qst_3.set_module_input('mach_number', val=0.11471427)
# qst_3.set_module_input('altitude', val=300)
# qst_3.set_module_input(name='range', val=20)
# qst_3.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_3_ac_states = qst_3.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_3",
#         f"{htail_vlm_mesh_name}_qst_3",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_3)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_3)

# pp_bem_model = PittPeters(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# rlo_bem_model = PittPeters(disk_prefix='qst_3_rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# rli_bem_model = PittPeters(disk_prefix='qst_3_rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# rri_bem_model = PittPeters(disk_prefix='qst_3_rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# rro_bem_model = PittPeters(disk_prefix='qst_3_rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# flo_bem_model = PittPeters(disk_prefix='qst_3_flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# fli_bem_model = PittPeters(disk_prefix='qst_3_fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# fri_bem_model = PittPeters(disk_prefix='qst_3_fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# fro_bem_model = PittPeters(disk_prefix='qst_3_fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=20, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_3_ac_states, design_condition=qst_3)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_3_ac_states, 
#     design_condition=qst_3
# )
# system_m3l_model.register_output(inertial_forces, qst_3)
# system_m3l_model.register_output(inertial_moments, qst_3)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     rlo_bem_forces, 
#     rlo_bem_moments, 
#     rli_bem_forces, 
#     rli_bem_moments,
#     rri_bem_forces, 
#     rri_bem_moments, 
#     rro_bem_forces, 
#     rro_bem_moments,  
#     flo_bem_forces, 
#     flo_bem_moments, 
#     fli_bem_forces, 
#     fli_bem_moments,
#     fri_bem_forces, 
#     fri_bem_moments, 
#     fro_bem_forces, 
#     fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_3,
# )
# system_m3l_model.register_output(total_forces,qst_3)
# system_m3l_model.register_output(total_moments, qst_3)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_3_ac_states,
#     design_condition=qst_3,
# )
# system_m3l_model.register_output(trim_residual, qst_3)
# # endregion

# # region qst 4
# qst_4 = cd.CruiseCondition(name='qst_4')
# qst_4.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_4.set_module_input('pitch_angle', val=0.10779469, dv_flag=True, lower=0.10779469 - np.deg2rad(2))
# qst_4.set_module_input('mach_number', val=0.13740796)
# qst_4.set_module_input('altitude', val=300)
# qst_4.set_module_input(name='range', val=20)
# qst_4.set_module_input(name='observer_location', val=np.array([0, 0, 0]))


# qst_4_ac_states = qst_4.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_4",
#         f"{htail_vlm_mesh_name}_qst_4",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_4)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_4)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# rlo_bem_model = PittPeters(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# rli_bem_model = PittPeters(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# rri_bem_model = PittPeters(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# rro_bem_model = PittPeters(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# flo_bem_model = PittPeters(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# fli_bem_model = PittPeters(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# fri_bem_model = PittPeters(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# fro_bem_model = PittPeters(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_4_ac_states, design_condition=qst_4)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_4_ac_states, 
#     design_condition=qst_4
# )
# system_m3l_model.register_output(inertial_forces, qst_4)
# system_m3l_model.register_output(inertial_moments, qst_4)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     rlo_bem_forces, 
#     rlo_bem_moments, 
#     rli_bem_forces, 
#     rli_bem_moments,
#     rri_bem_forces, 
#     rri_bem_moments, 
#     rro_bem_forces, 
#     rro_bem_moments,  
#     flo_bem_forces, 
#     flo_bem_moments, 
#     fli_bem_forces, 
#     fli_bem_moments,
#     fri_bem_forces, 
#     fri_bem_moments, 
#     fro_bem_forces, 
#     fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_4,
# )
# system_m3l_model.register_output(total_forces,qst_4)
# system_m3l_model.register_output(total_moments, qst_4)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_4_ac_states,
#     design_condition=qst_4,
# )
# system_m3l_model.register_output(trim_residual, qst_4)
# # endregion

# # region qst 5
# qst_5 = cd.CruiseCondition(name='qst_5')
# qst_5.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_5.set_module_input('pitch_angle', val=0.08224058, dv_flag=True, lower=0.08224058 - np.deg2rad(2.5))
# qst_5.set_module_input('mach_number', val=0.14708026)
# qst_5.set_module_input('altitude', val=300)
# qst_5.set_module_input(name='range', val=20)
# qst_5.set_module_input(name='observer_location', val=np.array([0, 0, 0]))


# qst_5_ac_states = qst_5.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_5",
#         f"{htail_vlm_mesh_name}_qst_5",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_5)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_5)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# rlo_bem_model = PittPeters(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# rli_bem_model = PittPeters(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# rri_bem_model = PittPeters(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# rro_bem_model = PittPeters(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# flo_bem_model = PittPeters(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# fli_bem_model = PittPeters(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# fri_bem_model = PittPeters(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# fro_bem_model = PittPeters(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=5, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_5_ac_states, design_condition=qst_5)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_5_ac_states, 
#     design_condition=qst_5
# )
# system_m3l_model.register_output(inertial_forces, qst_5)
# system_m3l_model.register_output(inertial_moments, qst_5)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     rlo_bem_forces, 
#     rlo_bem_moments, 
#     rli_bem_forces, 
#     rli_bem_moments,
#     rri_bem_forces, 
#     rri_bem_moments, 
#     rro_bem_forces, 
#     rro_bem_moments,  
#     flo_bem_forces, 
#     flo_bem_moments, 
#     fli_bem_forces, 
#     fli_bem_moments,
#     fri_bem_forces, 
#     fri_bem_moments, 
#     fro_bem_forces, 
#     fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_5,
# )
# system_m3l_model.register_output(total_forces,qst_5)
# system_m3l_model.register_output(total_moments, qst_5)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_5_ac_states,
#     design_condition=qst_5,
# )
# system_m3l_model.register_output(trim_residual, qst_5)
# # endregion

# # region qst 6
# qst_6 = cd.CruiseCondition(name='qst_6')
# qst_6.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_6.set_module_input('pitch_angle', val=0.06704556, dv_flag=True, lower=0.06704556 - np.deg2rad(1.5))
# qst_6.set_module_input('mach_number', val=0.15408429)
# qst_6.set_module_input('altitude', val=300)
# qst_6.set_module_input(name='range', val=20)
# qst_6.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_6_ac_states = qst_6.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_6",
#         f"{htail_vlm_mesh_name}_qst_6",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_6)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_6)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # rlo_bem_model = PittPeters(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# # rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # rli_bem_model = PittPeters(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# # rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # rri_bem_model = PittPeters(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# # rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # rro_bem_model = PittPeters(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# # rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # flo_bem_model = PittPeters(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# # flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # fli_bem_model = PittPeters(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# # fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # fri_bem_model = PittPeters(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# # fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# # fro_bem_model = PittPeters(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# # fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=10, upper=4000, scaler=1e-3)
# # fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_6_ac_states, design_condition=qst_6)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_6_ac_states, 
#     design_condition=qst_6
# )
# system_m3l_model.register_output(inertial_forces, qst_6)
# system_m3l_model.register_output(inertial_moments, qst_6)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     # rlo_bem_forces, 
#     # rlo_bem_moments, 
#     # rli_bem_forces, 
#     # rli_bem_moments,
#     # rri_bem_forces, 
#     # rri_bem_moments, 
#     # rro_bem_forces, 
#     # rro_bem_moments,  
#     # flo_bem_forces, 
#     # flo_bem_moments, 
#     # fli_bem_forces, 
#     # fli_bem_moments,
#     # fri_bem_forces, 
#     # fri_bem_moments, 
#     # fro_bem_forces, 
#     # fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_6,
# )
# system_m3l_model.register_output(total_forces, qst_6)
# system_m3l_model.register_output(total_moments, qst_6)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_6_ac_states,
#     design_condition=qst_6,
# )
# system_m3l_model.register_output(trim_residual, qst_6)
# # endregion

# # region qst 7
# qst_7 = cd.CruiseCondition(name='qst_7')
# qst_7.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_7.set_module_input('pitch_angle', val=0.05598293, dv_flag=True, lower=0.05598293-np.deg2rad(2))
# qst_7.set_module_input('mach_number', val=0.15983874)
# qst_7.set_module_input('altitude', val=300)
# qst_7.set_module_input(name='range', val=20)
# qst_7.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_7_ac_states = qst_7.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_7",
#         f"{htail_vlm_mesh_name}_qst_7",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_7)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_7)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # rlo_bem_model = BEM(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=rlo_bem_mesh)
# # rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # rli_bem_model = BEM(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=rli_bem_mesh)
# # rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # rri_bem_model = BEM(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=rri_bem_mesh)
# # rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # rro_bem_model = BEM(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=rro_bem_mesh)
# # rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # flo_bem_model = BEM(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=flo_bem_mesh)
# # flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # fli_bem_model = BEM(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=fli_bem_mesh)
# # fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # fri_bem_model = BEM(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=fri_bem_mesh)
# # fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# # fro_bem_model = BEM(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=fro_bem_mesh)
# # fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_7_ac_states, design_condition=qst_7)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_7_ac_states, 
#     design_condition=qst_7
# )
# system_m3l_model.register_output(inertial_forces, qst_7)
# system_m3l_model.register_output(inertial_moments, qst_7)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     # rlo_bem_forces, 
#     # rlo_bem_moments, 
#     # rli_bem_forces, 
#     # rli_bem_moments,
#     # rri_bem_forces, 
#     # rri_bem_moments, 
#     # rro_bem_forces, 
#     # rro_bem_moments,  
#     # flo_bem_forces, 
#     # flo_bem_moments, 
#     # fli_bem_forces, 
#     # fli_bem_moments,
#     # fri_bem_forces, 
#     # fri_bem_moments, 
#     # fro_bem_forces, 
#     # fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_7,
# )
# system_m3l_model.register_output(total_forces, qst_7)
# system_m3l_model.register_output(total_moments, qst_7)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_7_ac_states,
#     design_condition=qst_7,
# )
# system_m3l_model.register_output(trim_residual, qst_7)
# # endregion

# # region qst 8
# qst_8 = cd.CruiseCondition(name='qst_8')
# qst_8.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_8.set_module_input('pitch_angle', val=0.04712265, dv_flag=True, lower=0.04712265-np.deg2rad(3))
# qst_8.set_module_input('mach_number', val=0.16485417)
# qst_8.set_module_input('altitude', val=300)
# qst_8.set_module_input(name='range', val=20)
# qst_8.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_8_ac_states = qst_8.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_8",
#         f"{htail_vlm_mesh_name}_qst_8",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_8)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_8)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=3000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # rlo_bem_model = BEM(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=rlo_bem_mesh)
# # rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # rli_bem_model = BEM(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=rli_bem_mesh)
# # rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # rri_bem_model = BEM(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=rri_bem_mesh)
# # rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # rro_bem_model = BEM(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=rro_bem_mesh)
# # rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # flo_bem_model = BEM(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=flo_bem_mesh)
# # flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # fli_bem_model = BEM(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=fli_bem_mesh)
# # fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # fri_bem_model = BEM(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=fri_bem_mesh)
# # fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# # fro_bem_model = BEM(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=fro_bem_mesh)
# # fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# # fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_8_ac_states, design_condition=qst_8)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_8_ac_states, 
#     design_condition=qst_8
# )
# system_m3l_model.register_output(inertial_forces, qst_8)
# system_m3l_model.register_output(inertial_moments, qst_8)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     # rlo_bem_forces, 
#     # rlo_bem_moments, 
#     # rli_bem_forces, 
#     # rli_bem_moments,
#     # rri_bem_forces, 
#     # rri_bem_moments, 
#     # rro_bem_forces, 
#     # rro_bem_moments,  
#     # flo_bem_forces, 
#     # flo_bem_moments, 
#     # fli_bem_forces, 
#     # fli_bem_moments,
#     # fri_bem_forces, 
#     # fri_bem_moments, 
#     # fro_bem_forces, 
#     # fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_8,
# )
# system_m3l_model.register_output(total_forces, qst_8)
# system_m3l_model.register_output(total_moments, qst_8)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_8_ac_states,
#     design_condition=qst_8,
# )
# system_m3l_model.register_output(trim_residual, qst_8)
# # endregion

# region qst 9
qst_9 = cd.CruiseCondition(name='qst_9')
qst_9.atmosphere_model = cd.SimpleAtmosphereModel()
qst_9.set_module_input('pitch_angle', val=0.03981101, dv_flag=True, lower=0.03981101-np.deg2rad(4))
qst_9.set_module_input('mach_number', val=0.16937793)
qst_9.set_module_input('altitude', val=300)
qst_9.set_module_input(name='range', val=20)
qst_9.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

qst_9_ac_states = qst_9.evaluate_ac_states()

vlm_model = VASTFluidSover(
    surface_names=[
        f"{wing_vlm_mesh_name}_qst_9",
        f"{htail_vlm_mesh_name}_qst_9",
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
        (1, ) + htail_camber_surface.evaluate().shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    cl0=[0.25, 0]
)

forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)
system_m3l_model.register_output(vlm_forces, design_condition=qst_9)
system_m3l_model.register_output(vlm_moments, design_condition=qst_9)

pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=500, upper=2000, scaler=1e-3)
pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# rlo_bem_model = BEM(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=rlo_bem_mesh)
# rlo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# rli_bem_model = BEM(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=rli_bem_mesh)
# rli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# rri_bem_model = BEM(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=rri_bem_mesh)
# rri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# rro_bem_model = BEM(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=rro_bem_mesh)
# rro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# flo_bem_model = BEM(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=flo_bem_mesh)
# flo_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# fli_bem_model = BEM(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=fli_bem_mesh)
# fli_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# fri_bem_model = BEM(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=fri_bem_mesh)
# fri_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

# fro_bem_model = BEM(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=fro_bem_mesh)
# fro_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=4000, scaler=1e-3)
# fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_9_ac_states, design_condition=qst_9)

inertial_loads_model = cd.InertialLoads()
inertial_forces, inertial_moments = inertial_loads_model.evaluate(
    total_cg_vector=total_cg, 
    totoal_mass=total_mass, 
    ac_states=qst_9_ac_states, 
    design_condition=qst_9
)
system_m3l_model.register_output(inertial_forces, qst_9)
system_m3l_model.register_output(inertial_moments, qst_9)

total_forces_moments_model = cd.TotalForcesMoments()
total_forces, total_moments = total_forces_moments_model.evaluate(
    # rlo_bem_forces, 
    # rlo_bem_moments, 
    # rli_bem_forces, 
    # rli_bem_moments,
    # rri_bem_forces, 
    # rri_bem_moments, 
    # rro_bem_forces, 
    # rro_bem_moments,  
    # flo_bem_forces, 
    # flo_bem_moments, 
    # fli_bem_forces, 
    # fli_bem_moments,
    # fri_bem_forces, 
    # fri_bem_moments, 
    # fro_bem_forces, 
    # fro_bem_moments,  
    inertial_forces, 
    inertial_moments,
    pp_bem_forces,
    pp_bem_moments,
    vlm_forces,
    vlm_moments,
    design_condition=qst_9,
)
system_m3l_model.register_output(total_forces, qst_9)
system_m3l_model.register_output(total_moments, qst_9)

eom_m3l_model = cd.EoMEuler6DOF()
trim_residual = eom_m3l_model.evaluate(
    total_mass=total_mass, 
    total_cg_vector=total_cg, 
    total_inertia_tensor=total_inertia, 
    total_forces=total_forces, 
    total_moments=total_moments,
    ac_states=qst_9_ac_states,
    design_condition=qst_9,
)
system_m3l_model.register_output(trim_residual, qst_9)
# endregion

# # region qst 10
# qst_10 = cd.CruiseCondition(name='qst_10')
# qst_10.atmosphere_model = cd.SimpleAtmosphereModel()
# qst_10.set_module_input('pitch_angle', val=0.03369678, dv_flag=True, lower=np.deg2rad(-5), upper=np.deg2rad(5))
# qst_10.set_module_input('mach_number', val=0.17354959)
# qst_10.set_module_input('altitude', val=300)
# qst_10.set_module_input(name='range', val=20)
# qst_10.set_module_input(name='observer_location', val=np.array([0, 0, 0]))

# qst_10_ac_states = qst_10.evaluate_ac_states()

# vlm_model = VASTFluidSover(
#     surface_names=[
#         f"{wing_vlm_mesh_name}_qst_10",
#         f"{htail_vlm_mesh_name}_qst_10",
#     ],
#     surface_shapes=[
#         (1, ) + wing_camber_surface.evaluate().shape[1:],
#         (1, ) + htail_camber_surface.evaluate().shape[1:],
#     ],
#     fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
#     mesh_unit='ft',
#     cl0=[0.25, 0]
# )

# forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)
# system_m3l_model.register_output(vlm_forces, design_condition=qst_10)
# system_m3l_model.register_output(vlm_moments, design_condition=qst_10)

# pp_bem_model = BEM(disk_prefix='pp_disk', blade_prefix='pp', component=pp_disk, mesh=pusher_bem_mesh)
# pp_bem_model.set_module_input('rpm', val=1350, dv_flag=True, lower=800, upper=2000, scaler=1e-3)
# pp_bem_forces, pp_bem_moments, _, _, _ = pp_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # rlo_bem_model = PittPeters(disk_prefix='rlo_disk', blade_prefix='rlo', component=rlo_disk, mesh=pitt_peters_mesh_lift)
# # rlo_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # rlo_bem_forces, rlo_bem_moments,_ ,_ ,_ = rlo_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # rli_bem_model = PittPeters(disk_prefix='rli_disk', blade_prefix='rli', component=rli_disk, mesh=pitt_peters_mesh_lift)
# # rli_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # rli_bem_forces, rli_bem_moments,_ ,_ ,_ = rli_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # rri_bem_model = PittPeters(disk_prefix='rri_disk', blade_prefix='rri', component=rri_disk, mesh=pitt_peters_mesh_lift)
# # rri_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # rri_bem_forces, rri_bem_moments,_ ,_ ,_ = rri_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # rro_bem_model = PittPeters(disk_prefix='rro_disk', blade_prefix='rro', component=rro_disk, mesh=pitt_peters_mesh_lift)
# # rro_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # rro_bem_forces, rro_bem_moments,_ ,_ ,_ = rro_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # flo_bem_model = PittPeters(disk_prefix='flo_disk', blade_prefix='flo', component=flo_disk, mesh=pitt_peters_mesh_lift)
# # flo_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # flo_bem_forces, flo_bem_moments,_ ,_ ,_ = flo_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # fli_bem_model = PittPeters(disk_prefix='fli_disk', blade_prefix='fli', component=fli_disk, mesh=pitt_peters_mesh_lift)
# # fli_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # fli_bem_forces, fli_bem_moments,_ ,_ ,_ = fli_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # fri_bem_model = PittPeters(disk_prefix='fri_disk', blade_prefix='fri', component=fri_disk, mesh=pitt_peters_mesh_lift)
# # fri_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # fri_bem_forces, fri_bem_moments,_ ,_ ,_ = fri_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# # fro_bem_model = PittPeters(disk_prefix='fro_disk', blade_prefix='fro', component=fro_disk, mesh=pitt_peters_mesh_lift)
# # fro_bem_model.set_module_input('rpm', val=200, dv_flag=True, lower=0.5, upper=4000, scaler=4e-3)
# # fro_bem_forces, fro_bem_moments,_ ,_ ,_ = fro_bem_model.evaluate(ac_states=qst_10_ac_states, design_condition=qst_10)

# inertial_loads_model = cd.InertialLoads()
# inertial_forces, inertial_moments = inertial_loads_model.evaluate(
#     total_cg_vector=total_cg, 
#     totoal_mass=total_mass, 
#     ac_states=qst_10_ac_states, 
#     design_condition=qst_10
# )
# system_m3l_model.register_output(inertial_forces, qst_10)
# system_m3l_model.register_output(inertial_moments, qst_10)

# total_forces_moments_model = cd.TotalForcesMoments()
# total_forces, total_moments = total_forces_moments_model.evaluate(
#     # rlo_bem_forces, 
#     # rlo_bem_moments, 
#     # rli_bem_forces, 
#     # rli_bem_moments,
#     # rri_bem_forces, 
#     # rri_bem_moments, 
#     # rro_bem_forces, 
#     # rro_bem_moments,  
#     # flo_bem_forces, 
#     # flo_bem_moments, 
#     # fli_bem_forces, 
#     # fli_bem_moments,
#     # fri_bem_forces, 
#     # fri_bem_moments, 
#     # fro_bem_forces, 
#     # fro_bem_moments,  
#     inertial_forces, 
#     inertial_moments,
#     pp_bem_forces,
#     pp_bem_moments,
#     vlm_forces,
#     vlm_moments,
#     design_condition=qst_10,
# )
# system_m3l_model.register_output(total_forces, qst_10)
# system_m3l_model.register_output(total_moments, qst_10)

# eom_m3l_model = cd.EoMEuler6DOF()
# trim_residual = eom_m3l_model.evaluate(
#     total_mass=total_mass, 
#     total_cg_vector=total_cg, 
#     total_inertia_tensor=total_inertia, 
#     total_forces=total_forces, 
#     total_moments=total_moments,
#     ac_states=qst_10_ac_states,
#     design_condition=qst_10,
# )
# system_m3l_model.register_output(trim_residual, qst_10)
# # endregion

# design_scenario.add_design_condition(qst_1)
# design_scenario.add_design_condition(qst_2)
# design_scenario.add_design_condition(qst_3)
# design_scenario.add_design_condition(qst_4)
# design_scenario.add_design_condition(qst_5)
# design_scenario.add_design_condition(qst_6)
# design_scenario.add_design_condition(qst_7)
# design_scenario.add_design_condition(qst_8)
design_scenario.add_design_condition(qst_9)
# design_scenario.add_design_condition(qst_10)

system_model.add_m3l_model('system_m3l_model', system_m3l_model)

caddee_csdl_model = caddee.assemble_csdl()

# caddee_csdl_model.create_input('qst_1_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_1_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_1_wing_actuation', val=np.deg2rad(4))
tilt_1 = np.deg2rad(0)
tilt_2 = np.deg2rad(0)

# caddee_csdl_model.create_input('qst_1_rlo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_rlo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_rlo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_rlo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_rli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_rli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_rli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_rli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_rri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_rri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_rri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_rri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_rro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_rro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_rro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_rro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_flo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_flo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_flo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_flo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_fli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_fli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_fli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_fli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_fri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_fri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_fri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_fri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_1_fro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_1_fro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_1_fro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_1_fro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))



# caddee_csdl_model.create_input('qst_2_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_2_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_2_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_2_rlo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_rlo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_rlo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_rlo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_rli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_rli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_rli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_rli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_rri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_rri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_rri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_rri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_rro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_rro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_rro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_rro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_flo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_flo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_flo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_flo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_fli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_fli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_fli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_fli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_fri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_fri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_fri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_fri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_2_fro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_2_fro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_2_fro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_2_fro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))


# caddee_csdl_model.create_input('qst_3_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_3_tail_actuation', lower=np.deg2rad(-30), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_3_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_3_rlo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_rlo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_rlo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_rlo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_rli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_rli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_rli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_rli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_rri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_rri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_rri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_rri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_rro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_rro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_rro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_rro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_flo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_flo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_flo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_flo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_fli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_fli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_fli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_fli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_fri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_fri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_fri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_fri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_3_fro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_3_fro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_3_fro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_3_fro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))


# caddee_csdl_model.create_input('qst_4_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_4_tail_actuation', lower=np.deg2rad(-25), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_4_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_4_rlo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_rlo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_rlo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_rlo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_rli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_rli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_rli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_rli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_rri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_rri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_rri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_rri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_rro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_rro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_rro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_rro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_flo_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_flo_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_flo_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_flo_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_fli_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_fli_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_fli_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_fli_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_fri_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_fri_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_fri_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_fri_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))

# caddee_csdl_model.create_input('qst_4_fro_disk_actuation_1', val=tilt_1)
# caddee_csdl_model.create_input('qst_4_fro_disk_actuation_2', val=tilt_2)
# caddee_csdl_model.add_design_variable('qst_4_fro_disk_actuation_1', lower=np.deg2rad(-10), upper=np.deg2rad(10))
# caddee_csdl_model.add_design_variable('qst_4_fro_disk_actuation_2', lower=np.deg2rad(-10), upper=np.deg2rad(10))


# caddee_csdl_model.create_input('qst_5_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_5_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_5_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_6_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_6_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_6_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_7_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_7_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_7_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_8_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_8_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_8_wing_actuation', val=np.deg2rad(3.2))

caddee_csdl_model.create_input('qst_9_tail_actuation', val=np.deg2rad(-0.5))
caddee_csdl_model.add_design_variable('qst_9_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
caddee_csdl_model.create_input('qst_9_wing_actuation', val=np.deg2rad(3.2))

# caddee_csdl_model.create_input('qst_10_tail_actuation', val=np.deg2rad(-0.5))
# caddee_csdl_model.add_design_variable('qst_10_tail_actuation', lower=np.deg2rad(-15), upper=np.deg2rad(15))
# caddee_csdl_model.create_input('qst_10_wing_actuation', val=np.deg2rad(0))


# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_1_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_2_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_3_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_4_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_5_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_6_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_7_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_8_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_9_euler_eom_gen_ref_pt.trim_residual', equals=0)
# caddee_csdl_model.add_constraint('system_model.system_m3l_model.qst_10_euler_eom_gen_ref_pt.trim_residual', equals=0)


# trim_1 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_1_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_2 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_2_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_3 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_3_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_4 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_4_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_5 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_5_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_6 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_6_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_7 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_7_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_8 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_8_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
trim_9 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_9_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# trim_10 = caddee_csdl_model.declare_variable('system_model.system_m3l_model.qst_10_euler_eom_gen_ref_pt.trim_residual', shape=(1, ))
# caddee_csdl_model.add_objective('system_model.system_m3l_model.total_constant_mass_properties.total_mass', scaler=1e-3)

combined_trim = caddee_csdl_model.register_output('combined_trim', trim_9 * 1)
# combined_trim = caddee_csdl_model.register_output('combined_trim', trim_1 *1 + trim_2*1 + trim_3*1 + trim_4*1 + trim_5 * 1 + trim_6 * 1 + trim_7 * 1 + trim_8 * 1 + trim_9 * 1 + trim_10*1)
caddee_csdl_model.add_objective('combined_trim')


sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()

# print(sim['system_model.system_m3l_model.qst_1_rlo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rro_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_flo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fro_disk_bem_model.thrust_vector'])


sim.check_totals(of='system_model.system_m3l_model.qst_9_euler_eom_gen_ref_pt.trim_residual', wrt='system_model.system_m3l_model.qst_9_pp_disk_bem_model.rpm', step=1e-5)


# prob = CSDLProblem(problem_name='lpc', simulator=sim)
# optimizer = SLSQP(prob, maxiter=1000, ftol=1E-7)
# optimizer.solve()
# optimizer.print_results()

prob = CSDLProblem(problem_name='qs_transition_trim', simulator=sim)

# prob2 = CSDLProblem(problem_name='fuck', simulator=sim)

# opt = SNOPT(
#     prob2,
#     Major_iterations=300,
#     Major_optimality=1e-5,
#     Major_feasibility=1e-5,
#     append2file=True
# )
optimizer = SNOPT(
    prob, 
    Major_iterations=500, 
    Major_optimality=1e-5, 
    Major_feasibility=1e-8,
    append2file=True,
)

optimizer.solve()
# opt.print_results()

# print(sim['system_model.system_m3l_model.qst_1_rlo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_rro_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_flo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_1_fro_disk_bem_model.thrust_vector'])

print('\n')

# print(sim['system_model.system_m3l_model.qst_2_rlo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_rli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_rri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_rro_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_flo_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_fli_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_fri_disk_bem_model.thrust_vector'])
# print(sim['system_model.system_m3l_model.qst_2_fro_disk_bem_model.thrust_vector'])

# prob = CSDLProblem(problem_name='lpc', simulator=sim)
# optimizer = SLSQP(prob, maxiter=1000, ftol=1E-5)
# optimizer.solve()
# optimizer.print_results()