'''Example 2 : optimization of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel as CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
import resource

from VAST.core.submodels.geometric_submodels.mesh_parameterizartion_model import MeshParameterizationComp
from modopt.csdl_library import CSDLProblem


from python_csdl_backend import Simulator

before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def ex1_generate_model_vlm_fixed_wake(num_nodes,nx, ny):
    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')
    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]

    ####################################################################
    # 1. add aircraft states
    ####################################################################
    v_inf = np.ones((num_nodes,1))*248.136
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles
    model_1 = csdl.Model()

    submodel = CreateACSatesModule(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    model_1.add(submodel, 'InputsModule')

    # theta = model_1.create_input('theta',val=theta)

    chord = 1.49352
    span = 16.2 / chord
    chord_csdl = model_1.create_input('wing_chord_l', val=chord)
    span_csdl = model_1.create_input('wing_span_l', val=span)
    taper_ratio = model_1.create_input("taper_ratio",val=0.5)
    area = (1+taper_ratio)*chord_csdl*span_csdl/2
    model_1.register_output('wing_area',area)

    submodel = MeshParameterizationComp(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        )
    model_1.add(submodel, 'MeshParameterizationModel')


    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface 
    # (nx: number of points in streamwise direction; ny:number of points in spanwise direction)

    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }
    # Generate mesh of a rectangular wing
    # mesh = generate_mesh(meput('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    ####################################################################
    # 3. add VAST solver
    ####################################################################
    if fluid_problem.solver_option == 'VLM':
        eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
        submodel = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes,
            eval_pts_shapes=eval_pts_shapes,
            AcStates='dummy',
        )
    # wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    # wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))
    model_1.add(submodel, 'VLMSolverModel')
    ####################################################################

    model_1.add_design_variable("wing_chord_l", lower=2, upper=10.0)
    model_1.add_design_variable("wing_span_l", lower=3, upper=20.0)
    # model_1.add_design_variable("theta", lower=np.deg2rad(-10), upper=np.deg2rad(10.0))
    model_1.add_design_variable("taper_ratio", lower=0.3, upper=1.0)

    model_1.add_constraint("wing_area",equals=20)
    model_1.add_constraint("wing_C_L",lower=0.5)

    model_1.add_objective("wing_C_D_i")

    sim = Simulator(model_1) # add simulator
    return sim

sim = ex1_generate_model_vlm_fixed_wake(num_nodes=1,nx=3, ny=11)
sim.run()

from modopt.snopt_library import SNOPT
from modopt.scipy_library import SLSQP
# Define problem for the optimization
prob = CSDLProblem(
    problem_name='wing_shape_opt',
    simulator=sim,
)
# optimizer = SLSQP(prob, maxiter=20)

# before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# optimizer = SLSQP(prob, maxiter=1)
optimizer = SNOPT(
    prob, 
    Major_iterations=100,
    # Major_optimality=1e-6,
    Major_optimality=1e-9,
    Major_feasibility=1e-9,
    append2file=True,
    Major_step_limit=.25,
)


optimizer.solve()

# after_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# ALLOCATED_MEMORY = (after_mem - before_mem)/(1024**3)
# print('Allocated memory: ', ALLOCATED_MEMORY, 'Gib')
# # Print results of optimization
# optimizer.print_results()


# sim.compute_totals(of='wing_C_D_i', wrt=['theta','taper_ratio','wing_span_l','wing_chord_l'], return_format='dict')
# after_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# ALLOCATED_MEMORY = (after_mem - before_mem)/(1024**3)
# print('Allocated memory: ', ALLOCATED_MEMORY, 'Gib')