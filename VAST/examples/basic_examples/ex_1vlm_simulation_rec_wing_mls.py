'''Example 1 : simulation of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator

def ex1_generate_model_vlm_fixed_wake(num_nodes,nx, ny):
    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    model_1 = csdl.Model()
    ####################################################################
    # 1. add aircraft states
    ####################################################################
    v_inf = np.ones((num_nodes,1))*248.136
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

    submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    model_1.add(submodel, 'InputsModule')
    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface 
    # (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
    surface_names = ['wing','wing_1']
    surface_shapes = [(num_nodes, nx, ny, 3),(num_nodes, nx, ny-2, 3)]
    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }
    # Generate mesh of a rectangular wing
    mesh = np.array([[[[ 1.23431866e+01,  2.52445156e+01,  7.61899803e+00],
         [ 1.02636671e+01,  2.01968038e+01,  7.84026851e+00],
         [ 9.85856812e+00,  1.51500005e+01,  8.05305898e+00],
         [ 9.54411219e+00,  1.01000002e+01,  8.25003528e+00],
         [ 9.23152998e+00,  5.04999985e+00,  8.44626689e+00],
         [ 8.91777899e+00,  0.00000000e+00,  8.63965299e+00],
         [ 9.23109567e+00, -5.04999983e+00,  8.44627909e+00],
         [ 9.52079852e+00, -1.00966690e+01,  8.24109855e+00],
         [ 9.85807481e+00, -1.51500005e+01,  8.05319336e+00],
         [ 1.02636218e+01, -2.01968059e+01,  7.84025587e+00],
         [ 1.23431833e+01, -2.52445097e+01,  7.62035439e+00]],
        [[ 1.26074255e+01,  2.52496092e+01,  7.63634716e+00],
         [ 1.11314606e+01,  2.01967165e+01,  7.88454826e+00],
         [ 1.08748464e+01,  1.51491142e+01,  8.08919497e+00],
         [ 1.06737815e+01,  1.00991184e+01,  8.28657663e+00],
         [ 1.04730231e+01,  5.04914472e+00,  8.48320551e+00],
         [ 1.02704958e+01,  1.15338158e-08,  8.68014120e+00],
         [ 1.04726972e+01, -5.04914472e+00,  8.48382958e+00],
         [ 1.06562905e+01, -1.00966204e+01,  8.28744874e+00],
         [ 1.08744763e+01, -1.51491142e+01,  8.08967195e+00],
         [ 1.11314436e+01, -2.01967163e+01,  7.88514837e+00],
         [ 1.26074210e+01, -2.52402361e+01,  7.64125422e+00]],
        [[ 1.28702360e+01,  2.52495714e+01,  7.64012685e+00],
         [ 1.19992897e+01,  2.01966255e+01,  7.88056201e+00],
         [ 1.18911309e+01,  1.51482274e+01,  8.07768242e+00],
         [ 1.18034571e+01,  1.00982362e+01,  8.26962410e+00],
         [ 1.17145226e+01,  5.04828917e+00,  8.46108873e+00],
         [ 1.16243793e+01, -1.88206491e-08,  8.64616896e+00],
         [ 1.17143053e+01, -5.04828917e+00,  8.46082343e+00],
         [ 1.17917964e+01, -1.00965709e+01,  8.26968325e+00],
         [ 1.18908842e+01, -1.51482274e+01,  8.07747316e+00],
         [ 1.19992784e+01, -2.01966254e+01,  7.88033843e+00],
         [ 1.28699629e+01, -2.52495714e+01,  7.63996310e+00]],
        [[ 1.31352796e+01,  2.52495714e+01,  7.63953430e+00],
         [ 1.28671185e+01,  2.01965345e+01,  7.86226331e+00],
         [ 1.29074151e+01,  1.51473407e+01,  8.04937062e+00],
         [ 1.29331324e+01,  1.00973540e+01,  8.23384285e+00],
         [ 1.29560217e+01,  5.04743361e+00,  8.41815255e+00],
         [ 1.29746097e+01,  0.00000000e+00,  8.60336981e+00],
         [ 1.29559131e+01, -5.04743360e+00,  8.41863501e+00],
         [ 1.29273021e+01, -1.00965214e+01,  8.23462539e+00],
         [ 1.29072918e+01, -1.51473406e+01,  8.04977137e+00],
         [ 1.28671129e+01, -2.01965345e+01,  7.86258651e+00],
         [ 1.31352779e+01, -2.52495714e+01,  7.63961839e+00]],
        [[ 1.34001688e+01,  2.52495588e+01,  7.61381501e+00],
         [ 1.37342952e+01,  2.01964192e+01,  7.76010941e+00],
         [ 1.39228347e+01,  1.51464215e+01,  7.92224549e+00],
         [ 1.40617839e+01,  1.00964445e+01,  8.08795828e+00],
         [ 1.41963437e+01,  5.04654663e+00,  8.25398512e+00],
         [ 1.43306176e+01, -3.70259180e-17,  8.41987809e+00],
         [ 1.41963398e+01, -5.04654653e+00,  8.25391354e+00],
         [ 1.40617806e+01, -1.00964444e+01,  8.08789313e+00],
         [ 1.39228319e+01, -1.51464215e+01,  7.92218686e+00],
         [ 1.37342932e+01, -2.01964191e+01,  7.76005960e+00],
         [ 1.34002051e+01, -2.52495644e+01,  7.61380566e+00]]]]).reshape(5,11,3)
    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    mesh_1 = mesh.copy()[:,:-2,:]
    
    mesh_1[:,:,0] = mesh[:,:-2,:][:,:,0]*np.arange(1,10)
    print(mesh_1.shape)
    wing = model_1.create_input('wing_1', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_1-100))

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
            cl0 = [0.1,0.2],
        )
    # wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    # wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))
    model_1.add(submodel, 'VLMSolverModel')
    ####################################################################
    
    sim = Simulator(model_1) # add simulator
    return sim

sim = ex1_generate_model_vlm_fixed_wake(num_nodes=1,nx=5, ny=11)
sim.run()

# print('The number of nan in num_00e9 is: ', np.count_nonzero(np.isinf(sim['num_00e9'])))
# print('The number of nan in num_00eB is: ', np.count_nonzero(np.isinf(sim['num_00eB'])))
# print('The number of nan in num_00f2 is: ', np.count_nonzero(np.isinf(sim['num_00f2'])))
# print('The number of nan in num_00fu is: ', np.count_nonzero(np.isinf(sim['num_00fu'])))




# import pyvista as pv

# plotter = pv.Plotter()
# grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])

# plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
# plotter.set_background('white')
# plotter.show()
