from VAST.core.vlm_llt.vlm_system import VLMSystem
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_outputs_group import Outputs
import numpy as np
import csdl
from VAST.core.vlm_llt.NodalMapping import NodalMap,RadialBasisFunctions

class VLMSolverModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)

        self.parameters.declare('AcStates', default=None)

        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0.25)
        self.parameters.declare('eval_pts_names', default=None)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=[0])

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        cl0 = self.parameters['cl0']

        free_stream_velocities = self.parameters['free_stream_velocities']

        eval_pts_option = self.parameters['eval_pts_option']

        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']
        mesh_unit = self.parameters['mesh_unit']
        if self.parameters['AcStates'] == None:
            frame_vel_val = -free_stream_velocities

            frame_vel = self.create_input('frame_vel', val=frame_vel_val)

        self.add(
            VLMSystem(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                AcStates=self.parameters['AcStates'],
                solve_option=self.parameters['solve_option'],
                TE_idx=self.parameters['TE_idx'],
                mesh_unit=mesh_unit,
                eval_pts_option=eval_pts_option,
                eval_pts_location=eval_pts_location,
            ), 'VLM_system')
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=cl0,
        )
        self.add(sub, name='VLM_outputs')


    def construct_displacement_map(self, nodal_outputs_mesh):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=oml_mesh)
        return displacement_map

    def construct_force_map(self, nodal_force):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_force.mesh.value.reshape((-1, 3))
        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=oml_mesh)
        return force_map

    def construct_invariant_matrix(self):
        pass

if __name__ == "__main__":

    import csdl
    import numpy as np
    from VAST.core.fluid_problem import FluidProblem
    from VAST.utils.generate_mesh import *
    from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
    from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
    from python_csdl_backend import Simulator

    num_nodes=1; nx=3; ny=11

    solver_option = 'VLM'
    problem_type = 'fixed_wake'
    fluid_problem = FluidProblem(solver_option=solver_option, problem_type=problem_type)
    ####################################################################
    # 1. Define VLM inputs that share the common names within CADDEE
    ####################################################################
    num_nodes = 1
    create_opt = 'create_inputs'
    model_1 = csdl.Model()
    alpha = np.deg2rad(np.ones((num_nodes,1))*5)

    vx = 248.136
    vz = 0

    u = model_1.create_input('u',val=np.ones((num_nodes,1))*vx)
    v = model_1.create_input('v',val=np.zeros((num_nodes, 1)))
    w = model_1.create_input('w',val=np.ones((num_nodes,1))*vz)
    p = model_1.create_input('p',val=np.zeros((num_nodes, 1)))
    q = model_1.create_input('q',val=np.zeros((num_nodes, 1)))
    r = model_1.create_input('r',val=np.zeros((num_nodes, 1)))
    phi = model_1.create_input('phi',val=np.zeros((num_nodes, 1)))
    theta = model_1.create_input('theta',val=alpha)
    psi = model_1.create_input('psi',val=np.zeros((num_nodes, 1)))
    x = model_1.create_input('x',val=np.zeros((num_nodes, 1)))
    y = model_1.create_input('y',val=np.zeros((num_nodes, 1)))
    z = model_1.create_input('z',val=np.ones((num_nodes, 1))*1000)
    phiw = model_1.create_input('phiw',val=np.zeros((num_nodes, 1)))
    gamma = model_1.create_input('gamma',val=np.zeros((num_nodes, 1)))
    psiw = model_1.create_input('psiw',val=np.zeros((num_nodes, 1)))

    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface
    nx = 3  # number of points in streamwise direction
    ny = 11  # number of points in spanwise direction

    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]

    # chord = 1.49352
    # span = 16.2 / chord

    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": 10.0,
        "chord": 1,
        "span_cos_sppacing": 1.0,
        "chord_cos_sacing": 1.0,
    }
    # Generate OML MESH for a rectangular wing with NACA0012 airfoil
    airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
    z_val = np.linspace(-5, 5, 13)
    oml_mesh = np.zeros((21, 13, 3))
    oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
    oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
    oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))
    
    

    # add oml mesh as a csdl variable
    oml = model_1.create_input('oml', val=oml_mesh.reshape(1, 21, 13, 3))

    # create a random displacement on the oml mesh
    disp_temp = np.linspace(0.6, 0,7)
    disp_z = np.outer(np.ones(21),np.concatenate((disp_temp, disp_temp[:-1][::-1])))
    disp = np.zeros((1, 21, 13, 3))
    disp[0, :, :, 2] = disp_z
    oml_disp = model_1.create_input('oml_displacement', val=disp)



    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) #(nx,ny,3)

    ####################################################################
    # project the displacement on the oml mesh to the camber surface mesh
    G_mat = NodalMap(oml_mesh.reshape(-1,3), mesh.reshape(-1,3), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    map_in = model_1.create_input('map_in', val=G_mat.map)
    ####################################################################

    mapped_disp = np.einsum('ij,jk->ik',G_mat.map,disp.reshape(-1,3)).reshape(mesh.shape)
    deformed_mesh = mesh + mapped_disp
    
    import pyvista as pv
    plotter = pv.Plotter()
    grid = pv.StructuredGrid(oml_mesh[:, :, 0]+disp[0,:, :, 0], oml_mesh[:, :, 1]+disp[0,:, :, 1], oml_mesh[:, :, 2]+disp[0,:, :, 2])
    grid_1 = pv.StructuredGrid(deformed_mesh[:, :, 0], deformed_mesh[:, :, 1], deformed_mesh[:, :, 2])

    plotter.add_mesh(grid, show_edges=False,opacity=0.5, color='red')
    plotter.add_mesh(grid_1, show_edges=True,color='grey')
    plotter.set_background('white')
    plotter.show()
    offset = 0

    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    wing = model_1.create_input('wing_undef', val=mesh_val)
    oml_disp_reshaped = csdl.reshape(oml_disp, (21* 13, 3))
    wing_def = wing + csdl.reshape(csdl.einsum(map_in, oml_disp_reshaped,subscripts='ij,jk->ik'), (num_nodes, nx, ny, 3))
    model_1.register_output('wing', wing_def)
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
    wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))

    model_1.add(submodel, 'VLMSolverModel')
    sim = Simulator(model_1)
    sim.run()


    # project the force on the oml mesh to the camber surface mesh
    G_mat_out = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3),oml_mesh.reshape(-1,3),  RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    map_out = model_1.create_input('map_in', val=G_mat_out.map)

    mapped_force = np.einsum('ij,jk->ik',G_mat_out.map,sim['panel_forces'].reshape(-1,3)).reshape(oml_mesh.shape)

    # formulate the invariant matrix to map the force on the quarter-chord of each panel (num_nodes, nx-1, ny-1, 3) to the vlm deformed mesh (num_nodes, nx, ny, 3)
    N_A = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3), 
                    mesh.reshape(-1,3), 
                    RBF_width_par=2,
                    RBF_func=RadialBasisFunctions.Gaussian).map

