import csdl
import numpy as np
from numpy.core.fromnumeric import size


class EfficiencyModel(csdl.Model):
    """
    Compute the mesh at each time step for the eel actuation model given the kinematic variables.
    parameters
    ----------
    tail_amplitude : csdl variable [1,]
    tail_frequency : csdl variable [1,]
    v_x : csdl variable [1,]
    Returns
    -------
    1. mesh[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    bound vortices points
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # self.parameters.declare('mesh_unit', default='m')
        # self.parameters.declare('n_period')


    def define(self):
        # surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        # mesh_unit = self.parameters['mesh_unit']
        # N_period = self.parameters['n_period']

        panel_forces_all = self.declare_variable('panel_forces_all',shape=(surface_shapes[0][0],int(surface_shapes[0][1]-1)*int(surface_shapes[0][2]-1),3))
        velocities = self.declare_variable('eel_kinematic_vel',shape=panel_forces_all.shape)
        v_x = self.declare_variable('v_x')
        thrust = self.declare_variable('thrust',shape=(num_nodes,1))
        panel_forces_all_x = panel_forces_all[:,:,0]
        velocities_x = self.create_output('velocities_x',shape=velocities.shape,val=0)
        velocities_x[:,:,0] = velocities[:,:,0] #- csdl.expand(v_x,shape=velocities[:,:,0].shape)
        velocities_x[:,:,1] = velocities[:,:,1]
        velocities_x[:,:,2] = velocities[:,:,2]


        panel_thrust_power = csdl.sum(csdl.dot(panel_forces_all,-velocities_x,axis=2))
        thrust_power = csdl.sum(csdl.sum(thrust,axes=(1,))*csdl.expand(v_x,shape=(num_nodes,)))

        self.print_var(panel_thrust_power)
        self.print_var(thrust_power)

        # efficiency = thrust_power/(panel_thrust_power+thrust_power)
        efficiency = thrust_power/(panel_thrust_power)
        self.print_var(efficiency)
        self.register_output('efficiency',efficiency)

# panel_forces_all = sim['panel_forces_all']
# velocities_panel = - sim['eel_kinematic_vel']
# panel_forces_dynamic = sim['panel_forces_dynamic']
# panel_acc = np.gradient(velocities_panel,axis=0)/h_stepsize

# mid_panel_vel = velocities_panel[:,int(velocities_panel.shape[1]/2),:]*0.5
# mid_panel_acc = panel_acc[:,int(panel_acc.shape[1]/2),:]*0.1
# mid_panel_forces = panel_forces_all[:,int(panel_forces_all.shape[1]/2),:]
# mid_panel_forces_dynamic = panel_forces_dynamic[:,int(panel_forces_dynamic.shape[1]/2),:]
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.plot(mid_panel_vel[:,1])
# plt.plot(mid_panel_acc[:,1])
# plt.plot(mid_panel_forces[:,1])
# plt.plot(mid_panel_forces_dynamic[:,1])
# plt.plot(mid_panel_forces[:,1]-mid_panel_forces_dynamic[:,1])
# plt.legend(['vel','acc','force','force_dynamic','force_static'])
# plt.show()


        # compute energy efficiency here

# thrust = sim['thrust']
# v_x = sim['v_x']
# panel_forces_all = sim['panel_forces_all']
# velocities = sim['eel_kinematic_vel']
# -np.sum(np.sum(thrust,axis=(1,))*v_x)        

# -np.sum(np.einsum('ijk,ijk->ij',panel_forces_all,velocities))
# velocities_x = velocities.copy()
# velocities_x[:,:,0] = 0
# -np.sum(np.einsum('ijk,ijk->ij',panel_forces_all,velocities_x))
        
# thrust = sim['thrust']
# v_x = -sim['v_x']

# thrust_power = np.sum(thrust*v_x)/t_vec[-1]

# panel_forces_all = sim['panel_forces_all']
# velocities = -sim['eval_total_vel']
# v_0 = velocities.copy()
# v_0[:,:,0]=0
# panel_power = np.sum(panel_forces_all * v_0)/t_vec[-1]
# # efficiency = sim['efficiency']



# if __name__ == "__main__":
#     import python_csdl_backend
#     import numpy as np
#     import pyvista as pv
#     # simulator_name = 'csdl_om'
#     num_nodes = 30
#     num_pts_chord = 13 # nx
#     num_pts_span = 3

#     model = csdl.Model()
#     model.add(EelActuationModel(surface_names=['surface'],surface_shapes=[(num_nodes,num_pts_chord,num_pts_span)]),'actuation')
    

#     sim = python_csdl_backend.Simulator(model)
#     sim.run()

#     surface = sim["surface"]
#     print(surface.shape)

#     for i in range(num_nodes):
#         x = surface[i,:,:,0]
#         y = surface[i,:,:,1]
#         z = surface[i,:,:,2]

#         grid = pv.StructuredGrid(x,y,z)
#         grid.save(filename=f'eel_actuation_{i}.vtk')
