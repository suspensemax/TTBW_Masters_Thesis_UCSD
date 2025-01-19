mesh = sim['eel_bd_vtx_coords']
thrust = sim['thrust']
v_x = sim['v_x']
leading_edge_x = mesh[:,:,0,0]

panel_forces_all = sim['panel_forces_all']

normals = sim['eel_bd_vtx_normals']

vel_air = sim['eval_total_vel']

eel_vel = -sim['eel_kinematic_vel']
vel_air[:,:,0] = 0
eel_vel[:,:,0] = 0


fij_dot_nij = np.einsum('ijk,ijk->ij',panel_forces_all,normals.reshape(num_nodes,-1,3))
vij_dot_nij = np.einsum('ijk,ijk->ij',vel_air,normals.reshape(num_nodes,-1,3))
vij_dot_nij = np.einsum('ijk,ijk->ij',eel_vel,normals.reshape(num_nodes,-1,3))
P_in = np.sum(fij_dot_nij*vij_dot_nij,axis=1).flatten()

p_out = (thrust*v_x).flatten()
np.average(p_out/(-P_in+p_out))
panel_forces_all

v_norm = np.linalg.norm(vel_air,axis=2)
s_panel = sim['eel_s_panel'].reshape(num_nodes,-1)
area_total = np.sum(s_panel,axis=1)

relative_area = s_panel/np.outer(area_total,np.ones(s_panel.shape[1]))

v_scaled = np.sum(v_norm**2*relative_area,axis=1)

ratio = np.average(v_x**2/v_scaled)