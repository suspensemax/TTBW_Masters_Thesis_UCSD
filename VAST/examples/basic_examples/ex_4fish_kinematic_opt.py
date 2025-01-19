'''Example 4 : fish trim optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import resource

########################################
# define mesh resolution and num_nodes
########################################

profile_opt = False
compute_totals = True
before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t_start = time.time()

nx = 15; ny = 3
num_nodes = 30
nt = num_nodes

# kinematics variables
v_inf = 0.3
lambda_ = 1
N_period= 2         
st = 0.15
A = 0.125
f = 0.48
surface_properties_dict = {'eel':(nx,ny,3)}

u_val = (np.ones(num_nodes)).reshape((num_nodes,1)) * v_inf
w_vel = np.zeros((num_nodes, 1))

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': alpha_equ, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}
t_vec = np.linspace(0,N_period/f,num_nodes)
h_stepsize = t_vec[1]

#####################
# define the problem
#####################
import python_csdl_backend
sim = python_csdl_backend.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,n_period=N_period,
                                    surface_properties_dict=surface_properties_dict), mode='rev')
    
if profile_opt==True:
    sim.run()

    #####################
    # optimizaton
    #####################
    from modopt.csdl_library import CSDLProblem

    from modopt.scipy_library import SLSQP
    from modopt.snopt_library import SNOPT
    # Define problem for the optimization
    prob = CSDLProblem(
        problem_name='eel_kinematic_opt',
        simulator=sim,
    )
    # optimizer = SLSQP(prob, maxiter=1)
    optimizer = SNOPT(
        prob, 
        Major_iterations=1,
        Major_optimality=1e-6,
        Major_feasibility=1e-4,
        append2file=True,
        # Major_step_limit=.25,
    )

    optimizer.solve()
    optimizer.print_results(summary_table=True)
    print('total time is', time.time() - t_start)

    #####################
    # memory usage
    #####################
    after_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ALLOCATED_MEMORY = (after_mem - before_mem)/(1024**3) # for mac
    ALLOCATED_MEMORY = (after_mem - before_mem)*1e-6 # for linux

    print('Allocated memory: ', ALLOCATED_MEMORY, 'Gib')
elif profile_opt==False:

    sim.run()
    print('simulation time is', time.time() - t_start)
    aft_simulation = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ALLOCATED_MEMORY_simulation = (aft_simulation - before_mem)*1e-6 # for linux

    print('Allocated memory for simulations: ', ALLOCATED_MEMORY_simulation, 'Gib')

    if compute_totals==True:
        t_start = time.time()
        bf = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        sim.compute_total_derivatives()
        print('total time is', time.time() - t_start)
        aft_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ALLOCATED_MEMORY_total = (aft_total - bf)*1e-6
        print('Allocated memory for total derivatives: ', ALLOCATED_MEMORY_total, 'Gib')

# np.average(sim['thrust']/(0.5*1*sim['v_x']**2*0.13826040386294708))
# np.average(sim['thrust']/(0.5*1*sim['v_x']**2*0.13826040386294708))
