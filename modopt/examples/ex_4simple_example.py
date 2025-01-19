import numpy as np
from modopt.api import Problem


class X4(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'x^4'

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([.3, .3]))
        self.add_objective('f',)

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )
        self.declare_objective_hessian(of='x', wrt='x')

    # Compute the value of the objective, gradient and Hessian 
    # with the given design variable values
    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x']**3

    def compute_objective_hessian(self, dvs, hess):
        hess['x', 'x'] = 12 * np.diag(dvs['x']**2)


import numpy as np
import time
from modopt.api import Optimizer

class SteepestDescent(Optimizer):
    def initialize(self):

        # Name your algorithm
        self.solver_name = 'steepest_descent'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('opt_tol', types=float)

        # Specify format of outputs available from your optimizer after each iteration
        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
        }

        # Enable user to specify, as a list, which among the available outputs
        # need to be stored in memory and written to output files
        self.options.declare('outputs',
                             types=list,
                             default=['itr', 'obj', 'x', 'opt', 'time'])

    def solve(self):
        nx = self.problem.nx
        x = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        max_itr = self.options['max_itr']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Iteration counter
        itr = 0

        # Optimality
        opt = np.linalg.norm(g_k)

        # Initializing outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time)

        while (opt > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            p_k = -g_k

            x_k += p_k
            f_k = obj(x_k)
            g_k = grad(x_k)

            opt = np.linalg.norm(g_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Append arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time


# Set your optimality tolerance
opt_tol = 1E-8
# Set maximum optimizer iteration limit
max_itr = 100

prob = X4()

from modopt.optimization_algorithms import Newton, QuasiNewton, SQP

# Set up your optimizer with your problem and pass in optimizer parameters
optimizer = SteepestDescent(prob,
                            opt_tol=opt_tol,
                            max_itr=max_itr,
                            outputs=['itr', 'obj', 'x', 'opt', 'time'])
optimizer = Newton(prob, opt_tol=opt_tol)
optimizer = QuasiNewton(prob, opt_tol=opt_tol)

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x.get_data())

# Solve your optimization problem
optimizer.solve()

# Print results of optimization (summary_table contains information from each iteration)
optimizer.print_results(summary_table=True, compact_print=True)

# Print to see any output that was declared
# Since the arrays are long, here we only print the last entry and
# verify it with the print_results() above

print('\n')
print('Optimizer data')
print('num_iterations:', optimizer.outputs['itr'][-1])
print('optimized_dvs:', optimizer.outputs['x'][-1])
print('optimization_time:', optimizer.outputs['time'][-1])
print('optimized_obj:', optimizer.outputs['obj'][-1])
print('final_optimality:', optimizer.outputs['opt'][-1])

print('\n')
print('Final problem data')
print('optimized_dvs:', prob.x.get_data())
print('optimized_obj:', prob.obj['f'])