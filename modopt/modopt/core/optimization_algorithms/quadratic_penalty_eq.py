import numpy as np
import time

from modopt.api import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo
from modopt.merit_functions import AugmentedLagrangianEq, L2Eq
# from modopt.approximate_hessians import BFGS
from modopt.approximate_hessians import BFGSM1 as BFGS
from modopt.core.approximate_hessians.bfgs_function import bfgs_update


class L2PenaltyEq(Optimizer):
    def initialize(self):
        self.solver_name = 'penalty_l2_eq'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.con = self.problem._compute_constraints
        self.jac = self.problem._compute_constraint_jacobian

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('opt_tol', types=float)
        self.options.declare('feas_tol', types=float)

        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            'con': (float, (self.problem.nc, )),
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'feas': float,
            'time': float,
            'num_f_evals': int,
            'num_g_evals': int,
            'step': float,
            'merit': float,
        }
        self.options.declare('outputs',
                             types=list,
                             default=[
                                 'itr', 'obj', 'x', 'opt', 'feas',
                                 'con', 'time', 'num_f_evals',
                                 'num_g_evals', 'step', 'merit'
                             ])

    def setup(self):
        nx = self.problem.nx
        nc = self.problem.nc
        self.OF = L2Eq(nx=nx,
                       nc=nc,
                       f=self.obj,
                       c=self.con,
                       g=self.grad,
                       j=self.jac)
        self.LS = BacktrackingArmijo(f=self.OF.compute_function,
                                     g=self.OF.compute_gradient)
        # self.LS = ScipyLS(f=self.OF.compute_function,
        #                   g=self.OF.compute_gradient)
        self.QN = BFGS(nx=self.problem.nx)

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        nc = self.problem.nc
        x0 = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        # feas_tol = self.options['feas_tol']
        max_itr = self.options['max_itr']
        rho = 1000000.

        obj = self.obj
        grad = self.grad
        con = self.con
        jac = self.jac

        LS = self.LS
        QN = self.QN
        OF = self.OF

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)
        c_k = con(x_k)
        J_k = jac(x_k)

        OF.set_rho(rho)
        of_k = OF.evaluate_function(x_k, f_k, c_k)
        ofg_k = OF.evaluate_gradient(x_k, f_k, c_k, g_k, J_k)

        # Iteration counter
        itr = 0

        opt = np.linalg.norm(ofg_k)
        feas = np.linalg.norm(c_k)
        num_f_evals = 1
        num_g_evals = 1

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            con=c_k,
                            opt=opt,
                            feas=feas,
                            time=time.time() - start_time,
                            num_f_evals=num_f_evals,
                            num_g_evals=num_g_evals,
                            step=0.,
                            merit=of_k)

        # B_k = np.identity(nx)
        while (opt > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # Hessian approximation
            B_k = QN.B_k
            # if itr % 2 == 0:
            #     B_k = np.diag(np.diag(B_k))

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            p_k = np.linalg.solve(B_k, -ofg_k)
            print((x_k + p_k))

            # Compute the step length along the search direction via a line search
            # alpha, of_k, ofg_new, of_slope_new, new_f_evals, new_g_evals, converged = LS.search(
            #     x=x_k, p=p_k, f0=of_k, g0=ofg_k)
            alpha, of_k, new_f_evals, new_g_evals, converged = LS.search(
                x=x_k, p=p_k, f0=of_k, g0=ofg_k)

            num_f_evals += new_f_evals
            num_g_evals += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                alpha = None
                d_k = p_k * 1.

                x_k += d_k

                f_k = obj(x_k)
                g_k = grad(x_k)
                c_k = con(x_k)
                J_k = jac(x_k)

                of_k = OF.evaluate_function(x_k, f_k, c_k)
                ofg_new = OF.evaluate_gradient(x_k, f_k, c_k, g_k, J_k)
                w_k = (ofg_new - ofg_k)
                ofg_k = ofg_new

            else:
                d_k = alpha * p_k
                x_k += d_k

                f_k = obj(x_k)
                g_k = grad(x_k)
                c_k = con(x_k)
                J_k = jac(x_k)

                # if ofg_new == 'Unavailable':
                ofg_new = OF.evaluate_gradient(x_k, f_k, c_k, g_k, J_k)
                w_k = (ofg_new - ofg_k)
                ofg_k = ofg_new

            opt = np.linalg.norm(ofg_k)
            feas = np.linalg.norm(c_k)
            # print(pi_k)

            # Update the Hessian approximation
            QN.update(d_k, w_k)

            # B_k = self.problem.compute_lagrangian_hessian(x_k, pi_k)
            # B_k = bfgs_update(B_k, d_k[:nx], w_k[:nx])

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                con=c_k,
                                opt=opt,
                                feas=feas,
                                time=time.time() - start_time,
                                num_f_evals=num_f_evals,
                                num_g_evals=num_g_evals,
                                step=alpha,
                                merit=of_k)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time