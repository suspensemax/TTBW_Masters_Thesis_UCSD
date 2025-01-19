from lsdo_modules.module.module_maker import ModuleMaker
from lsdo_modules.module.module import Module
from python_csdl_backend import Simulator
import csdl
from csdl import NewtonSolver, ScipyKrylov
import numpy as np
from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem


class TestModule1CSDL(ModuleMaker):
    def initialize_module(self):
        self.parameters.declare('scaling_factor')

    def define_module(self):
        s = self.parameters['scaling_factor']
        a = self.register_module_input('a')
        b = self.register_module_input('b')

        c =  (s * a)**2 + 2 * csdl.cos(b)
        self.register_module_output('c', c)

        x = self.register_module_output('x', shape=(2, ))
        x[0] = 2 * a
        x[1] = 1 + b
        
        some_mod = ModuleMaker()
        var = some_mod.register_module_input('x', shape=(2, ))
        some_mod.register_module_output('test_var', csdl.sin(var))

        self.add_module(some_mod, 'test_mod')

        print('declare_vars', self.module_inputs)


class TestModule2CSDL(ModuleMaker):
    def initialize_module(self): pass

    def define_module(self):
        a = self.register_module_input('a', shape=(1, ))
        b = self.register_module_input('b', shape=(1, ))
        c = self.register_module_input('c', shape=(1, ))
        self.register_constraint('c', lower=2.5, upper=4, )
        
        d = c**(1/3)
        self.register_module_output('d', d)
        self.register_objective('d')

        implicit_module = ModuleMaker(module=None)
        y = implicit_module.register_module_input('y')
        a_sub = implicit_module.register_module_input('a')
        b_sub = implicit_module.register_module_input('b')
        c_sub = implicit_module.register_module_input('c')
        residual = a_sub * y**2 +  b_sub * y - c_sub
        implicit_module.register_module_output('residual', residual)
        solve_residual = self.create_implicit_operation(implicit_module)
        solve_residual.declare_state('y', residual='residual')#, bracket=(-5, 5))
        solve_residual.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_residual.linear_solver = ScipyKrylov()

        y = solve_residual(a, b, c)

        self.register_module_output('z', y**2)
        
        self.add_module(implicit_module, 'implicit_module')

        implicit_module_2 = ModuleMaker()
        theta = implicit_module_2.register_module_input('theta')
        residual = csdl.sin(theta)
        implicit_module_2.register_module_output('sin_residual', residual)
        solve_residual_2 = self.create_implicit_operation(implicit_module_2)
        solve_residual_2.declare_state('theta', residual='sin_residual', bracket=(-1, 1))
        # solve_residual_2.nonlinear_solver = NewtonSolver
        
        # theta = self.register_module_input('theta')
        theta = solve_residual_2()
        
        print('implicit inputs', implicit_module.module_inputs)
        self.register_module_output('angle', theta * 2)
        self.add_module(implicit_module_2,'sin_implicit')


class ExCustomExplicitOperation(csdl.CustomExplicitOperation):
    def initialize(self): pass

    def define(self):
        self.add_input('y')
        self.add_output('w')

    def compute(self, inputs, outputs):
        y = inputs['y']
        
        outputs['w'] = y**3

class TestModuleCSDL3(ModuleMaker):
    def define_module(self):
        y = self.register_module_input('y')
        w = csdl.custom(y, op=ExCustomExplicitOperation())
        self.register_module_output('w', w)
        

class TestModuleCSDL(ModuleMaker):
    def initialize_module(self):
        self.parameters.declare('test_parameter')
    
    def define_module(self):
        p = self.parameters['test_parameter']
        # some_var = self.register_module_input('some_var')
        # self.register_module_output('some_operation', csdl.log(some_var))
        
        test_module_1 = TestModule1CSDL(
            module=self.module,
            scaling_factor=1.2
        )
        self.add_module(test_module_1, 'test_module_1')
        # print('module_promoted_vars', test_module_1.promoted_vars)
        # print('module_inputs', test_module_1.module_inputs)
        
        test_module_2 = TestModule2CSDL(module=None)
        self.add_module(test_module_2, 'test_module_2')
        # print('module_promoted_vars', test_module_2.promoted_vars)
        # print('module_inputs', test_module_2.module_inputs)

        # test_module_3 = TestModuleCSDL3(module=None)
        # self.add_module(test_module_3, 'test_module_3')
        y = self.register_module_input('y')
        w = csdl.custom(y, op=ExCustomExplicitOperation())
        print('type W', type(w))
        self.register_module_output('w', w)
        self.register_constraint('y', equals=2)


class TestModule(Module):
    def assemble_module(self):
        module_maker = TestModuleCSDL(
            module=self,
            test_parameter=1.,
        )
        return module_maker


test_module = TestModule()
test_module.set_module_input('a', val=5, dv_flag=True)
test_module.set_module_input('b', val=0, dv_flag=True)
test_module.set_module_input('some_var', val=10)

module_maker = test_module.assemble_module()
test_module_csdl = module_maker.assemble_csdl()
module_maker.generate_html()


sim = Simulator(test_module_csdl)
sim.run()
# sim.check_partials()
print('a', sim['a'])
print('b', sim['b'])
print('c', sim['c'])
print('d', sim['d'])
print('x', sim['x'])
print('y', sim['y'])
print('z', sim['z'])
print('w', sim['w'])
# print('test_var', sim['test_var'])
# print('some_var', sim['some_var'])
# print('some_operation', sim['some_operation'])
exit()

prob = CSDLProblem(problem_name='module_test', simulator=sim)
optimizer = SNOPT(
    prob, 
    Major_iterations = 100, 
    Major_optimality=1e-7, 
    Major_feasibility=1e-8,
    append2file=True,
)
# optimizer = SLSQP(prob, maxiter=100, ftol=1e-7)

optimizer.solve()
print('a', sim['test_module_1.a'])
print('b', sim['test_module_1.b'])
print('c', sim['test_module_1.c'])
print('d', sim['d'])
print('x', sim['x'])
print('y', sim['y'])
print('z', sim['z'])