

class FluidProblem(object):
    '''
    FluidProblem is a class containing the mesh and the dictionaries of
    the solver option.
    '''

    def __init__(self, solver_option, problem_type, num_ts=1, num_nodes=1, **kwargs):

        self.solver_option = solver_option # val=['LL', 'VLM', 'PM']
        self.problem_type = problem_type  # val=['fixed_wake', 'prescribed_wake', 'free_wake', 'non_prescibed_geometry']
        
        self.symmetry = bool()      # val=[True, False]
        self.lifting_surface_dict = dict()
        self.non_lifting_surface_dict = dict()

        self.wake_surface_dict = dict()
        

    def add_lifting_surface(self, name, shape, *arguments):
        if name in self.non_lifting_surface_dict:
            raise ValueError('name '+name+ 'has already been used for an non-lifting surface')

        self.lifting_surface_dict[name] = dict(
            shape=shape,
            arguments=arguments,
        )

    def add_non_lifting_surface(self, name, shape, *arguments):
        if name in self.lifting_surface_dict:
            raise ValueError('name '+name+ 'has already been used for an lifting surface')
        self.non_lifting_surface_dict[name] = dict(
            shape=shape,
            arguments=arguments,
        )
        if self.non_lifting_surface_dict.keys() and (self.solver_option=='VLM' or self.solver_option=='LL'):
            raise ValueError('non_lifting_surface_dict is not empty, this is not allowed for VLM or LL')

    def add_wake(self, shape=None):
        for name in self.lifting_surface_dict:
            self.wake_surface_dict[name+'_wake']=dict(
                shape=shape,
            )

