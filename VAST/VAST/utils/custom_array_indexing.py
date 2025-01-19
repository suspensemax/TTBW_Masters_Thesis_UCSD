import csdl
import numpy as np

class MatrixIndexing(csdl.CustomExplicitOperation):
    '''
    This operation is used to index a matrix with a vector of indices.

    parameters
    ----------
    matrix : csdl 2d array [num_nodes, nn]
        Matrix to be indexed
    ind_array : numpy 0d array [n_ind,]
        array of indices

    Returns
    -------
    indexed_matrix : csdl 2d array [num_nodes, n_ind]
        Indexed matrix    
    '''
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('ind_array', types=np.ndarray)
        self.parameters.declare('out_name', types=str)
    def define(self):
        in_name = self.parameters['in_name']
        in_shape = self.parameters['in_shape']
        ind_array = self.parameters['ind_array']
        out_name = self.parameters['out_name']

        num_nodes = in_shape[0]
        n_ind = ind_array.size
        out_shape = (num_nodes,n_ind)

        self.add_input(in_name,shape=in_shape)
        self.add_output(out_name,shape=out_shape)

        row_indices  = np.arange(num_nodes*n_ind)
        col_indices = np.arange(in_shape[0]*in_shape[1]).reshape(in_shape)[:,ind_array].flatten()
        
        self.declare_derivatives(out_name, in_name, rows=row_indices,cols=col_indices,val=np.ones(row_indices.size))

    def compute(self, inputs, outputs):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']
        ind_array = self.parameters['ind_array']

        outputs[out_name] = inputs[in_name][:,ind_array]

if __name__ == "__main__":
    import python_csdl_backend

    num_nodes = 3
    nn = 7
    in_name = 'in_mat'
    out_name = 'out_mat'
    in_shape = (num_nodes,nn)
    ind_array = np.array([0,2,3,3,4,5,5,6])
    out_name = 'out_mat'

    model = csdl.Model()
    in_mat = model.create_input(in_name, val=np.random.random(in_shape))
    out_mat = csdl.custom(in_mat,
                          op=MatrixIndexing(in_name=in_name,
                                            in_shape=in_shape,
                                            ind_array=ind_array,
                                            out_name=out_name))

    model.register_output(out_name, out_mat)
    sim = python_csdl_backend.Simulator(model)

    sim.run()
    sim.check_partials(compact_print=True)
    print('in_mat:',sim['in_mat'].shape,'\n',sim['in_mat'])
    print('out_mat:',sim['out_mat'].shape,'\n',sim['out_mat'])