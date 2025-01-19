from pytikz.dsm import DSM


def make_xdsm(data_dict, name='dsm'):
    dsm = DSM()

    diagonals = data_dict['diagonals']
    off_diagonals = data_dict['off_diagonals']

    for diag, color, stack in diagonals:
        dsm.add_diag(diag, diag, color=color, stack=stack)

    for src, tgt, vars, stack in off_diagonals:
        dsm.add_offdiag(src, tgt, vars, stack=stack)


    dsm.write('dsm')

# test_date_dict = {
#     'diagonals' : [
#         ('Optimizer', 'green!20', False ),
#         ('Aerodynamics', 'red!20', True),
#         ('Structures', 'red!20', True),
#         ('Fuel Burn', 'blue!20', False)
#     ],
#     'off_diagonals' : [
#         ('Optimizer', 'Aerodynamics', r'$x_a$', True),
#         ('Structures', 'Fuel Burn', r'$y_s$, $x_s$', True),
#         ('Structures', 'Aerodynamics', r'$x_s$', True),
#         ('Aerodynamics', 'Structures',  r'$x_a$', True)
#     ]
# }

# make_xdsm(test_date_dict)