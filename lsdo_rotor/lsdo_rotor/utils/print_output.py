import numpy as np
import pandas as pd 
pd.set_option('colheader_justify', 'center')
import os
from lsdo_rotor import BEM, PittPeters


def print_output(sim, rotor, comprehensive_print : bool=False, write_to_csv : bool=False, file_name : str=None):
    """
    Function to print out BEM outputs and write them to a csv file if desired.
    """
    if write_to_csv is False and file_name is not None:
        raise Exception("Cannot specifiy 'file_name' if 'write_to_csv' is False")
    if not isinstance(file_name, (str, type(None))):
        raise TypeError("argument 'file_name' must be of type string")

    bem_object = None
    pitt_peters_object = None
    for operation_name, operation in rotor.operations.items():
        if isinstance(operation, BEM):
            bem_object = operation
        elif isinstance(operation, PittPeters):
            pitt_peters_object = operation

    if bem_object is not None:
        B = bem_object.parameters['BEM_parameters'].parameters['num_blades']
        rotor_name = bem_object.name
    elif pitt_peters_object is not None:
        B = pitt_peters_object.parameters['pitt_peters_parameters'].parameters['num_blades']
        rotor_name = pitt_peters_object.name

    T = sim[f'{rotor_name}.T'].flatten()
    Q = sim[f'{rotor_name}.Q'].flatten()
    M = sim[f'{rotor_name}.M'].flatten()
    eta = sim[f'{rotor_name}.eta'].flatten()
    FM = sim[f'{rotor_name}.FOM'].flatten()
    C_T = sim[f'{rotor_name}.C_T'].flatten()
    C_Q = sim[f'{rotor_name}.C_Q'].flatten()
    C_P = sim[f'{rotor_name}.C_P_compute'].flatten()

    chord = sim[f'{rotor_name}._chord'].flatten()
    dr = sim[f'{rotor_name}._dr'].flatten()
    twist = sim[f'{rotor_name}._pitch'].flatten()
    radius = sim[f'{rotor_name}._radius'].flatten()
    R = sim[f'{rotor_name}.propeller_radius'].flatten()

    sigma = B * np.trapz(chord, radius) / np.pi / R**2
    na = '-----'
    high_level_data = {
        'Efficiency' : [na, np.round(eta, 2)],
        'Figure of merit' : [na, np.round(FM, 2)],
        'Thrust' : [np.round(T, 2), np.round(C_T, 2)], 
        'Torque' : [np.round(Q, 2), np.round(C_Q, 2)], 
        'Blade solidity' : [na, np.round(sigma, 3)],
        'Blade loading': [na, np.round(C_T/sigma, 3)],
        'Disk loading' : [np.round(T / np.pi/R**2, 3),  na],
        'Moments (Mx, My, Mz)' : [np.round(M, 2), na],
    }

    high_level_df = pd.DataFrame(data=high_level_data)
    # s = high_level_df.style.format('{:.0f}').hide([('Random', 'Tumour'), ('Random', 'Non-Tumour')], axis="columns")
    high_level_df.style.set_properties(**{'text-align': 'center'})
    high_level_df.index = ['Value (SI)', 'Coeff./ dim.-less qty.']

    # distributions 
    radius = sim[f'{rotor_name}._radius'][0, :, 0].flatten()
    twist = sim[f'{rotor_name}._pitch'][0, :, 0].flatten() * 180/np.pi
    chord = sim[f'{rotor_name}._chord'][0, :, 0].flatten()
    dT = sim[f'{rotor_name}._dT'][0, :, 0].flatten()
    dQ = sim[f'{rotor_name}._dQ'][0, :, 0].flatten()
    aoa = sim[f'{rotor_name}.alpha_distribution'][0, :, 0].flatten() * 180/np.pi
    if bem_object is not None:
        Cl = sim[f'{rotor_name}.Cl'] # [0, :, 0].flatten()
        Cd = sim[f'{rotor_name}.Cd']# [0, :, 0].flatten()
        LoD = Cl/Cd
    else:
        Cl = sim[f'{rotor_name}.Cl'][0, :, 0].flatten()
        Cd = sim[f'{rotor_name}.Cd'][0, :, 0].flatten()
        LoD = Cl/Cd

    distributions = {
        'radius' : np.round(radius, 3),
        'chord length' : np.round(chord, 3),
        'blade twist (deg)' : np.round(twist, 3), 
        'dT' : np.round(dT, 3),
        'dQ' : np.round(dQ, 3),
        'AoA (deg)' : np.round(aoa, 3),
        'Cl' : np.round(Cl, 3),
        'Cd' : np.round(Cd, 3),
        'Cl/Cd' : np.round(LoD, 3),
    }

    distributions_df = pd.DataFrame(data=distributions)
    print('\n')
    message1 =  '-------------------------------------' + '\n' \
                '| High-level performance parameters |' + '\n' + \
                '-------------------------------------'
    print(message1)
    print(high_level_df)
    print('\n')
    message2 =  '----------------' + '\n' \
                '| Distributions |' + '\n' + \
                '----------------'
    print(message2)
    print(distributions_df)

    if write_to_csv == True:
        cwd = os.getcwd()
        file_path_1 = cwd + f'/{file_name}_performance.csv'
        file_path_2 = cwd + f'/{file_name}_performance_distributions.csv'
        high_level_df.to_csv(file_path_1)
        distributions_df.to_csv(file_path_2)
    # print(s)

    if comprehensive_print:
        message3 =  '---------------------------' + '\n' \
                    '| User-registered outputs |' + '\n' + \
                    '---------------------------'
        print('\n')
        print(message3)
        for output, var in rotor.outputs.items():
            try:
                var.value = sim[f"{var.operation.name}.{var.name}"]
                print('Variable name:\t\t', var.name)
                print('Variable operation:\t', var.operation.name)
                print('Variable value:',  '\t' + str(var.value).replace('\n', '\n\t\t\t'))
                print('\n')
            except:
                pass
