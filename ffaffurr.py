#!/usr/bin/python3

import sys
import os
import copy
import math
import numpy
import operator
import pandas
import re
import ast

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def printf(format, *args):
    sys.stdout.write(format % args)


############################################################
# main function
############################################################

def main():

    print("\n==================================================================")
    print("Framework For Adjusting Force Fields Using Regularized Regression"  )
    print("==================================================================\n")

    # read input keywords from input file 'ffaffurr.input'
    global dict_keywords
    dict_keywords = get_input()

    # get original FF parameters from TINKER provided 'ffaffurr.input.originalFF'
    #   --> run '/path/to/tinker/bin/directory/analyze -k key <txyz-file> P ALL > ffaffurr.input.originalFF'
    #   here ONLY the file ffaffurr.input.originalFF is read in
    #   (TINKER itself is NOT called)
    global n_atoms              # number of atoms
    global n_atom__type         # dictionary: atom number --> atom type
    global n_atom__class        # dictionary: atom number --> atom class
    global type__class          # dictionary: atom type --> atom class
    global class__symbol        # dictionary: atom class --> atom symbol
    global n_atom__atomic       # dictionary: atom number --> atomic number
    global n_atom__mass         # dictionary: atom number --> mass
    global n_atom__valence      # dictionary: atom number --> valence
    global class__atomic        # dictionary: atom class --> atomic number
    global class__mass          # dictionary: atom class --> mass
    global class__valence       # dictionary: atom class --> valence
    global n_atom__description  # dictionary: atom number --> description
    global type__description    # dictionary: atom type --> description
    global list_123_interacts   # list of 123 interactions (needed for angle bending calculations)
    global list_1234_interacts  # list of 1234 interactions (needed for torsional angle calculations)
    global list_imp1234_interacts  # list of 1234 improper interactions (needed for improper torsional angle calculations)
    n_atoms, \
     n_atom__type, \
     n_atom__class, \
     type__class, \
     class__symbol, \
     n_atom__atomic, \
     n_atom__mass, \
     n_atom__valence, \
     class__atomic, \
     class__mass, \
     class__valence, \
     n_atom__description, \
     type__description, \
     origFF___type__charge, \
     origFF___type__sigma, \
     origFF___type__epsilon, \
     origFF___classpair__Kb, \
     origFF___classpair__r0, \
     origFF___classtriple__Ktheta, \
     origFF___classtriple__theta0, \
     list_123_interacts, \
     origFF___classquadruple__V1, \
     origFF___classquadruple__V2, \
     origFF___classquadruple__V3, \
     list_1234_interacts, \
     origFF___classquadruple__impV2, \
     list_imp1234_interacts = get_originalFF_params()

    # get 1-2-interactions, ..., 1-5-interactions between pairs of atoms
    #   --> get interactions from TINKER provided 'ffaffurr.input.interactionsFF'
    #   --> run '/path/to/tinker/bin/directory/analyze -k key <txyz-file> C ALL > ffaffurr.input.interactionsFF'
    #   here ONLY the file ffaffurr.input.interactionsFF is read in
    #   (TINKER itself is NOT called)
    global list_12_interacts
    global list_13_interacts
    global list_14_interacts
    global list_15_interacts
    list_12_interacts, \
     list_13_interacts, \
     list_14_interacts, \
     list_15_interacts = get_FFinteractions()

    # get pair-wise charge parameters (original FF) for Coulomb interactions
    origFF___pairs__charge = get_charge_pairs_params(origFF___type__charge)

    # get (type)pair-wise vdW parameters (sigma & epsilon) for original FF
    #   - original FF --> geometric mean for sigma & epsilon
    origFF___pairs__sigma, \
     origFF___pairs__epsilon, \
     origFF___typepairs__sigma, \
     origFF___typepairs__epsilon = get_origFF_vdW_pairs_params(origFF___type__sigma, \
                                                                origFF___type__epsilon)

    # test function only used for cross-ckecking with TINKER
    # --> all energy contributions (bonds, angles, torsions, impropers, vdW, Coulomb)
    #     are calculated using a file called 'tinker.xyz'
    #     --> energies can then be cross-checked with TINKER
#    cross_check_tinker(origFF___classpair__Kb, \
#                        origFF___classpair__r0, \
#                        origFF___classtriple__Ktheta, \
#                        origFF___classtriple__theta0, \
#                        origFF___classquadruple__V1, \
#                        origFF___classquadruple__V2, \
#                        origFF___classquadruple__V3, \
#                        origFF___classquadruple__impV2, \
#                        origFF___pairs__sigma, \
#                        origFF___pairs__epsilon, \
#                        origFF___pairs__charge)

    # get logfiles (FHI-aims output files)
    # from input file 'ffaffurr.input.FHI-aims-logfiles'
    global list_logfiles
    list_logfiles = get_logfiles()

    # get xyz for all logfiles
    global listofdicts_logfiles___n_atom__xyz
    listofdicts_logfiles___n_atom__xyz = get_logfiles_xyz()

    # fine-tune r0 (or not)
    if dict_keywords['fine_tune_r0'] == True:
        newFF___classpair__r0 = get_average_r0()
    elif dict_keywords['fine_tune_r0'] == False:
        newFF___classpair__r0 = copy.deepcopy(origFF___classpair__r0)

    # fine-tune theta0 (or not)
    if dict_keywords['fine_tune_theta0'] == True:
        newFF___classtriple__theta0 = get_average_theta0()
    elif dict_keywords['fine_tune_theta0'] == False:
        newFF___classtriple__theta0 = copy.deepcopy(origFF___classtriple__theta0)

    # fine-tune partial charge (or not)
    if dict_keywords['fine_tune_charge'] == 'False':
        newFF___type__charge = copy.deepcopy(origFF___type__charge)
    elif ( dict_keywords['fine_tune_charge'] == 'Hirshfeld' ) or ( dict_keywords['fine_tune_charge'] == 'ESP') or ( dict_keywords['fine_tune_charge'] == 'RESP') :
        newFF___type__charge = get_average_charge()

    # get pair-wise charge parameters (new FF) for Coulomb interactions
    newFF___pairs__charge = get_charge_pairs_params(newFF___type__charge)

    # fine-tune sigma parameters for (type)pairs from TS using R0eff (or not)
    if dict_keywords['fine_tune_sigma'] == 'False':
        newFF___pairs__sigma = copy.deepcopy(origFF___pairs__sigma)
        newFF___typepairs__sigma = copy.deepcopy(origFF___typepairs__sigma)
    elif dict_keywords['fine_tune_sigma'] == 'TS':
        newFF___pairs__sigma, \
         newFF___typepairs__sigma = get_sigmas_TS()
        # if requested, set sigmas to zero that were also zero in original FF
        if dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] == True:
            newFF___pairs__sigma = set_some_dict_entries_zero(origFF___pairs__sigma, \
                                                               newFF___pairs__sigma)
            newFF___typepairs__sigma = set_some_dict_entries_zero(origFF___typepairs__sigma, \
                                                                   newFF___typepairs__sigma)

    # fine-tune epsilon parameters for (type)pairs from TS (or not)
    #   (if regression on either TS or MBD or total energies is requested, we do it later)
    if dict_keywords['fine_tune_epsilon'] == 'False':
        newFF___pairs__epsilon = copy.deepcopy(origFF___pairs__epsilon)
        newFF___typepairs__epsilon = copy.deepcopy(origFF___typepairs__epsilon)
    elif dict_keywords['fine_tune_epsilon'] == 'TS':
        newFF___pairs__epsilon, \
         newFF___typepairs__epsilon = get_epsilons_TS()
        # if requested, set epsilons to zero that were also zero in original FF
        if dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] == True:
            newFF___pairs__epsilon = set_some_dict_entries_zero(origFF___pairs__epsilon, \
                                                                 newFF___pairs__epsilon)
            newFF___typepairs__epsilon = set_some_dict_entries_zero(origFF___typepairs__epsilon, \
                                                                     newFF___typepairs__epsilon)


    ########################################################################################
    # now we start to collect (and/or calculate) all kinds of energies (total energies,    #
    #   force field energy terms, energy contributions, etc.) for all FHI-aims logfiles    #
    #   and store them into one big DataFrame (named 'data') that we can then access       #
    #   easily when wanting to apply Linear or Ridge or Lasso regression                   #
    ########################################################################################

    global data
    data = pandas.DataFrame()

    # get total energies from high-level (DFT; FHI-aims) calculations output
    # also: get vdW(TS) or MBD energies if requested
    get_fhiaims_energies()


    # get bonding energies with newFF parameters
    # bonding terms are of form: kB * (r-r0)**2
    get_bonding_energies(origFF___classpair__Kb, newFF___classpair__r0)


    # get angles energies with newFF parameters
    # angles bending terms are of form: Ktheta * (theta-theta0)**2
    get_angles_energies(origFF___classtriple__Ktheta, newFF___classtriple__theta0)


    # get torsions energies with origFF parameters
    #   (not really useful at the moment but I put it in just for completeness
    #    and in case I need this function in the future for some reason)
    # -> torsion (dihedral angles) terms are of form:
    #       (V1 / 2) * ( 1+cos(  phi) )
    #       (V2 / 2) * ( 1-cos(2*phi) )
    #       (V3 / 2) * ( 1+cos(3*phi) )
    get_torsions_energies(origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3)


    # get improper torsions energies with origFF parameters
    # -> improper torsion terms are of form:
    #       (V2imp / 2) * ( 1-cos(2*phi) )
    get_improper_energies(origFF___classquadruple__impV2)


    # get Coulomb energies with newFF parameters
    # -> Coulomb terms are of form: f * q1*q2 / r12
    #          {   0 for 1-2-interactions and 1-3-interactions
    #       f ={ 1/2 for 1-4-interactions
    #          {   1 for 1-5-interactions and higher
    get_Coulomb_energies(newFF___pairs__charge)


    # get Coulomb energy contributions per 1-X-interaction with newFF parameters
    # -> terms are of form: q1*q2 / r12
    get_Coulomb_energyContribsPer_1_X_interaction(newFF___pairs__charge)


    # get vdW energies with origFF parameters
    # -> van der Waals (vdW; Lennard-Jones) terms are of form:
    #       4*epsilon * f * [ (sigma/r)**12 - (sigma/r)**6 ]
    #            {   0 for 1-2-interactions and 1-3-interactions
    #         f ={ 1/2 for 1-4-interactions
    #            {   1 for 1-5-interactions and higher
    #get_vdW_energies(newFF___pairs__sigma, newFF___pairs__epsilon, 'Evdw_(origFF)')
    get_vdW_energies(origFF___pairs__sigma, origFF___pairs__epsilon, 'Evdw_(origFF)')


    # do regression for estimating fudge factors in Coulomb energies
    f14 = 0.5 # default
    f15 = 1.0 # default
    dielectric = 1.0 # default
    if ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == True ):
        f14, f15, dielectric = do_regression_Coulomb_fudge_factors() 
        # reset Coulomb energies with new fudge factors
        data['Ecoul_(FF)'] = (1./dielectric) * (   f14 * data['Ecoul_(FF)_14Intacts'] \
                                                 + f15 * data['Ecoul_(FF)_15Intacts'] \
                                                 +   data['Ecoul_(FF)_16plusIntacts'] )
        

#    with pandas.option_context('display.max_rows', 999, 'display.max_columns', 999):
#        print(data.head())
#        print(data)


    # get bonding energy contributions classpair-wise
    #   (not really useful at the moment but I put it in just for completeness
    #    and in case I need this function in the future for some reason)
    # -> bonding contributions are of form: (r-r0)**2
#    get_bondEnergyContribs(newFF___classpair__r0)


    # get angles energy contributions classtriple-wise
    #   (not really useful at the moment but I put it in just for completeness
    #    and in case I need this function in the future for some reason)
    # -> angles bending contributions are of form: (theta-theta0)**2
#    get_anglesEnergyContribs(newFF___classtriple__theta0)


    # get improper torsions energy contributions classquadruple-wise
    #   (not really useful at the moment but I put it in just for completeness
    #    and in case I need this function in the future for some reason)
    # -> improper torsion contributions are of form:
    #       0.5 * ( 1-cos(2*phi) )
    if dict_keywords['fine_tune_imptorsionalV'] == True:
        get_impropsEnergyContribs()


    # get torsions energy contributions classquadruple-wise
    # torsion (dihedral angles) contributions are of form:
    #       0.5 * ( 1+cos(  phi) )
    #       0.5 * ( 1-cos(2*phi) )
    #       0.5 * ( 1+cos(3*phi) )
    if dict_keywords['fine_tune_torsionalV'] == True:
        get_torsionsEnergyContribs(origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3)

    # get vdW energy contributions typepair-wise
    # -> van der Waals (vdW; Lennard-Jones) contributions are of form:
    #       4 * f * [ (sigma/r)**12 - (sigma/r)**6 ]
    #            {   0 for 1-2-interactions and 1-3-interactions
    #         f ={ 1/2 for 1-4-interactions
    #            {   1 for 1-5-interactions and higher
    get_vdWenergyContribs(newFF___pairs__sigma)


    # do regression for vdW (either TS or MBD or total energy) to estimate epsilon parameters
    if ( dict_keywords['fine_tune_epsilon'] == 'RegressionMBD' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionTS' ):
        newFF___pairs__epsilon, \
         newFF___typepairs__epsilon = do_regression_epsilon_vdW() 
    elif ( dict_keywords['fine_tune_epsilon'] == 'RegressionTot' ):
        newFF___pairs__epsilon, \
         newFF___typepairs__epsilon = do_regression_epsilon_Tot(origFF___typepairs__epsilon) 


    # get vdW energies with newFF parameters
    # -> van der Waals (vdW; Lennard-Jones) terms are of form:
    #       4*epsilon * f * [ (sigma/r)**12 - (sigma/r)**6 ]
    #            {   0 for 1-2-interactions and 1-3-interactions
    #         f ={ 1/2 for 1-4-interactions
    #            {   1 for 1-5-interactions and higher
    get_vdW_energies(newFF___pairs__sigma, newFF___pairs__epsilon, 'Evdw_(FF)')


    # do regression to estimate torsional parameters (V1, V2, V3)
    if dict_keywords['fine_tune_torsionalV'] == True:
        newFF___classquadruple__V1, \
         newFF___classquadruple__V2, \
         newFF___classquadruple__V3 = do_regression_torsionalV() 
    elif dict_keywords['fine_tune_torsionalV'] == False:
        newFF___classquadruple__V1 = copy.deepcopy(origFF___classquadruple__V1)
        newFF___classquadruple__V2 = copy.deepcopy(origFF___classquadruple__V2)
        newFF___classquadruple__V3 = copy.deepcopy(origFF___classquadruple__V3)

    # do regression to estimate improper torsional parameters (V2imp)
    if dict_keywords['fine_tune_imptorsionalV'] == True:
        newFF___classquadruple__impV2 = do_regression_imptorsionalV() 
    elif dict_keywords['fine_tune_imptorsionalV'] == False:
        newFF___classquadruple__impV2 = copy.deepcopy(origFF___classquadruple__impV2)

    ############################################################
    # print all kinds of information
    ############################################################

    # print atom type/class overview
    print_atom_type_class_overview()

    # print original and new r0 parameters classpair-wise, and kB parameters (unaltered))
    print_r0_kB_params(origFF___classpair__r0, newFF___classpair__r0, origFF___classpair__Kb)

    # print original and new theta0 parameters classtriple-wise, and Ktheta parameters (unaltered)
    print_theta0_params(origFF___classtriple__theta0, newFF___classtriple__theta0, origFF___classtriple__Ktheta)

    # print torsions parameters (V1, V2, V3)
    print_torsions_params(origFF___classquadruple__V1, \
                           origFF___classquadruple__V2, \
                           origFF___classquadruple__V3, \
                           newFF___classquadruple__V1, \
                           newFF___classquadruple__V2, \
                           newFF___classquadruple__V3)

    # print improper torsions parameters (unaltered)
    print_improps_params(origFF___classquadruple__impV2, newFF___classquadruple__impV2)

    # print original and new sigma parameters typepair-wise
    print_sigma_params(origFF___typepairs__sigma, newFF___typepairs__sigma)

    # print original and new epsilon parameters typepair-wise
    print_epsilon_params(origFF___typepairs__epsilon, newFF___typepairs__epsilon)

    # print original and new charge parameters type-wise
    print_charge_params(origFF___type__charge, newFF___type__charge)

    # print original and new fudge factors
    print_fudge_factors(f14, f15, dielectric)


    # write TINKER force field parameter file
    write_TINKER_ff_params_file(newFF___classpair__r0, \
                                 origFF___classpair__Kb, \
                                 newFF___classtriple__theta0, \
                                 origFF___classtriple__Ktheta, \
                                 newFF___classquadruple__V1, \
                                 newFF___classquadruple__V2, \
                                 newFF___classquadruple__V3, \
                                 origFF___classquadruple__impV2, \
                                 newFF___classquadruple__impV2, \
                                 newFF___typepairs__sigma, \
                                 newFF___typepairs__epsilon, \
                                 newFF___type__charge, \
                                 f14, f15, dielectric)

    print("========")
    print("Goodbye!"  )
    print("========\n")


############################################################
# read input keywords from input file 'ffaffurr.input'
############################################################

def get_input():

    # check if file is there
    if not os.path.exists('ffaffurr.input'):
        sys.exit('== Error: Input file \'ffaffurr.input\' does not exist. Exiting now...')
    else:
        print('Now reading from input file \'ffaffurr.input\'...')

    # read file
    in_file = open("ffaffurr.input", 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    dict_keywords = {}
    
    astring = get_input_loop_lines(lines, 'readparamsfromffaffurr')
    if astring in ['True', 'true']:
        dict_keywords['readparamsfromffaffurr'] = True
    elif astring in ['False', 'false']:
        dict_keywords['readparamsfromffaffurr'] = False
    else:
        sys.exit('== Error: keyword \'readparamsfromffaffurr\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_r0')
    if astring in ['True', 'true']:
        dict_keywords['fine_tune_r0'] = True
    elif astring in ['False', 'false']:
        dict_keywords['fine_tune_r0'] = False
    else:
        sys.exit('== Error: keyword \'fine_tune_r0\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_theta0')
    if astring in ['True', 'true']:
        dict_keywords['fine_tune_theta0'] = True
    elif astring in ['False', 'false']:
        dict_keywords['fine_tune_theta0'] = False
    else:
        sys.exit('== Error: keyword \'fine_tune_theta0\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_torsionalV')
    if astring in ['Regression', 'regression']:
        dict_keywords['fine_tune_torsionalV'] = True
    elif astring in ['False', 'false']:
        dict_keywords['fine_tune_torsionalV'] = False
    else:
        sys.exit('== Error: keyword \'fine_tune_torsionalV\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_torsionalV'] == True ):
        astring = get_input_loop_lines(lines, 'RegressionTorsionalVall')
        if astring in ['True', 'true']:
            dict_keywords['RegressionTorsionalVall'] = True
        elif astring in ['False', 'false']:
            dict_keywords['RegressionTorsionalVall'] = False
        else:
            sys.exit('== Error: keyword \'RegressionTorsionalVall\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_torsionalV'] == True ):
        astring = get_input_loop_lines(lines, 'Regression_torsionalV_Method')
        if astring in ['LinearRegression', 'linearregression', 'linear']:
            dict_keywords['Regression_torsionalV_Method'] = 'LinearRegression'
        elif astring in ['Ridge', 'ridge']:
            dict_keywords['Regression_torsionalV_Method'] = 'Ridge'
        elif astring in ['Lasso', 'lasso', 'LASSO']:
            dict_keywords['Regression_torsionalV_Method'] = 'Lasso'
        else:
            sys.exit('== Error: keyword \'Regression_torsionalV_Method\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_torsionalV'] == True ) and ( ( dict_keywords['Regression_torsionalV_Method'] == 'Ridge' ) or ( dict_keywords['Regression_torsionalV_Method'] == 'Lasso' ) ):
        astring = get_input_loop_lines(lines, 'regularization_parameter_torsionalV')
        dict_keywords['regularization_parameter_torsionalV'] = float(astring)
    
    astring = get_input_loop_lines(lines, 'fine_tune_imptorsionalV')
    if astring in ['Regression', 'regression']:
        dict_keywords['fine_tune_imptorsionalV'] = True
    elif astring in ['False', 'false']:
        dict_keywords['fine_tune_imptorsionalV'] = False
    else:
        sys.exit('== Error: keyword \'fine_tune_imptorsionalV\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')
        
    #if ( dict_keywords['fine_tune_imptorsionalV'] == True ):
    #    astring = get_input_loop_lines(lines, 'RegressionTorsionalVall')
    #    if astring in ['True', 'true']:
    #        dict_keywords['RegressionTorsionalVall'] = True
    #    elif astring in ['False', 'false']:
    #        dict_keywords['RegressionTorsionalVall'] = False
    #    else:
    #        sys.exit('== Error: keyword \'RegressionTorsionalVall\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_imptorsionalV'] == True ):
        astring = get_input_loop_lines(lines, 'Regression_imptorsionalV_Method')
        if astring in ['LinearRegression', 'linearregression', 'linear']:
            dict_keywords['Regression_imptorsionalV_Method'] = 'LinearRegression'
        elif astring in ['Ridge', 'ridge']:
            dict_keywords['Regression_imptorsionalV_Method'] = 'Ridge'
        elif astring in ['Lasso', 'lasso', 'LASSO']:
            dict_keywords['Regression_imptorsionalV_Method'] = 'Lasso'
        else:
            sys.exit('== Error: keyword \'Regression_imptorsionalV_Method\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_imptorsionalV'] == True ) and ( ( dict_keywords['Regression_imptorsionalV_Method'] == 'Ridge' ) or ( dict_keywords['Regression_imptorsionalV_Method'] == 'Lasso' ) ):
        astring = get_input_loop_lines(lines, 'regularization_parameter_imptorsionalV')
        dict_keywords['regularization_parameter_imptorsionalV'] = float(astring)
    
    astring = get_input_loop_lines(lines, 'fine_tune_sigma')
    if astring in ['False', 'false']:
        dict_keywords['fine_tune_sigma'] = 'False'
    elif astring in ['TS']:
        dict_keywords['fine_tune_sigma'] = 'TS'
    else:
        sys.exit('== Error: keyword \'fine_tune_sigma\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_epsilon')
    if astring in ['False', 'false']:
        dict_keywords['fine_tune_epsilon'] = 'False'
    elif astring in ['TS']:
        dict_keywords['fine_tune_epsilon'] = 'TS'
    elif astring in ['RegressionTS', 'regressionTS']:
        dict_keywords['fine_tune_epsilon'] = 'RegressionTS'
    elif astring in ['RegressionMBD', 'regressionMBD']:
        dict_keywords['fine_tune_epsilon'] = 'RegressionMBD'
    elif astring in ['RegressionTot', 'regressionTot']:
        dict_keywords['fine_tune_epsilon'] = 'RegressionTot'
    else:
        sys.exit('== Error: keyword \'fine_tune_epsilon\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_sigma'] == 'TS' ) or ( dict_keywords['fine_tune_epsilon'] == 'TS' ):
        astring = get_input_loop_lines(lines, 'SetExplicitlyZero_vdW_SigmaEpsilon')
        if astring in ['True', 'true']:
            dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] = True
        elif astring in ['False', 'false']:
            dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] = False
        else:
            sys.exit('== Error: keyword \'SetExplicitlyZero_vdW_SigmaEpsilon\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_epsilon'] == 'RegressionTS' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionMBD' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionTot' ):
        astring = get_input_loop_lines(lines, 'RegressionEpsilonMethod')
        if astring in ['LinearRegression', 'linearregression', 'linear']:
            dict_keywords['RegressionEpsilonMethod'] = 'LinearRegression'
        elif astring in ['Ridge', 'ridge']:
            dict_keywords['RegressionEpsilonMethod'] = 'Ridge'
        elif astring in ['Lasso', 'lasso', 'LASSO']:
            dict_keywords['RegressionEpsilonMethod'] = 'Lasso'
        else:
            sys.exit('== Error: keyword \'RegressionEpsilonMethod\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( ( dict_keywords['fine_tune_epsilon'] == 'RegressionTS' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionMBD' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionTot' ) ) and ( ( dict_keywords['RegressionEpsilonMethod'] == 'Ridge' ) or ( dict_keywords['RegressionEpsilonMethod'] == 'Lasso' ) ):
        astring = get_input_loop_lines(lines, 'regularization_parameter_epsilon')
        dict_keywords['regularization_parameter_epsilon'] = float(astring)

    if ( ( dict_keywords['fine_tune_epsilon'] == 'RegressionTS' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionMBD' ) or ( dict_keywords['fine_tune_epsilon'] == 'RegressionTot' ) ) and ( dict_keywords['RegressionEpsilonMethod'] == 'Lasso' ):
        astring = get_input_loop_lines(lines, 'RestrictRegressionEpsilonPositive')
        if astring in ['True', 'true']:
            dict_keywords['RestrictRegressionEpsilonPositive'] = True
        elif astring in ['False', 'false']:
            dict_keywords['RestrictRegressionEpsilonPositive'] = False
        else:
            sys.exit('== Error: keyword \'RestrictRegressionEpsilonPositive\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_charge')
    if astring in ['False', 'false']:
        dict_keywords['fine_tune_charge'] = 'False'
    elif astring in ['Hirshfeld', 'hirshfeld']:
        dict_keywords['fine_tune_charge'] = 'Hirshfeld'
    elif astring in ['ESP', 'esp']:
        dict_keywords['fine_tune_charge'] = 'ESP'
    elif astring in ['RESP', 'resp']:
        dict_keywords['fine_tune_charge'] = 'RESP'
    else:
        sys.exit('== Error: keyword \'fine_tune_charge\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    astring = get_input_loop_lines(lines, 'fine_tune_Coulomb_fudge_factors')
    if astring in ['True', 'true']:
        dict_keywords['fine_tune_Coulomb_fudge_factors'] = True
    elif astring in ['False', 'false']:
        dict_keywords['fine_tune_Coulomb_fudge_factors'] = False
    else:
        sys.exit('== Error: keyword \'fine_tune_Coulomb_fudge_factors\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    if ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == True ):
        astring = get_input_loop_lines(lines, 'fine_tune_only_f14_and_f15')
        if astring in ['True', 'true']:
            dict_keywords['fine_tune_only_f14_and_f15'] = True
        elif astring in ['False', 'false']:
            dict_keywords['fine_tune_only_f14_and_f15'] = False
        else:
            sys.exit('== Error: keyword \'fine_tune_only_f14_and_f15\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')

    print('\n====\n')

    return(dict_keywords)

def get_input_loop_lines(lines, key):
    string = 'dummy'
    for line in lines:
        line = line.rstrip() # skip blank lines
        if line:
            line = line.lstrip()
            if not line.startswith("#"):
                keyword, keycontent = line.split(None, 1)
                if keyword in [key]:
                    string = keycontent
                    break
    if string in ['dummy']:
        sys.exit('== Error: keyword \''+key+'\' not found in \'ffaffurr.input\'! Exiting now...')
    return(string)


############################################################
# get original FF parameters from TINKER provided 'ffaffurr.input.originalFF'
#   --> run '/path/to/tinker/bin/directory/analyze -k key <txyz-file> P ALL > ffaffurr.input.originalFF'
#   here ONLY the file ffaffurr.input.originalFF is used
#   (TINKER itself is NOT called)
############################################################

def get_originalFF_params():

    # check if file is there
    if not os.path.exists('ffaffurr.input.originalFF'):
        sys.exit('== Error: Input file \'ffaffurr.input.originalFF\' does not exist. Exiting now...')
    else:
        print('Now reading from input file \'ffaffurr.input.originalFF\'...')

    # read file
    in_file = open("ffaffurr.input.originalFF", 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    n_atoms    = -1
    n_bonds    = -1
    n_angles   = -1
    n_torsions = -1
    n_improps  = -1
    n_vdws     = -1

    n_atoms    = get_originalFF_basic_properties(lines, 'Atomic Sites')
    print('    Atomic Sites:        '+str(n_atoms))
    if not n_atoms > 0:
        sys.exit('== Error: Number of atoms should be a positive number. Exiting now...')

    n_bonds    = get_originalFF_basic_properties(lines, 'Bond Stretches')
    print('    Bond Stretches:      '+str(n_bonds))
    if not n_bonds > -1:
        sys.exit('== Error: Number of bonds should be a natural number. Exiting now...')

    n_angles   = get_originalFF_basic_properties(lines, 'Angle Bends')
    print('    Angle Bends:         '+str(n_angles))
    if not n_angles > -1:
        sys.exit('== Error: Number of angles should be a natural number. Exiting now...')

    n_improps  = get_originalFF_basic_properties(lines, 'Improper Torsions')
    print('    Improper Torsions:   '+str(n_improps))
    if not n_improps > -1:
        sys.exit('== Error: Number of improper torsions should be a natural number. Exiting now...')

    n_torsions = get_originalFF_basic_properties(lines, 'Torsional Angles')
    print('    Torsional Angles:    '+str(n_torsions))
    if not n_torsions > -1:
        sys.exit('== Error: Number of torsional angles should be a natural number. Exiting now...')

    n_vdws     = get_originalFF_basic_properties(lines, 'Van der Waals Sites')
    print('    van der Waals Sites: '+str(n_vdws))
    if not n_vdws > -1:
        sys.exit('== Error: Number of van der Waals Sites should be a natural number. Exiting now...')

    print('\n====\n')

    n_atom__type, \
     n_atom__class, \
     type__class, \
     class__symbol, \
     n_atom__atomic, \
     n_atom__mass, \
     n_atom__valence, \
     class__atomic, \
     class__mass, \
     class__valence, \
     n_atom__description, \
     type__description, \
     type__charge, \
     type__sigma, \
     type__epsilon, \
     classpair__Kb, \
     classpair__r0, \
     classtriple__Ktheta, \
     classtriple__theta0, \
     list_123_interacts, \
     classquadruple__V1, \
     classquadruple__V2, \
     classquadruple__V3, \
     list_1234_interacts, \
     classquadruple__impV2, \
     list_imp1234_interacts = get_originalFF_AllFFparams(lines, \
                                                          n_atoms, \
                                                          n_bonds, \
                                                          n_angles, \
                                                          n_torsions, \
                                                          n_improps)

    return(n_atoms, \
            n_atom__type, \
            n_atom__class, \
            type__class, \
            class__symbol, \
            n_atom__atomic, \
            n_atom__mass, \
            n_atom__valence, \
            class__atomic, \
            class__mass, \
            class__valence, \
            n_atom__description, \
            type__description, \
            type__charge, \
            type__sigma, \
            type__epsilon, \
            classpair__Kb, \
            classpair__r0, \
            classtriple__Ktheta, \
            classtriple__theta0, \
            list_123_interacts, \
            classquadruple__V1, \
            classquadruple__V2, \
            classquadruple__V3, \
            list_1234_interacts, \
            classquadruple__impV2, \
            list_imp1234_interacts)

def get_originalFF_basic_properties(lines, key):
    number = -1
    for line in lines:
        if key in line:
            number = int(line.rsplit(None, 1)[-1])
            break
    if number == -1:
        sys.exit('== Error: keyword \''+key+'\' not found in \'ffaffurr.input.originalFF\'! Exiting now...')
    return(number)

def get_originalFF_AllFFparams(lines, n_atoms, n_bonds, n_angles, n_torsions, n_improps):

    n_atom__type = {}
    n_atom__class = {}
    type__class = {}
    class__symbol = {}
    n_atom__atomic = {}
    n_atom__mass = {}
    n_atom__valence = {}
    class__atomic = {}
    class__mass = {}
    class__valence = {}
    n_atom__description = {}
    type__description = {}

    type__charge = {}

    type__sigma = {}
    type__epsilon = {}

    classpair__Kb = {}
    classpair__r0 = {}

    classtriple__Ktheta = {}
    classtriple__theta0 = {}

    list_123_interacts = []

    classquadruple__V1 = {}
    classquadruple__V2 = {}
    classquadruple__V3 = {}

    list_1234_interacts = []

    classquadruple__impV2 = {}

    list_imp1234_interacts = []

    lines_position = lines.index(' Atom Type Definition Parameters :\n')
    for line in lines[lines_position+4:lines_position+4+n_atoms]:
        n_atom, symbol, ttype, cclass, atomic, mass, valence, description = line.split(None, 7)
        n_atom__type[int(n_atom)] = int(ttype)
        n_atom__class[int(n_atom)] = int(cclass)
        type__class[int(ttype)] = int(cclass)
        class__symbol[int(cclass)] = symbol
        n_atom__atomic[int(n_atom)] = int(atomic)
        n_atom__mass[int(n_atom)] = float(mass)
        n_atom__valence[int(n_atom)] = int(valence)
        class__atomic[int(cclass)] = int(atomic)
        class__mass[int(cclass)] = float(mass)
        class__valence[int(cclass)] = int(valence)
        n_atom__description[int(n_atom)] = str(description)
        type__description[int(ttype)] = str(description).rstrip()
        
    lines_position = lines.index(' Angle Bending Parameters :\n')
    for line in lines[lines_position+4:lines_position+4+n_angles]:
        n_angle, n_atom1, n_atom2, n_atom3, Ktheta, theta0 = line.split(None, 5)
        # int(n_atom1) <= int(n_atom3) always (property of TINKER)
        atuple = (int(n_atom1),int(n_atom2),int(n_atom3))
        list_123_interacts.append( atuple )
        
    lines_position = lines.index(' Torsional Angle Parameters :\n')
    for line in lines[lines_position+4:lines_position+4+n_torsions]:
        if len(line.split()) == 5:
            n_torsion, n_atom1, n_atom2, n_atom3, n_atom4 = line.split(None, 4)
        elif len(line.split()) == 7:
            n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, V, phasestring = line.split(None, 6)
        elif len(line.split()) == 9:
            n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, Va, phasestringa, Vb, phasestringb = line.split(None, 8)
        elif len(line.split()) == 11:
            n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, Va, phasestringa, Vb, phasestringb, Vc, phasestringc = line.split(None, 10)
        # int(n_atom2) <= int(n_atom3) always (property of TINKER)
        atuple = (int(n_atom1),int(n_atom2),int(n_atom3),int(n_atom4))
        list_1234_interacts.append( atuple )
    
    lines_position = lines.index(' Improper Torsion Parameters :\n')
    for line in lines[lines_position+4:lines_position+4+n_improps]:
        n_improp, n_atom1, n_atom2, n_atom3, n_atom4, impV2, phase, periodicity = line.split(None, 7)
        atuple = (int(n_atom1),int(n_atom2),int(n_atom3),int(n_atom4))
        list_imp1234_interacts.append( atuple )
        
    if dict_keywords['readparamsfromffaffurr'] == False:
		    
        lines_position = lines.index(' Atomic Partial Charge Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_atoms]:
            n_atom, number, charge = line.split(None, 2)
            type__charge[n_atom__type[int(n_atom)]] = float(charge)
        
        lines_position = lines.index(' Van der Waals Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_atoms]:
            n_atom, number, sigma, epsilon = line.split(None, 3)
            type__sigma[n_atom__type[int(n_atom)]] = float(sigma)
            type__epsilon[n_atom__type[int(n_atom)]] = float(epsilon)
        
        lines_position = lines.index(' Bond Stretching Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_bonds]:
            n_bond, n_atom1, n_atom2, Kb, r0 = line.split(None, 4)
            if n_atom__class[int(n_atom1)] <= n_atom__class[int(n_atom2)]:
                aclasspair = ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)] )
            else:
                aclasspair = ( n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] )
            classpair__Kb[ aclasspair ] = float(Kb)
            classpair__r0[ aclasspair ] = float(r0)
        
        lines_position = lines.index(' Angle Bending Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_angles]:
            n_angle, n_atom1, n_atom2, n_atom3, Ktheta, theta0 = line.split(None, 5)
            if n_atom__class[int(n_atom1)] <= n_atom__class[int(n_atom3)]:
                aclasstriple = ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)] )
            else:
                aclasstriple = ( n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] )
            classtriple__Ktheta[ aclasstriple ] = float(Ktheta)
            classtriple__theta0[ aclasstriple ] = float(theta0)
        
        lines_position = lines.index(' Torsional Angle Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_torsions]:
            if len(line.split()) == 5:
                n_torsion, n_atom1, n_atom2, n_atom3, n_atom4 = line.split(None, 4)
            elif len(line.split()) == 7:
                n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, V, phasestring = line.split(None, 6)
                if phasestring == '0/1' or phasestring == '0/1\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V1[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(V)
                    else:
                        classquadruple__V1[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(V)
                elif phasestring == '180/2' or phasestring == '180/2\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(V)
                    else:
                        classquadruple__V2[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(V)
                elif phasestring == '0/3' or phasestring == '0/3\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(V)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(V)
            elif len(line.split()) == 9:
                n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, Va, phasestringa, Vb, phasestringb = line.split(None, 8)
                if phasestringa == '0/1' or phasestringa == '0/1\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V1[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V1[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                elif phasestringa == '180/2' or phasestringa == '180/2\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V2[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                elif phasestringa == '0/3' or phasestringa == '0/3\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                if phasestringb == '180/2' or phasestringb == '180/2\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Vb)
                    else:
                        classquadruple__V2[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Vb)
                elif phasestringb == '0/3' or phasestringb == '0/3\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Vb)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Vb)
            elif len(line.split()) == 11:
                n_torsion, n_atom1, n_atom2, n_atom3, n_atom4, Va, phasestringa, Vb, phasestringb, Vc, phasestringc = line.split(None, 10)
                if phasestringa == '0/1' or phasestringa == '0/1\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V1[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V1[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                elif phasestringa == '180/2' or phasestringa == '180/2\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V2[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                elif phasestringa == '0/3' or phasestringa == '0/3\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Va)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Va)
                if phasestringb == '180/2' or phasestringb == '180/2\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Vb)
                    else:
                        classquadruple__V2[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Vb)
                elif phasestringb == '0/3' or phasestringb == '0/3\n':
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Vb)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Vb)
                if phasestringc == '0/3' or phasestringc == '0/3\n': # always true
                    if n_atom__class[int(n_atom2)] <= n_atom__class[int(n_atom3)]:
                        classquadruple__V3[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(Vc)
                    else:
                        classquadruple__V3[ ( n_atom__class[int(n_atom4)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom1)] ) ] = float(Vc)
        
        lines_position = lines.index(' Improper Torsion Parameters :\n')
        for line in lines[lines_position+4:lines_position+4+n_improps]:
            n_improp, n_atom1, n_atom2, n_atom3, n_atom4, impV2, phase, periodicity = line.split(None, 7)
            classquadruple__impV2[ ( n_atom__class[int(n_atom1)],n_atom__class[int(n_atom2)],n_atom__class[int(n_atom3)],n_atom__class[int(n_atom4)] ) ] = float(impV2)
            
    elif dict_keywords['readparamsfromffaffurr'] == True:
		# check if file is there
        if not os.path.exists('oplsaa-ffaffurr.prm'):
            sys.exit('== Error: Input file \'oplsaa-ffaffurr.prm\' does not exist. Exiting now...')
        else:
            print('Now reading from input file \'oplsaa-ffaffurr.prm\'...')
        
        # read file
        in_file_ffaffurr = open("oplsaa-ffaffurr.prm", 'r')
        file_lines_ffaffurr = in_file_ffaffurr.readlines()
        lines_ffaffurr = list(file_lines_ffaffurr)
        in_file_ffaffurr.close()
        
        lines_position = lines_ffaffurr.index('      ##  Atomic Partial Charge Parameters  ##\n')
        for line in lines_ffaffurr[lines_position+5:]:
            stringa, atomtype, charge = line.split(None, 2)
            type__charge[int(atomtype)] = float(charge)
            
        lines_position_1 = lines_ffaffurr.index('      ##  Bond Stretching Parameters  ##\n')
        lines_position_2 = lines_ffaffurr.index('      ##  Angle Bending Parameters  ##\n')
        for line in lines_ffaffurr[lines_position_1+5:lines_position_2-4]:
            stringa, atomclass1, atomclass2, Kb, r0 = line.split(None, 4)
            aclasspair = ( int(atomclass1), int(atomclass2) )
            classpair__Kb[ aclasspair ] = float(Kb)
            classpair__r0[ aclasspair ] = float(r0)
            
        lines_position_3 = lines_ffaffurr.index('      ##  Improper Torsional Parameters  ##\n')
        for line in lines_ffaffurr[lines_position_2+5:lines_position_3-4]:
            stringa, atomclass1, atomclass2, atomclass3, Ktheta, theta0 = line.split(None, 5)
            aclasstriple = ( int(atomclass1), int(atomclass2), int(atomclass3) )
            classtriple__Ktheta[ aclasstriple ] = float(Ktheta)
            classtriple__theta0[ aclasstriple ] = float(theta0)
            
        lines_position_4 = lines_ffaffurr.index('      ##  Torsional Parameters  ##\n')
        for line in lines_ffaffurr[lines_position_3+5:lines_position_4-4]:
            stringa, atomclass1, atomclass2, atomclass3, atomclass4, impV2, phase, periodicity = line.split(None, 7)
            classquadruple__impV2[ ( int(atomclass1), int(atomclass2), int(atomclass3),int(atomclass4) ) ] = float(impV2)
            
        lines_position_5 = lines_ffaffurr.index('      ##  Atomic Partial Charge Parameters  ##\n')
        for line in lines_ffaffurr[lines_position_4+5:lines_position_5-4]:
            stringa, atomclass1, atomclass2, atomclass3, atomclass4, Va, phasestringa1, phasestringa2, Vb, phasestringb1, phasestringb2, Vc, phasestringc1, phasestringc2 = line.split(None, 13)
            if float(Va) != 0:
                classquadruple__V1[ ( int(atomclass1), int(atomclass2), int(atomclass3), int(atomclass4) ) ] = float(Va)
            if float(Vb) != 0:    
                classquadruple__V2[ ( int(atomclass1), int(atomclass2), int(atomclass3), int(atomclass4) ) ] = float(Vb)
            if float(Vc) != 0:    
                classquadruple__V3[ ( int(atomclass1), int(atomclass2), int(atomclass3), int(atomclass4) ) ] = float(Vc)

    return(n_atom__type, \
            n_atom__class, \
            type__class, \
            class__symbol, \
            n_atom__atomic, \
            n_atom__mass, \
            n_atom__valence, \
            class__atomic, \
            class__mass, \
            class__valence, \
            n_atom__description, \
            type__description, \
            type__charge, \
            type__sigma, \
            type__epsilon, \
            classpair__Kb, \
            classpair__r0, \
            classtriple__Ktheta, \
            classtriple__theta0, \
            list_123_interacts, \
            classquadruple__V1, \
            classquadruple__V2, \
            classquadruple__V3, \
            list_1234_interacts, \
            classquadruple__impV2, \
            list_imp1234_interacts)


############################################################
# get 1-2-interactions, ..., 1-5-interactions between pairs of atoms
#   --> get interactions from TINKER provided 'ffaffurr.input.interactionsFF'
#   --> run '/path/to/tinker/bin/directory/analyze -k key <txyz-file> C ALL > ffaffurr.input.interactionsFF'
#   here ONLY the file ffaffurr.input.interactionsFF is used
#   (TINKER itself is NOT called)
############################################################

def get_FFinteractions():

    # check if file is there
    if not os.path.exists('ffaffurr.input.interactionsFF'):
        sys.exit('== Error: Input file \'ffaffurr.input.interactionsFF\' does not exist. Exiting now...')
    else:
        print('Now reading from input file \'ffaffurr.input.interactionsFF\'...')

    # read file
    in_file = open("ffaffurr.input.interactionsFF", 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    n_12_interacts = get_n_FFinteractions(lines, 'Number of 1-2 Pairs')
    print('    Number of 1-2 Pairs:        '+str(n_12_interacts))
    if not n_12_interacts > -1:
        sys.exit('== Error: Number of 1-2 Pairs should be a positive number. Exiting now...')

    list_12_interacts = get_list_FFinteractions(lines, ' List of 1-2 Connected Atomic Interactions :\n', n_12_interacts)
    if not len(list_12_interacts) == n_12_interacts:
        sys.exit('== Error: Sanity check - Number of 1-2 Pairs not correct. Exiting now...')

    n_13_interacts = get_n_FFinteractions(lines, 'Number of 1-3 Pairs')
    print('    Number of 1-3 Pairs:        '+str(n_13_interacts))
    if not n_13_interacts > -1:
        sys.exit('== Error: Number of 1-3 Pairs should be a positive number. Exiting now...')

    list_13_interacts = get_list_FFinteractions(lines, ' List of 1-3 Connected Atomic Interactions :\n', n_13_interacts)
    if not len(list_13_interacts) == n_13_interacts:
        sys.exit('== Error: Sanity check - Number of 1-3 Pairs not correct. Exiting now...')

    n_14_interacts = get_n_FFinteractions(lines, 'Number of 1-4 Pairs')
    print('    Number of 1-4 Pairs:        '+str(n_14_interacts))
    if not n_14_interacts > -1:
        sys.exit('== Error: Number of 1-4 Pairs should be a positive number. Exiting now...')

    list_14_interacts = get_list_FFinteractions(lines, ' List of 1-4 Connected Atomic Interactions :\n', n_14_interacts)
    if not len(list_14_interacts) == n_14_interacts:
        sys.exit('== Error: Sanity check - Number of 1-4 Pairs not correct. Exiting now...')

    n_15_interacts = get_n_FFinteractions(lines, 'Number of 1-5 Pairs')
    print('    Number of 1-5 Pairs:        '+str(n_15_interacts))
    if not n_15_interacts > -1:
        sys.exit('== Error: Number of 1-5 Pairs should be a positive number. Exiting now...')

    list_15_interacts = get_list_FFinteractions(lines, ' List of 1-5 Connected Atomic Interactions :\n', n_15_interacts)
    if not len(list_15_interacts) == n_15_interacts:
        sys.exit('== Error: Sanity check - Number of 1-5 Pairs not correct. Exiting now...')

    print('\n====\n')

    return(list_12_interacts,\
            list_13_interacts,\
            list_14_interacts,\
            list_15_interacts)

def get_n_FFinteractions(lines, key):
    number = -1
    for line in lines:
        if key in line:
            number = int(line.rsplit(None, 1)[-1])
            break
    if number == -1:
        sys.exit('== Error: keyword \''+key+'\' not found in \'ffaffurr.input.interactionsFF\'! Exiting now...')
    return(number)

def get_list_FFinteractions(lines, key, n_interacts):
    number1 = -1 # atom1
    number2 = -1 # atom2
    lines_position = lines.index(key)
    list_interacts = []
    for line in lines[lines_position+2:lines_position+2+n_interacts]:
        number1, number2 = line.split(None, 1)
        # int(number1) <= int(number2) always (property of TINKER)
        atuple = (int(number1), int(number2))
        list_interacts.append( atuple )

    if ( number1 == -1 ) or ( number2 == -1 ):
        sys.exit('== Error: keyword \''+key+'\' not found in \'ffaffurr.input.interactionsFF\'! Exiting now...')

    return(list_interacts)


############################################################
# get pair-wise charge parameters for Coulomb interactions
############################################################

def get_charge_pairs_params(type__charge):

    pairs__charge = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):
                pairs__charge[ (i+1,j+1) ] = type__charge[ n_atom__type[i+1] ] * type__charge[ n_atom__type[j+1] ]

    return(pairs__charge)


############################################################
# get (type)pair-wise vdW parameters (sigma & epsilon) for original FF
#   - original FF --> geometric mean for sigma & epsilon
############################################################

def get_origFF_vdW_pairs_params(type__sigma, type__epsilon):

    pairs__sigma = {}
    pairs__epsilon = {}
    typepairs__sigma = {}
    typepairs__epsilon = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):
                if dict_keywords['readparamsfromffaffurr'] == False:
                    pairs__sigma[ (i+1,j+1) ] = math.sqrt( type__sigma[ n_atom__type[i+1] ] * type__sigma[ n_atom__type[j+1] ] )
                    pairs__epsilon[ (i+1,j+1) ] = math.sqrt( type__epsilon[ n_atom__type[i+1] ] * type__epsilon[ n_atom__type[j+1] ] )
                    if n_atom__type[i+1] <= n_atom__type[j+1]:
                        typepairs__sigma[ ( n_atom__type[i+1],n_atom__type[j+1] ) ] = pairs__sigma[ (i+1,j+1) ]
                        typepairs__epsilon[ ( n_atom__type[i+1],n_atom__type[j+1] ) ] = pairs__epsilon[ (i+1,j+1) ]
                    else:
                        typepairs__sigma[ ( n_atom__type[j+1],n_atom__type[i+1] ) ] = pairs__sigma[ (i+1,j+1) ]
                        typepairs__epsilon[ ( n_atom__type[j+1],n_atom__type[i+1] ) ] = pairs__epsilon[ (i+1,j+1) ]
                        
                elif dict_keywords['readparamsfromffaffurr'] == True:
					# check if file is there
                    if not os.path.exists('oplsaa-ffaffurr.prm'):
                        sys.exit('== Error: Input file \'oplsaa-ffaffurr.prm\' does not exist. Exiting now...')
                   
                    
                    # read file
                    in_file_ffaffurr = open("oplsaa-ffaffurr.prm", 'r')
                    file_lines_ffaffurr = in_file_ffaffurr.readlines()
                    lines_ffaffurr = list(file_lines_ffaffurr)
                    in_file_ffaffurr.close()
                    
                    lines_position_1 = lines_ffaffurr.index('      ##  Van der Waals Pair Parameters  ##\n')
                    lines_position_2 = lines_ffaffurr.index('      ##  Bond Stretching Parameters  ##\n')
                    for line in lines_ffaffurr[lines_position_1+5:lines_position_2-4]:
                        stringa, atomtype1, atomtype2, pairsigma, pairepsilon = line.split(None, 4)
                        typepairs__sigma[ ( int(atomtype1), int(atomtype2) ) ] = float(pairsigma)
                        typepairs__epsilon[ ( int(atomtype1), int(atomtype2) ) ] = float(pairepsilon)
                        
                    if n_atom__type[i+1] <= n_atom__type[j+1]:
                        pairs__sigma[ (i+1,j+1) ] = typepairs__sigma[ ( n_atom__type[i+1],n_atom__type[j+1] ) ]
                        pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[i+1],n_atom__type[j+1] ) ]					
                    else:
                        pairs__sigma[ (i+1,j+1) ] = typepairs__sigma[ ( n_atom__type[j+1],n_atom__type[i+1] ) ]
                        pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[j+1],n_atom__type[i+1] ) ]


    return(pairs__sigma, \
            pairs__epsilon, \
            typepairs__sigma, \
            typepairs__epsilon)


############################################################
# test function only used for cross-ckecking with TINKER
# --> all energy contributions (bonds, angles, torsions, impropers, vdW, Coulomb)
#     are calculated using a file called 'tinker.xyz'
#     --> energies can then be cross-checked with TINKER
############################################################

def cross_check_tinker(classpair__Kb, \
                        classpair__r0, \
                        classtriple__Ktheta, \
                        classtriple__theta0, \
                        classquadruple__V1, \
                        classquadruple__V2, \
                        classquadruple__V3, \
                        classquadruple__impV2, \
                        pairs__sigma, \
                        pairs__epsilon, \
                        pairs__charge):

    print('Cross-checking all energy contributions on a provided file called \'tinker.xyz\'.')

    # get xyz from a tinker-xyz file called 'tinker.xyz'
    n_atom__xyz = get_tinker_xyz()

    # get distances between any two atoms
    pairs__distances = get_distances(n_atom__xyz)

    # get angles for any 123 interactions
    triples__angles = get_angles(n_atom__xyz)

    # get torsional (dihedral) angles for any 1234 interactions
    quadruples__torsions = get_torsions(n_atom__xyz)

    # get improper torsional angles for any improper 1234 interactions
    quadruples__improps = get_improps(n_atom__xyz)

    # get bonding energy
    Ebonds = get_bonding_energy(pairs__distances, classpair__Kb, classpair__r0)
    printf("  Bond Stretching:       %9.4f\n", Ebonds)

    # get angle bending energy
    Eangles = get_angles_energy(triples__angles, classtriple__Ktheta, classtriple__theta0)
    printf("    Angle Bending:       %9.4f\n", Eangles)

    # get torsions bending energy
    Etorsions = get_torsions_energy(quadruples__torsions, \
                                     classquadruple__V1, classquadruple__V2, classquadruple__V3)
    printf("  Torsional Angle:       %9.4f\n", Etorsions)

    # get improper torsions energy
    Eimprops = get_improps_energy(quadruples__improps, classquadruple__impV2)
    printf(" Improper Torsion:       %9.4f\n", Eimprops)

    # get vdW energy
    Evdw = get_vdW_energy(pairs__distances, pairs__sigma, pairs__epsilon)
    printf("    Van der Waals:       %9.4f\n", Evdw)

    # get Coulomb energy
    Ecoul = get_Coulomb_energy(pairs__distances, pairs__charge)
    printf("    Charge-Charge:       %9.4f\n", Ecoul)

    print('\n====\n')

    return()


############################################################
# get xyz from one tinker-xyz file called 'tinker.xyz'
############################################################

def get_tinker_xyz():

    # check if file is there
    if not os.path.exists('tinker.xyz'):
        sys.exit('== Error: Input file \'tinker.xyz\' does not exist. Exiting now...')
    else:
        print('Now reading from input file \'tinker.xyz\'...\n')

    # read file
    in_file = open("tinker.xyz", 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    n_atom__xyz = {}

    for line in lines[1:n_atoms+1]:
        n_atom, symbol, x, y, z, stuff = line.split(None, 5)
        n_atom__xyz[int(n_atom)] = [float(x), float(y), float(z)]

    return(n_atom__xyz)


############################################################
# get distances between any two atoms
############################################################

def get_distances(n_atom__xyz):

    pairs__distances = {}

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = math.sqrt(   ( n_atom__xyz[i+1][0] - n_atom__xyz[j+1][0] ) ** 2. \
                                  + ( n_atom__xyz[i+1][1] - n_atom__xyz[j+1][1] ) ** 2. \
                                  + ( n_atom__xyz[i+1][2] - n_atom__xyz[j+1][2] ) ** 2. )
            pairs__distances[ (i+1,j+1) ] = distance

    return(pairs__distances)


############################################################
# get angles for any 123 interactions
############################################################

def get_angles(n_atom__xyz):

    triples__angles = {}

    for atriple in list_123_interacts:
        xyz1 = n_atom__xyz[atriple[0]]
        xyz2 = n_atom__xyz[atriple[1]]
        xyz3 = n_atom__xyz[atriple[2]]
        theta = get_theta( xyz1, xyz2, xyz3 )
        triples__angles[ atriple ] = theta

    return(triples__angles)


############################################################
# get angle (theta) for a specific 123 interaction
############################################################

def get_theta( xyz1, xyz2, xyz3 ):

    x1 = xyz1[0]
    y1 = xyz1[1]
    z1 = xyz1[2]
    x2 = xyz2[0]
    y2 = xyz2[1]
    z2 = xyz2[2]
    x3 = xyz3[0]
    y3 = xyz3[1]
    z3 = xyz3[2]

    x12 = x1 - x2
    y12 = y1 - y2
    z12 = z1 - z2
    x32 = x3 - x2
    y32 = y3 - y2
    z32 = z3 - z2

    r12 = math.sqrt( x12**2. + y12**2. + z12**2. )
    r32 = math.sqrt( x32**2. + y32**2. + z32**2. )

    scalarprod = x12*x32 + y12*y32 + z12*z32

    costheta = scalarprod / ( r12*r32 )

    costheta = min(1.,max(-1.,costheta))

    theta = (180./math.pi) * math.acos(costheta)

    return(theta)


############################################################
# get torsional angles for any 1234 interactions
############################################################

def get_torsions(n_atom__xyz):

    quadruples__torsions = {}

    for aquadruple in list_1234_interacts:
        xyz1 = n_atom__xyz[aquadruple[0]]
        xyz2 = n_atom__xyz[aquadruple[1]]
        xyz3 = n_atom__xyz[aquadruple[2]]
        xyz4 = n_atom__xyz[aquadruple[3]]
        phi = get_phi( xyz1, xyz2, xyz3, xyz4 )
        quadruples__torsions[ aquadruple ] = phi

    return(quadruples__torsions)


############################################################
# get improper torsional angles for any improper 1234 interactions
############################################################

def get_improps(n_atom__xyz):

    quadruples__improps = {}

    for aquadruple in list_imp1234_interacts:
        xyz1 = n_atom__xyz[aquadruple[0]]
        xyz2 = n_atom__xyz[aquadruple[1]]
        xyz3 = n_atom__xyz[aquadruple[2]]
        xyz4 = n_atom__xyz[aquadruple[3]]
        phi = get_phi( xyz1, xyz2, xyz3, xyz4 )
        quadruples__improps[ aquadruple ] = phi

    return(quadruples__improps)


############################################################
# get torsional angle (phi) for a specific 1234 interaction
############################################################

def get_phi( xyz1, xyz2, xyz3, xyz4 ):

    x1 = xyz1[0]
    y1 = xyz1[1]
    z1 = xyz1[2]
    x2 = xyz2[0]
    y2 = xyz2[1]
    z2 = xyz2[2]
    x3 = xyz3[0]
    y3 = xyz3[1]
    z3 = xyz3[2]
    x4 = xyz4[0]
    y4 = xyz4[1]
    z4 = xyz4[2]

    x21 = x2 - x1
    y21 = y2 - y1
    z21 = z2 - z1
    x32 = x3 - x2
    y32 = y3 - y2
    z32 = z3 - z2
    x43 = x4 - x3
    y43 = y4 - y3
    z43 = z4 - z3

    x123 = y21*z32 - y32*z21
    y123 = z21*x32 - z32*x21
    z123 = x21*y32 - x32*y21
    x234 = y32*z43 - y43*z32
    y234 = z32*x43 - z43*x32
    z234 = x32*y43 - x43*y32

    x1234 = y123*z234 - y234*z123
    y1234 = z123*x234 - z234*x123
    z1234 = x123*y234 - x234*y123

    r123 = math.sqrt( x123**2. + y123**2. + z123**2. )
    r234 = math.sqrt( x234**2. + y234**2. + z234**2. )

    r32 = math.sqrt( x32**2. + y32**2. + z32**2. )

    cosphi = ( x123*x234 + y123*y234 + z123*z234 ) / ( r123*r234 )

    cosphi = min(1.,max(-1.,cosphi))

    phi = (180./math.pi) * math.acos(cosphi)

    sinphi = ( x32*x1234 + y32*y1234 + z32*z1234 ) / ( r32*r123*r234 )

    if sinphi < 0.: phi = -phi

    return(phi)


############################################################
# get bonding energy
############################################################

def get_bonding_energy(pairs__distances, classpair__Kb, classpair__r0):

    Ebonds = 0.

    for pair in list_12_interacts:

        if n_atom__class[pair[0]] <= n_atom__class[pair[1]]:
            aclasspair = ( n_atom__class[pair[0]],n_atom__class[pair[1]] )
        else:
            aclasspair = ( n_atom__class[pair[1]],n_atom__class[pair[0]] )

        Kb = classpair__Kb[ aclasspair ] 
        r0 = classpair__r0[ aclasspair ] 

        r = pairs__distances[pair]

        Ebonds += Kb * (r-r0)**2.

    return(Ebonds)


############################################################
# get angles bending energy
############################################################

def get_angles_energy(triples__angles, classtriple__Ktheta, classtriple__theta0):

    Eangles = 0.

    for triple in list_123_interacts:

        if n_atom__class[triple[0]] <= n_atom__class[triple[2]]:
            aclasstriple = ( n_atom__class[triple[0]],n_atom__class[triple[1]],n_atom__class[triple[2]] )
        else:
            aclasstriple = ( n_atom__class[triple[2]],n_atom__class[triple[1]],n_atom__class[triple[0]] )

        Ktheta = classtriple__Ktheta[ aclasstriple ]
        theta0 = classtriple__theta0[ aclasstriple ]

        theta = triples__angles[triple]

        Eangles += Ktheta * (math.pi/180.)**2. * (theta-theta0)**2.

    return(Eangles)


############################################################
# get torsions bending energy
############################################################

def get_torsions_energy(quadruples__torsions, \
                         classquadruple__V1, classquadruple__V2, classquadruple__V3):

    Etorsions = 0.

    for quadruple in list_1234_interacts:

        phi = quadruples__torsions[quadruple]

        if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
            atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
        else:
            atuple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )

        if atuple in classquadruple__V1:
            V1 = classquadruple__V1[atuple]
            Etorsions += V1 * 0.5 * ( 1. + math.cos(      ( math.pi/180. ) * phi ) )
        if atuple in classquadruple__V2:
            V2 = classquadruple__V2[atuple]
            Etorsions += V2 * 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )
        if atuple in classquadruple__V3:
            V3 = classquadruple__V3[atuple]
            Etorsions += V3 * 0.5 * ( 1. + math.cos( 3. * ( math.pi/180. ) * phi ) )

    return(Etorsions)


############################################################
# get improper torsions energy
############################################################

def get_improps_energy(quadruples__improps, classquadruple__impV2):

    Eimprops = 0.

    for quadruple in list_imp1234_interacts:

        phi = quadruples__improps[quadruple]

        atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )

        impV2 = classquadruple__impV2[atuple]

        Eimprops += impV2 * 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )

    return(Eimprops)


############################################################
# get vdW energy
############################################################

def get_vdW_energy(pairs__distances, pairs__sigma, pairs__epsilon):

    Evdw = 0.

    for pair,sigma in pairs__sigma.items():
        if pair in list_14_interacts:
            f = 0.5
        else:
            f = 1.
        r = pairs__distances[pair]
        epsilon = pairs__epsilon[pair]
        Evdw += 4. * epsilon * f * ( (sigma/r)**12. - (sigma/r)**6. )

    return(Evdw)


############################################################
# get Coulomb energy
############################################################

def get_Coulomb_energy(pairs__distances, pairs__charge):

    qqr2kcalpermole = 332.063714

    Ecoul = 0.

    for pair,qq in pairs__charge.items():
        if pair in list_14_interacts:
            f = 0.5
        else:
            f = 1.
        r = pairs__distances[pair]
        Ecoul += f * qqr2kcalpermole * qq / r

    return(Ecoul)


############################################################
# get logfiles (FHI-aims output files)
# from input file 'ffaffurr.input.FHI-aims-logfiles'
############################################################

def get_logfiles():

    list_logfiles = []

    # check if file is there
    if not os.path.exists('ffaffurr.input.FHI-aims-logfiles'):
        sys.exit('== Error: Input file \'ffaffurr.input.FHI-aims-logfiles\' does not exist. Exiting now...')
    else:
        print('Now reading from input file \'ffaffurr.input.FHI-aims-logfiles\'...')

    # read file
    in_file = open("ffaffurr.input.FHI-aims-logfiles", 'r')
    list_logfiles = in_file.read().splitlines()
    in_file.close()

    if not list_logfiles:
        sys.exit('== Error: No FHI-aims logfiles in \'ffaffurr.input.FHI-aims-logfiles\'. Exiting now...')

    # check if logfile exists
    for logfile in list_logfiles:
        if not os.path.exists(logfile):
            sys.exit('== Error: Input logfile '+logfile+' does not exist. Exiting now...')

    print('\n====\n')
    
    return(list_logfiles)


############################################################
# get xyz for all logfiles
############################################################

def get_logfiles_xyz():

    listofdicts_logfiles___n_atom__xyz = []

    for logfile in list_logfiles:
        n_atom__xyz = get_fhiaims_xyz(logfile)
        listofdicts_logfiles___n_atom__xyz.append( n_atom__xyz )

    return(listofdicts_logfiles___n_atom__xyz)


############################################################
# get xyz for a specific FHI-aims logfile
############################################################

def get_fhiaims_xyz(logfile):

    # read file
    in_file = open(logfile, 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    n_atom__xyz = {}

    lines_position = lines.index('  Input geometry:\n')

    for line in lines[lines_position+4:lines_position+4+n_atoms]:
        stringa, n_atom, stringb, symbol, x, y, z = line.split(None, 6)
        n_atom__xyz[int(n_atom.rstrip(":"))] = [float(x), float(y), float(z)]

    return(n_atom__xyz)


############################################################
# fine-tune r0 by averaging distances over all FHI-aims input logfiles
############################################################

def get_average_r0():

    classpair__r0collected = {}

    for pair in list_12_interacts:

        if n_atom__class[pair[0]] <= n_atom__class[pair[1]]:
            aclasspair = ( n_atom__class[pair[0]],n_atom__class[pair[1]] )
        else:
            aclasspair = ( n_atom__class[pair[1]],n_atom__class[pair[0]] )

        classpair__r0collected[ aclasspair ] = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        for pair in list_12_interacts:

            if n_atom__class[pair[0]] <= n_atom__class[pair[1]]:
                aclasspair = ( n_atom__class[pair[0]],n_atom__class[pair[1]] )
            else:
                aclasspair = ( n_atom__class[pair[1]],n_atom__class[pair[0]] )

            classpair__r0collected[ aclasspair ].append( pairs__distances[pair] )

    classpair__r0averaged = {}

    for classpair in classpair__r0collected:
        classpair__r0averaged[ classpair ] = numpy.mean(classpair__r0collected[ classpair ])

    return(classpair__r0averaged)


############################################################
# fine-tune theta0 by averaging angles over all FHI-aims input logfiles
############################################################

def get_average_theta0():

    classtriple__theta0collected = {}

    for triple in list_123_interacts:

        if n_atom__class[triple[0]] <= n_atom__class[triple[2]]:
            aclasstriple = ( n_atom__class[triple[0]],n_atom__class[triple[1]],n_atom__class[triple[2]] )
        else:
            aclasstriple = ( n_atom__class[triple[2]],n_atom__class[triple[1]],n_atom__class[triple[0]] )

        classtriple__theta0collected[ aclasstriple ] = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        triples__angles = get_angles(n_atom__xyz)

        for triple in list_123_interacts:

            if n_atom__class[triple[0]] <= n_atom__class[triple[2]]:
                aclasstriple = ( n_atom__class[triple[0]],n_atom__class[triple[1]],n_atom__class[triple[2]] )
            else:
                aclasstriple = ( n_atom__class[triple[2]],n_atom__class[triple[1]],n_atom__class[triple[0]] )

            classtriple__theta0collected[ aclasstriple ].append( triples__angles[triple] )

    classtriple__theta0averaged = {}

    for classtriple in classtriple__theta0collected:
        classtriple__theta0averaged[ classtriple ] = numpy.mean(classtriple__theta0collected[ classtriple ])

    return(classtriple__theta0averaged)


############################################################
# fine-tune partial charges by averaging charges over all FHI-aims input logfiles
############################################################

def get_average_charge():

    sortedlist_types = []
    for n_atom,atype in n_atom__type.items():
        sortedlist_types.append( atype )
    sortedlist_types = sorted(set(sortedlist_types))

    type__collectedCharges = {}
    for atype in sortedlist_types:
        type__collectedCharges[atype] = []

    for logfile in list_logfiles:
        n_atom__charges = get_fhiaims_charges(logfile)
        for i in range(n_atoms):
            type__collectedCharges[ n_atom__type[i+1] ].append( n_atom__charges[i+1] )

    type__averageCharge = {}
    for atype in sortedlist_types:
        type__averageCharge[atype] = numpy.mean(type__collectedCharges[ atype ])

    return(type__averageCharge)


############################################################
# get charges of every atom for a specific FHI-aims logfile
############################################################

def get_fhiaims_charges(logfile):

    n_atom__charges = {}
    list_charges = []
    
    if dict_keywords['fine_tune_charge'] == 'RESP':
        little_path = logfile.rsplit('/',1)[0]
        
        #read respfile
        with open(os.path.join(little_path, 'resp.chrg'), 'r') as respfile:
            resp_lines = respfile.readlines()
            
        for i in resp_lines:
            list_charges.append(float(i))
    else:        
        # read logfile
        in_file = open(logfile, 'r')
        file_lines = in_file.readlines()
        lines = list(file_lines)
        in_file.close()
        
        
        if dict_keywords['fine_tune_charge'] == 'Hirshfeld':
            string = '|   Hirshfeld charge        :'
        elif dict_keywords['fine_tune_charge'] == 'ESP':
            string = 'ESP charge:'
        
        list_charges = []
        for line in lines:
            if string in line:
                charge = float(line.rsplit(None, 1)[-1])
                list_charges.append( charge )
        
        # FIXBUG in FHI-aims
        # ESP charges are of opposite charge in some FHI-aims versions, please check
#        if dict_keywords['fine_tune_charge'] == 'ESP': list_charges = [ -x for x in list_charges ]
        
        if len(list_charges) == 0:
            sys.exit('== Error: No charge information found in '+logfile+'. Exiting now...')
        elif len(list_charges) > n_atoms:
#            sys.exit('== Error in '+logfile+': Too much charge information found. Single point energy calculations only! Exiting now...')
            list_charges = list_charges[-n_atoms:]
#        else:
    for i in range(n_atoms):
        n_atom__charges[i+1] = list_charges[i]

    return(n_atom__charges)


############################################################
# fine-tune and get (type)pair-wise sigmas from TS (using R0eff)
############################################################

def get_sigmas_TS():

    sortedlist_types = []
    for n_atom,atype in n_atom__type.items():
        sortedlist_types.append( atype )
    sortedlist_types = sorted(set(sortedlist_types))

    type__collectedR0eff = {}
    for atype in sortedlist_types:
        type__collectedR0eff[atype] = []

    # get R0free atom-wise
    n_atom__R0free = get_R0free()

    for logfile in list_logfiles:

        # get free atom volumes & Hirshfeld volumes atom-wise
        n_atom__free_atom_volume, n_atom__hirshfeld_volume = get_fhiaims_hirshfeld_volume(logfile)

        # calculate R0eff atom-wise and collect type-wise
        for i in range(n_atoms):
            R0eff = ( ( n_atom__hirshfeld_volume[i+1] / n_atom__free_atom_volume[i+1] )**(1./3.) ) * n_atom__R0free[i+1]
            type__collectedR0eff[ n_atom__type[i+1] ].append( R0eff )

    type__averageR0eff = {}

    # average R0eff
    for atype in sortedlist_types:
        type__averageR0eff[atype] = numpy.mean(type__collectedR0eff[ atype ])

    pairs__sigma = {}
    typepairs__sigma = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):

                pairs__sigma[ (i+1,j+1) ] = ( type__averageR0eff[ n_atom__type[i+1] ] + type__averageR0eff[ n_atom__type[j+1] ] ) * (2.**(-1./6.))

                if n_atom__type[i+1] <= n_atom__type[j+1]:
                    typepairs__sigma[ ( n_atom__type[i+1],n_atom__type[j+1] ) ] = pairs__sigma[ (i+1,j+1) ]
                else:
                    typepairs__sigma[ ( n_atom__type[j+1],n_atom__type[i+1] ) ] = pairs__sigma[ (i+1,j+1) ]

    return(pairs__sigma, \
            typepairs__sigma)


############################################################
# get R0free atom-wise
############################################################

def get_R0free():

    n_atom__R0free = {}

    for i in range(n_atoms):
        if n_atom__atomic[i+1] == 1:
            n_atom__R0free[i+1] = 3.1000/2.
        elif n_atom__atomic[i+1] == 2:     
            n_atom__R0free[i+1] = 2.6500/2.
        elif n_atom__atomic[i+1] == 3:     
            n_atom__R0free[i+1] = 4.1600/2.
        elif n_atom__atomic[i+1] == 4:     
            n_atom__R0free[i+1] = 4.1700/2.
        elif n_atom__atomic[i+1] == 5:     
            n_atom__R0free[i+1] = 3.8900/2.
        elif n_atom__atomic[i+1] == 6:     
            n_atom__R0free[i+1] = 3.5900/2.
        elif n_atom__atomic[i+1] == 7:     
            n_atom__R0free[i+1] = 3.3400/2.
        elif n_atom__atomic[i+1] == 8:     
            n_atom__R0free[i+1] = 3.1900/2.
        elif n_atom__atomic[i+1] == 9:     
            n_atom__R0free[i+1] = 3.0400/2.
        elif n_atom__atomic[i+1] == 10:    
            n_atom__R0free[i+1] = 2.9100/2.
        elif n_atom__atomic[i+1] == 11:    
            n_atom__R0free[i+1] = 3.7300/2.
        elif n_atom__atomic[i+1] == 12:    
            n_atom__R0free[i+1] = 4.2700/2.
        elif n_atom__atomic[i+1] == 13:    
            n_atom__R0free[i+1] = 4.3300/2.
        elif n_atom__atomic[i+1] == 14:    
            n_atom__R0free[i+1] = 4.2000/2.
        elif n_atom__atomic[i+1] == 15:    
            n_atom__R0free[i+1] = 4.0100/2.
        elif n_atom__atomic[i+1] == 16:    
            n_atom__R0free[i+1] = 3.8600/2.
        elif n_atom__atomic[i+1] == 17:    
            n_atom__R0free[i+1] = 3.7100/2.
        elif n_atom__atomic[i+1] == 18:    
            n_atom__R0free[i+1] = 3.5500/2.
        elif n_atom__atomic[i+1] == 19:    
            n_atom__R0free[i+1] = 3.7100/2.
        elif n_atom__atomic[i+1] == 20:    
            n_atom__R0free[i+1] = 4.6500/2.
        elif n_atom__atomic[i+1] == 21:    
            n_atom__R0free[i+1] = 4.5900/2.
        elif n_atom__atomic[i+1] == 22:    
            n_atom__R0free[i+1] = 4.5100/2.
        elif n_atom__atomic[i+1] == 23:    
            n_atom__R0free[i+1] = 4.4400/2.
        elif n_atom__atomic[i+1] == 24:    
            n_atom__R0free[i+1] = 3.9900/2.
        elif n_atom__atomic[i+1] == 25:    
            n_atom__R0free[i+1] = 3.9700/2.
        elif n_atom__atomic[i+1] == 26:    
            n_atom__R0free[i+1] = 4.2300/2.
        elif n_atom__atomic[i+1] == 27:    
            n_atom__R0free[i+1] = 4.1800/2.
        elif n_atom__atomic[i+1] == 28:    
            n_atom__R0free[i+1] = 3.8200/2.
        elif n_atom__atomic[i+1] == 29:    
            n_atom__R0free[i+1] = 3.7600/2.
        elif n_atom__atomic[i+1] == 30:    
            n_atom__R0free[i+1] = 4.0200/2.
        elif n_atom__atomic[i+1] == 31:    
            n_atom__R0free[i+1] = 4.1900/2.
        elif n_atom__atomic[i+1] == 32:    
            n_atom__R0free[i+1] = 4.2000/2.
        elif n_atom__atomic[i+1] == 33:    
            n_atom__R0free[i+1] = 4.1100/2.
        elif n_atom__atomic[i+1] == 34:    
            n_atom__R0free[i+1] = 4.0400/2.
        elif n_atom__atomic[i+1] == 35:    
            n_atom__R0free[i+1] = 3.9300/2.
        elif n_atom__atomic[i+1] == 36:    
            n_atom__R0free[i+1] = 3.8200/2.
        else:
            sys.exit('== Error: Current implementation of free vdW radii only for elements up to Kr. Exiting now...')

    return(n_atom__R0free)


############################################################
# get free atom volumes & Hirshfeld volumes atom-wise for a specific FHI-aims logfile
############################################################

def get_fhiaims_hirshfeld_volume(logfile):

    # read file
    in_file = open(logfile, 'r')
    file_lines = in_file.readlines()
    lines = list(file_lines)
    in_file.close()

    n_atom__free_atom_volume = {}
    n_atom__hirshfeld_volume = {}

    alist_free_atom_volumes = []
    for line in lines:
        if '|   Free atom volume        :' in line:
            afree_atom_volume = float(line.rsplit(None, 1)[-1])
            alist_free_atom_volumes.append( afree_atom_volume )

    alist_hirshfeld_volumes = []
    for line in lines:
        if '|   Hirshfeld volume        :' in line:
            ahirshfeld_volume = float(line.rsplit(None, 1)[-1])
            alist_hirshfeld_volumes.append( ahirshfeld_volume )

    if ( len(alist_free_atom_volumes) == 0 ) or ( len(alist_hirshfeld_volumes) == 0 ):
        sys.exit('== Error: No hirshfeld volume information found in '+logfile+'. Exiting now...')
    elif ( len(alist_free_atom_volumes) > n_atoms ) or ( len(alist_hirshfeld_volumes) > n_atoms ):
        alist_free_atom_volumes = alist_free_atom_volumes[-n_atoms:]
        alist_hirshfeld_volumes = alist_hirshfeld_volumes[-n_atoms:]
        #sys.exit('== Error in '+logfile+': Too many hirshfeld volume information found. Single-point energy calculations only! Exiting now...')
    #else:
    for i in range(n_atoms):
        n_atom__free_atom_volume[i+1] = alist_free_atom_volumes[i]
        n_atom__hirshfeld_volume[i+1] = alist_hirshfeld_volumes[i]

    return(n_atom__free_atom_volume, n_atom__hirshfeld_volume)


############################################################
# this function returns dict2, but with values set to zero that are also zero in dict1
############################################################

def set_some_dict_entries_zero(dict1, dict2):

    small = 0.00001

    for key,value in dict1.items():
        if ( value < small ):
            dict2[key] = 0.

    return(dict2)


############################################################
# fine-tune and get (type)pair-wise epsilons from TS (using R0eff, C6eff, alpha0eff)
############################################################

def get_epsilons_TS():

    sortedlist_types = []
    for n_atom,atype in n_atom__type.items():
        sortedlist_types.append( atype )
    sortedlist_types = sorted(set(sortedlist_types))

    type__collectedR0eff = {}
    type__collectedC6eff = {}
    type__collectedAlpha0eff = {}
    for atype in sortedlist_types:
        type__collectedR0eff[atype] = []
        type__collectedC6eff[atype] = []
        type__collectedAlpha0eff[atype] = []

    # get R0free atom-wise
    n_atom__R0free = get_R0free()

    # get C6free, alpha0free atom-wise
    n_atom__C6free, n_atom__alpha0free = get_C6free_alpha0free()

    for logfile in list_logfiles:

        # get free atom volumes & Hirshfeld volumes atom-wise
        n_atom__free_atom_volume, n_atom__hirshfeld_volume = get_fhiaims_hirshfeld_volume(logfile)

        # calculate R0eff, C6eff, alpha0eff atom-wise and collect type-wise
        for i in range(n_atoms):

            R0eff = ( ( n_atom__hirshfeld_volume[i+1] / n_atom__free_atom_volume[i+1] )**(1./3.) ) * n_atom__R0free[i+1]
            type__collectedR0eff[ n_atom__type[i+1] ].append( R0eff )

            C6eff = ( n_atom__hirshfeld_volume[i+1] / n_atom__free_atom_volume[i+1] )**2. * n_atom__C6free[i+1]
            type__collectedC6eff[ n_atom__type[i+1] ].append( C6eff )

            alpha0eff = ( n_atom__hirshfeld_volume[i+1] / n_atom__free_atom_volume[i+1] ) * n_atom__alpha0free[i+1]
            type__collectedAlpha0eff[ n_atom__type[i+1] ].append( alpha0eff )

    type__averageR0eff = {}
    type__averageC6eff = {}
    type__averageAlpha0eff = {}

    # average R0eff, C6eff, alpha0eff
    for atype in sortedlist_types:
        type__averageR0eff[atype] = numpy.mean(type__collectedR0eff[ atype ])
        type__averageC6eff[atype] = numpy.mean(type__collectedC6eff[ atype ])
        type__averageAlpha0eff[atype] = numpy.mean(type__collectedAlpha0eff[ atype ])

    pairs__epsilon = {}
    typepairs__epsilon = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):

                sigma = ( type__averageR0eff[ n_atom__type[i+1] ] + type__averageR0eff[ n_atom__type[j+1] ] ) * (2.**(-1./6.))

                C6i = type__averageC6eff[ n_atom__type[i+1] ]
                C6j = type__averageC6eff[ n_atom__type[j+1] ]

                alpha0i = type__averageAlpha0eff[ n_atom__type[i+1] ]
                alpha0j = type__averageAlpha0eff[ n_atom__type[j+1] ]

                C6 = ( 2. * C6i * C6j ) / ( (alpha0j/alpha0i)*C6i + (alpha0i/alpha0j)*C6j )

                pairs__epsilon[ (i+1,j+1) ] = C6 / ( 4.* sigma**6. ) * 23.060542 # eV to kcal/mol

                if n_atom__type[i+1] <= n_atom__type[j+1]:
                    typepairs__epsilon[ ( n_atom__type[i+1],n_atom__type[j+1] ) ] = pairs__epsilon[ (i+1,j+1) ]
                else:
                    typepairs__epsilon[ ( n_atom__type[j+1],n_atom__type[i+1] ) ] = pairs__epsilon[ (i+1,j+1) ]

    return(pairs__epsilon, \
            typepairs__epsilon)


############################################################
# get C6free, alpha0free atom-wise
############################################################

def get_C6free_alpha0free():

    n_atom__C6free = {}
    n_atom__alpha0free = {}

    for i in range(n_atoms):
        if n_atom__atomic[i+1] == 1:
            n_atom__C6free[i+1] =    6.5000 ; n_atom__alpha0free[i+1] =   4.5000
        elif n_atom__atomic[i+1] == 2:                                                                            
            n_atom__C6free[i+1] =    1.4600 ; n_atom__alpha0free[i+1] =   1.3800
        elif n_atom__atomic[i+1] == 3:                                                                            
            n_atom__C6free[i+1] = 1387.0000 ; n_atom__alpha0free[i+1] = 164.2000
        elif n_atom__atomic[i+1] == 4:                                                                            
            n_atom__C6free[i+1] =  214.0000 ; n_atom__alpha0free[i+1] =  38.0000
        elif n_atom__atomic[i+1] == 5:                                                                            
            n_atom__C6free[i+1] =   99.5000 ; n_atom__alpha0free[i+1] =  21.0000
        elif n_atom__atomic[i+1] == 6:                                                                            
            n_atom__C6free[i+1] =   46.6000 ; n_atom__alpha0free[i+1] =  12.0000
        elif n_atom__atomic[i+1] == 7:                                                                            
            n_atom__C6free[i+1] =   24.2000 ; n_atom__alpha0free[i+1] =   7.4000
        elif n_atom__atomic[i+1] == 8:                                                                            
            n_atom__C6free[i+1] =   15.6000 ; n_atom__alpha0free[i+1] =   5.4000
        elif n_atom__atomic[i+1] == 9:                                                                            
            n_atom__C6free[i+1] =    9.5200 ; n_atom__alpha0free[i+1] =   3.8000
        elif n_atom__atomic[i+1] == 10:                                                                           
            n_atom__C6free[i+1] =    6.3800 ; n_atom__alpha0free[i+1] =   2.6700
        elif n_atom__atomic[i+1] == 11:                                                                           
            n_atom__C6free[i+1] = 1556.0000 ; n_atom__alpha0free[i+1] = 162.7000
        elif n_atom__atomic[i+1] == 12:                                                                           
            n_atom__C6free[i+1] =  627.0000 ; n_atom__alpha0free[i+1] =  71.0000
        elif n_atom__atomic[i+1] == 13:                                                                           
            n_atom__C6free[i+1] =  528.0000 ; n_atom__alpha0free[i+1] =  60.0000
        elif n_atom__atomic[i+1] == 14:                                                                           
            n_atom__C6free[i+1] =  305.0000 ; n_atom__alpha0free[i+1] =  37.0000
        elif n_atom__atomic[i+1] == 15:                                                                           
            n_atom__C6free[i+1] =  185.0000 ; n_atom__alpha0free[i+1] =  25.0000
        elif n_atom__atomic[i+1] == 16:                                                                           
            n_atom__C6free[i+1] =  134.0000 ; n_atom__alpha0free[i+1] =  19.6000
        elif n_atom__atomic[i+1] == 17:                                                                           
            n_atom__C6free[i+1] =   94.6000 ; n_atom__alpha0free[i+1] =  15.0000
        elif n_atom__atomic[i+1] == 18:                                                                           
            n_atom__C6free[i+1] =   64.3000 ; n_atom__alpha0free[i+1] =  11.1000
        elif n_atom__atomic[i+1] == 19:                                                                           
            n_atom__C6free[i+1] = 3897.0000 ; n_atom__alpha0free[i+1] = 292.9000
        elif n_atom__atomic[i+1] == 20:                                                                           
            n_atom__C6free[i+1] = 2221.0000 ; n_atom__alpha0free[i+1] = 160.0000
        elif n_atom__atomic[i+1] == 21:                                                                           
            n_atom__C6free[i+1] = 1383.0000 ; n_atom__alpha0free[i+1] = 120.0000
        elif n_atom__atomic[i+1] == 22:                                                                           
            n_atom__C6free[i+1] = 1044.0000 ; n_atom__alpha0free[i+1] =  98.0000
        elif n_atom__atomic[i+1] == 23:                                                                           
            n_atom__C6free[i+1] =  832.0000 ; n_atom__alpha0free[i+1] =  84.0000
        elif n_atom__atomic[i+1] == 24:                                                                           
            n_atom__C6free[i+1] =  602.0000 ; n_atom__alpha0free[i+1] =  78.0000
        elif n_atom__atomic[i+1] == 25:                                                                           
            n_atom__C6free[i+1] =  552.0000 ; n_atom__alpha0free[i+1] =  63.0000
        elif n_atom__atomic[i+1] == 26:                                                                           
            n_atom__C6free[i+1] =  482.0000 ; n_atom__alpha0free[i+1] =  56.0000
        elif n_atom__atomic[i+1] == 27:                                                                           
            n_atom__C6free[i+1] =  408.0000 ; n_atom__alpha0free[i+1] =  50.0000
        elif n_atom__atomic[i+1] == 28:                                                                           
            n_atom__C6free[i+1] =  373.0000 ; n_atom__alpha0free[i+1] =  48.0000
        elif n_atom__atomic[i+1] == 29:                                                                           
            n_atom__C6free[i+1] =  253.0000 ; n_atom__alpha0free[i+1] =  42.0000
        elif n_atom__atomic[i+1] == 30:                                                                           
            n_atom__C6free[i+1] =  284.0000 ; n_atom__alpha0free[i+1] =  40.0000
        elif n_atom__atomic[i+1] == 31:                                                                           
            n_atom__C6free[i+1] =  498.0000 ; n_atom__alpha0free[i+1] =  60.0000
        elif n_atom__atomic[i+1] == 32:                                                                           
            n_atom__C6free[i+1] =  354.0000 ; n_atom__alpha0free[i+1] =  41.0000
        elif n_atom__atomic[i+1] == 33:                                                                           
            n_atom__C6free[i+1] =  246.0000 ; n_atom__alpha0free[i+1] =  29.0000
        elif n_atom__atomic[i+1] == 34:                                                                           
            n_atom__C6free[i+1] =  210.0000 ; n_atom__alpha0free[i+1] =  25.0000
        elif n_atom__atomic[i+1] == 35:                                                                           
            n_atom__C6free[i+1] =  162.0000 ; n_atom__alpha0free[i+1] =  20.0000
        elif n_atom__atomic[i+1] == 36:                                                                           
            n_atom__C6free[i+1] =  129.6000 ; n_atom__alpha0free[i+1] =  16.8000
        else:
            sys.exit('== Error: Current implementation of free C6 and alpha0 only for elements up to Kr. Exiting now...')

    return(n_atom__C6free, n_atom__alpha0free)


############################################################
# get total energies from high-level (DFT; FHI-aims) calculations output
# also: get vdW(TS) or MBD energies if requested
############################################################

def get_fhiaims_energies():

    list_Ehl = []

    if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
        list_Embd = []
    elif dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
        list_EvdW = []

    for logfile in list_logfiles:

        # read file
        in_file = open(logfile, 'r')
        file_lines = in_file.readlines()
        lines = list(file_lines)
        in_file.close()

        for line in lines:
            if '| Total energy of the DFT ' in line:
                astring1, energy, astring2 = line.rsplit(None, 2)
                energy = float(energy) * 23.060542 # eV to kcal/mol
                list_Ehl.append( energy )

        if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
            energylist = []
            for line in lines:
                if '| MBD@rsSCS energy              :' in line:
                    astring1, energy, astring2 = line.rsplit(None, 2)
                    energylist.append( float(energy) * 23.060542 ) # eV to kcal/mol
            if not energylist:
                sys.exit('== Error: No MBD information found in '+logfile+'. Exiting now...')
            energy = energylist[-1] # last element
            list_Embd.append( energy )

        if dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
            energylist = []
            for line in lines:
                if '| vdW energy correction         :' in line:
                    astring1, energy, astring2 = line.rsplit(None, 2)
                    energylist.append( float(energy) * 23.060542 ) # eV to kcal/mol
            if not energylist:
                sys.exit('== Error: No vdW information found in '+logfile+'. Exiting now...')
            energy = energylist[-1] # last element
            list_EvdW.append( energy )

    # set maximum energy to -100. to avoid large energy values (relatve energy is preserved)
    list_Ehl = [x-max(list_Ehl)-100. for x in list_Ehl]

    # add Ehl to DataFrame
    data['E_high-level (Ehl)'] = list_Ehl

    # add Embd to DataFrame
    if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
        data['E_MBD (Embd)'] = list_Embd

    # add EvdW (TS) to DataFrame
    if dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
        data['E_vdW(TS) (EvdW)'] = list_EvdW

    return()


############################################################
# get bonding energies with FF parameters
# bonding terms are of form: kB * (r-r0)**2
############################################################

def get_bonding_energies(classpair__Kb, classpair__r0):

    list_Ebonds = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        Ebonds = get_bonding_energy(pairs__distances, classpair__Kb, classpair__r0)

        list_Ebonds.append( Ebonds )

    data['Ebonds_(FF)'] = list_Ebonds

    return()


############################################################
# get angles energies with FF parameters
# angles bending terms are of form: Ktheta * (theta-theta0)**2
############################################################

def get_angles_energies(classtriple__Ktheta, classtriple__theta0):

    list_Eangles = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        triples__angles = get_angles(n_atom__xyz)

        Eangles = get_angles_energy(triples__angles, classtriple__Ktheta, classtriple__theta0)

        list_Eangles.append( Eangles )

    data['Eangles_(FF)'] = list_Eangles

    return()


############################################################
# get torsions energies with FF parameters
# -> torsion (dihedral angles) terms are of form:
#       (V1 / 2) * ( 1+cos(  phi) )
#       (V2 / 2) * ( 1-cos(2*phi) )
#       (V3 / 2) * ( 1+cos(3*phi) )
############################################################

def get_torsions_energies(classquadruple__V1, classquadruple__V2, classquadruple__V3):

    list_Etorsions = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        quadruples__torsions = get_torsions(n_atom__xyz)

        Etorsions = get_torsions_energy(quadruples__torsions, \
                                         classquadruple__V1, classquadruple__V2, classquadruple__V3)

        list_Etorsions.append( Etorsions )

    data['Etorsions_(FF)'] = list_Etorsions

    return()


############################################################
# get improper torsions energies with FF parameters
# -> improper torsion terms are of form:
#       (V2imp / 2) * ( 1-cos(2*phi) )
############################################################

def get_improper_energies(classquadruple__impV2):

    list_Eimprops = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        quadruples__improps = get_improps(n_atom__xyz)

        Eimprops = get_improps_energy(quadruples__improps, classquadruple__impV2)

        list_Eimprops.append( Eimprops )

    data['Eimprops_(FF)'] = list_Eimprops

    return()


############################################################
# get vdW energies with FF parameters
# -> van der Waals (vdW; Lennard-Jones) terms are of form:
#       4*epsilon * f * [ (sigma/r)**12 - (sigma/r)**6 ]
#            {   0 for 1-2-interactions and 1-3-interactions
#         f ={ 1/2 for 1-4-interactions
#            {   1 for 1-5-interactions and higher
############################################################

def get_vdW_energies(pairs__sigma, pairs__epsilon, dataString):

    list_Evdw = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        Evdw = get_vdW_energy(pairs__distances, pairs__sigma, pairs__epsilon)
        list_Evdw.append( Evdw )

    data[dataString] = list_Evdw

    return()


############################################################
# get Coulomb energies with FF parameters
# -> Coulomb terms are of form: f * q1*q2 / r12
#          {   0 for 1-2-interactions and 1-3-interactions
#       f ={ 1/2 for 1-4-interactions
#          {   1 for 1-5-interactions and higher
############################################################

def get_Coulomb_energies(pairs__charge):

    list_Ecoul = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        Ecoul = get_Coulomb_energy(pairs__distances, pairs__charge)

        list_Ecoul.append( Ecoul )

    data['Ecoul_(FF)'] = list_Ecoul

    return()


############################################################
# get Coulomb energy contributions per 1-X-interaction with newFF parameters
# -> terms of form: q1*q2 / r12
############################################################

def get_Coulomb_energyContribsPer_1_X_interaction(pairs__charge):

    qqr2kcalpermole = 332.063714

    list_Ecoul_14Intacts = []
    list_Ecoul_15Intacts = []
    list_Ecoul_16plusIntacts = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        Ecoul_14Intacts = 0.
        Ecoul_15Intacts = 0.
        Ecoul_16plusIntacts = 0.

        pairs__distances = get_distances(n_atom__xyz)

        # 1-2- and 1-3-interactions are already excluded in pairs__charge dictionary
        # (see function get_charge_pairs_params())
        for pair,qq in pairs__charge.items():
            r = pairs__distances[pair]
            if pair in list_14_interacts:
                Ecoul_14Intacts += qqr2kcalpermole * qq / r
            elif pair in list_15_interacts:
                Ecoul_15Intacts += qqr2kcalpermole * qq / r
            else:
                Ecoul_16plusIntacts += qqr2kcalpermole * qq / r

        list_Ecoul_14Intacts.append(Ecoul_14Intacts)
        list_Ecoul_15Intacts.append(Ecoul_15Intacts)
        list_Ecoul_16plusIntacts.append(Ecoul_16plusIntacts)

    data['Ecoul_(FF)_14Intacts'] = list_Ecoul_14Intacts
    data['Ecoul_(FF)_15Intacts'] = list_Ecoul_15Intacts
    data['Ecoul_(FF)_16plusIntacts'] = list_Ecoul_16plusIntacts

    return()


############################################################
# get bonding energy contributions classpair-wise
# -> bonding contributions are of form: (r-r0)**2
############################################################

def get_bondEnergyContribs(classpair__r0):

    listofdicts_logfiles___classpairs__bondEnergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        classpairs__bondEnergyContribs = {}

        for pair in list_12_interacts:

            r = pairs__distances[pair]

            if n_atom__class[pair[0]] <= n_atom__class[pair[1]]:
                atuple = ( n_atom__class[pair[0]],n_atom__class[pair[1]] )
            else:
                atuple = ( n_atom__class[pair[1]],n_atom__class[pair[0]] )

            r0 = classpair__r0[ atuple ]

            if atuple in classpairs__bondEnergyContribs:
                classpairs__bondEnergyContribs[ atuple ] += (r-r0)**2.
            else:
                classpairs__bondEnergyContribs[ atuple ] = (r-r0)**2.

        listofdicts_logfiles___classpairs__bondEnergyContribs.append( classpairs__bondEnergyContribs )

    sortedlist_classpairs = []

    for classpair in listofdicts_logfiles___classpairs__bondEnergyContribs[0]:
        sortedlist_classpairs.append( classpair )
    sortedlist_classpairs = sorted(sortedlist_classpairs)

    for classpair in sortedlist_classpairs:

        list_classpair_contribs = []

        for classpairs__bondEnergyContribs in listofdicts_logfiles___classpairs__bondEnergyContribs:
            list_classpair_contribs.append( classpairs__bondEnergyContribs[ classpair ] )

        data['bonds_(r-r0)**2_classpair'+str(classpair)] = list_classpair_contribs

    return()


############################################################
# get angles energy contributions classtriple-wise
# -> angles bending contributions are of form: (theta-theta0)**2
############################################################

def get_anglesEnergyContribs(classtriple__theta0):

    listofdicts_logfiles___classtriples__anglesEnergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        triples__angles = get_angles(n_atom__xyz)

        classtriples__anglesEnergyContribs = {}

        for triple in list_123_interacts:

            theta = triples__angles[triple]

            if n_atom__class[triple[0]] <= n_atom__class[triple[2]]:
                atuple = ( n_atom__class[triple[0]],n_atom__class[triple[1]],n_atom__class[triple[2]] )
            else:
                atuple = ( n_atom__class[triple[2]],n_atom__class[triple[1]],n_atom__class[triple[0]] )

            theta0 = classtriple__theta0[ atuple ]

            if atuple in classtriples__anglesEnergyContribs:
                classtriples__anglesEnergyContribs[ atuple ] += (math.pi/180.)**2. * (theta-theta0)**2.
            else:
                classtriples__anglesEnergyContribs[ atuple ] = (math.pi/180.)**2. * (theta-theta0)**2.

        listofdicts_logfiles___classtriples__anglesEnergyContribs.append( classtriples__anglesEnergyContribs )

    sortedlist_classtriples = []

    for classtriple in listofdicts_logfiles___classtriples__anglesEnergyContribs[0]:
        sortedlist_classtriples.append( classtriple )
    sortedlist_classtriples = sorted(sortedlist_classtriples, key=operator.itemgetter(1,2,0))

    for classtriple in sortedlist_classtriples:

        list_classtriple_contribs = []

        for classtriples__anglesEnergyContribs in listofdicts_logfiles___classtriples__anglesEnergyContribs:
            list_classtriple_contribs. append( classtriples__anglesEnergyContribs[ classtriple ] )

        data['angles_(pi/180.)**2_*_(theta-theta0)**2_classtriple'+str(classtriple)] = list_classtriple_contribs

    return()


############################################################
# get improper torsions energy contributions classquadruple-wise
# -> improper torsion contributions are of form:
#       0.5 * ( 1-cos(2*phi) )
############################################################

def get_impropsEnergyContribs():

    listofdicts_logfiles___classquadruples__impropsEnergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        quadruples__improps = get_improps(n_atom__xyz)

        classquadruples__impropsEnergyContribs = {}

        for quadruple in list_imp1234_interacts:

            phi = quadruples__improps[quadruple]

            atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )

            if atuple in classquadruples__impropsEnergyContribs:
                classquadruples__impropsEnergyContribs[ atuple ] += 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )
            else:
                classquadruples__impropsEnergyContribs[ atuple ] = 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )

        listofdicts_logfiles___classquadruples__impropsEnergyContribs.append( classquadruples__impropsEnergyContribs )

    sortedlist_imps_classquadruples = []

    for classquadruple in listofdicts_logfiles___classquadruples__impropsEnergyContribs[0]:
        sortedlist_imps_classquadruples.append( classquadruple )
    sortedlist_imps_classquadruples = sorted(sortedlist_imps_classquadruples, key=operator.itemgetter(2,3,0,1))

    for classquadruple in sortedlist_imps_classquadruples:

        list_classquadruple_contribs = []

        for classquadruples__impropsEnergyContribs in listofdicts_logfiles___classquadruples__impropsEnergyContribs:
            list_classquadruple_contribs.append( classquadruples__impropsEnergyContribs[ classquadruple ] )

        data['improps_0.5*(1-cos(2*(pi/180)*phi))_classquadruple'+str(classquadruple)] = list_classquadruple_contribs

    return()


############################################################
# get torsions energy contributions classquadruple-wise
# torsion (dihedral angles) contributions are of form:
#       0.5 * ( 1+cos(  phi) )
#       0.5 * ( 1-cos(2*phi) )
#       0.5 * ( 1+cos(3*phi) )
############################################################

def get_torsionsEnergyContribs(origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3):

    # origFF___classquadruple__V{1,2,3} -> only needed to look up non-zero terms from original FF

    listofdicts_logfiles___classquadruples__torsionsV1EnergyContribs = []
    listofdicts_logfiles___classquadruples__torsionsV2EnergyContribs = []
    listofdicts_logfiles___classquadruples__torsionsV3EnergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        quadruples__torsions = get_torsions(n_atom__xyz)

        classquadruples__torsionsV1EnergyContribs = {}
        classquadruples__torsionsV2EnergyContribs = {}
        classquadruples__torsionsV3EnergyContribs = {}

        for quadruple in list_1234_interacts:

            phi = quadruples__torsions[quadruple]

            if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
                atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
            else:
                atuple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )

            # classquadruples for which V1 == V2 == V3 == 0 are never taken into account
            if ( atuple in origFF___classquadruple__V1 ) or ( atuple in origFF___classquadruple__V2 ) or ( atuple in origFF___classquadruple__V3 ):
                # add only if original FF parameter is also non-zero (same for V2, V3 below),
                #   except if requested to use all possible V1, V2, V3 -> then do that
                 if ( atuple in origFF___classquadruple__V1 ) or ( dict_keywords['RegressionTorsionalVall'] == True ):
                     if atuple in classquadruples__torsionsV1EnergyContribs:
                         classquadruples__torsionsV1EnergyContribs[ atuple ] += 0.5 * ( 1. + math.cos( ( math.pi/180. ) * phi ) )
                     else:
                         classquadruples__torsionsV1EnergyContribs[ atuple ] = 0.5 * ( 1. + math.cos( ( math.pi/180. ) * phi ) )
                 
                 if ( atuple in origFF___classquadruple__V2 ) or ( dict_keywords['RegressionTorsionalVall'] == True ):
                     if atuple in classquadruples__torsionsV2EnergyContribs:
                         classquadruples__torsionsV2EnergyContribs[ atuple ] += 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )
                     else:
                         classquadruples__torsionsV2EnergyContribs[ atuple ] = 0.5 * ( 1. - math.cos( 2. * ( math.pi/180. ) * phi ) )
                 
                 if ( atuple in origFF___classquadruple__V3 ) or ( dict_keywords['RegressionTorsionalVall'] == True ):
                     if atuple in classquadruples__torsionsV3EnergyContribs:
                         classquadruples__torsionsV3EnergyContribs[ atuple ] += 0.5 * ( 1. + math.cos( 3. * ( math.pi/180. ) * phi ) )
                     else:
                         classquadruples__torsionsV3EnergyContribs[ atuple ] = 0.5 * ( 1. + math.cos( 3. * ( math.pi/180. ) * phi ) )

        listofdicts_logfiles___classquadruples__torsionsV1EnergyContribs.append( classquadruples__torsionsV1EnergyContribs )
        listofdicts_logfiles___classquadruples__torsionsV2EnergyContribs.append( classquadruples__torsionsV2EnergyContribs )
        listofdicts_logfiles___classquadruples__torsionsV3EnergyContribs.append( classquadruples__torsionsV3EnergyContribs )

    sortedlist_V1classquadruples = []
    sortedlist_V2classquadruples = []
    sortedlist_V3classquadruples = []

    for V1classquadruple in listofdicts_logfiles___classquadruples__torsionsV1EnergyContribs[0]:
        sortedlist_V1classquadruples.append( V1classquadruple )
    sortedlist_V1classquadruples = sorted(sortedlist_V1classquadruples, key=operator.itemgetter(1,2,0,3))

    for V2classquadruple in listofdicts_logfiles___classquadruples__torsionsV2EnergyContribs[0]:
        sortedlist_V2classquadruples.append( V2classquadruple )
    sortedlist_V2classquadruples = sorted(sortedlist_V2classquadruples, key=operator.itemgetter(1,2,0,3))

    for V3classquadruple in listofdicts_logfiles___classquadruples__torsionsV3EnergyContribs[0]:
        sortedlist_V3classquadruples.append( V3classquadruple )
    sortedlist_V3classquadruples = sorted(sortedlist_V3classquadruples, key=operator.itemgetter(1,2,0,3))

    for V1classquadruple in sortedlist_V1classquadruples:

        list_V1classquadruple_contribs = []

        for classquadruples__torsionsV1EnergyContribs in listofdicts_logfiles___classquadruples__torsionsV1EnergyContribs:
            list_V1classquadruple_contribs.append( classquadruples__torsionsV1EnergyContribs[ V1classquadruple ] )

        data['torsionsV1_0.5*(1+cos((pi/180)*phi))_classquadruple'+str(V1classquadruple)] = list_V1classquadruple_contribs

    for V2classquadruple in sortedlist_V2classquadruples:

        list_V2classquadruple_contribs = []

        for classquadruples__torsionsV2EnergyContribs in listofdicts_logfiles___classquadruples__torsionsV2EnergyContribs:
            list_V2classquadruple_contribs.append( classquadruples__torsionsV2EnergyContribs[ V2classquadruple ] )

        data['torsionsV2_0.5*(1-cos(2*(pi/180)*phi))_classquadruple'+str(V2classquadruple)] = list_V2classquadruple_contribs

    for V3classquadruple in sortedlist_V3classquadruples:

        list_V3classquadruple_contribs = []

        for classquadruples__torsionsV3EnergyContribs in listofdicts_logfiles___classquadruples__torsionsV3EnergyContribs:
            list_V3classquadruple_contribs.append( classquadruples__torsionsV3EnergyContribs[ V3classquadruple ] )

        data['torsionsV3_0.5*(1+cos(3*(pi/180)*phi))_classquadruple'+str(V3classquadruple)] = list_V3classquadruple_contribs

    return()


############################################################
# get vdW energy contributions typepair-wise
# -> van der Waals (vdW; Lennard-Jones) contributions are of form:
#       4 * f * [ (sigma/r)**12 - (sigma/r)**6 ]
#            {   0 for 1-2-interactions and 1-3-interactions
#         f ={ 1/2 for 1-4-interactions
#            {   1 for 1-5-interactions and higher
############################################################

def get_vdWenergyContribs(pairs__sigma):

    listofdicts_logfiles___typepairs__vdWenergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        pairs__distances = get_distances(n_atom__xyz)

        typepairs__vdWenergyContribs = {}

        for pair,sigma in pairs__sigma.items():

            if pair in list_14_interacts:
                f = 0.5
            else:
                f = 1.

            r = pairs__distances[pair]

            if n_atom__type[pair[0]] <= n_atom__type[pair[1]]:
                atuple = ( n_atom__type[pair[0]],n_atom__type[pair[1]] )
            else:
                atuple = ( n_atom__type[pair[1]],n_atom__type[pair[0]] )

            if atuple in typepairs__vdWenergyContribs:
                typepairs__vdWenergyContribs[ atuple ] += 4. * f * ( (sigma/r)**12. - (sigma/r)**6. )
            else:
                typepairs__vdWenergyContribs[ atuple ] = 4. * f * ( (sigma/r)**12. - (sigma/r)**6. )

        listofdicts_logfiles___typepairs__vdWenergyContribs.append( typepairs__vdWenergyContribs )

    sortedlist_typepairs = []

    for typepair in listofdicts_logfiles___typepairs__vdWenergyContribs[0]:
        sortedlist_typepairs.append( typepair )
    sortedlist_typepairs = sorted(sortedlist_typepairs)

    for typepair in sortedlist_typepairs:

        list_typepair_contribs = []

        for typepairs__vdWenergyContribs in listofdicts_logfiles___typepairs__vdWenergyContribs:
            list_typepair_contribs.append( typepairs__vdWenergyContribs[ typepair ] )

        data['vdW_4*f*((sigma/r)**12-(sigma/r)**6)_typepair'+str(typepair)] = list_typepair_contribs

    return()


###############################################################################
# do regression for total energy to estimate Coulomb fudge factors
###############################################################################

def do_regression_Coulomb_fudge_factors():

    predictors = []

    for colname in data:
        if ( 'Ecoul_(FF)_14Intacts' in colname ):
            predictors.append( colname )
        if ( 'Ecoul_(FF)_15Intacts' in colname ):
            predictors.append( colname )
        if dict_keywords['fine_tune_only_f14_and_f15'] == False:
            if ( 'Ecoul_(FF)_16plusIntacts' in colname ):
                predictors.append( colname )

    reg = LinearRegression(fit_intercept=True, normalize=False)

    # note that the vdW energy of the origFF is used here
    data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds'] = data['E_high-level (Ehl)'] \
                                                          - data['Evdw_(origFF)'] \
                                                          - data['Etorsions_(FF)'] \
                                                          - data['Eimprops_(FF)'] \
                                                          - data['Eangles_(FF)'] \
                                                          - data['Ebonds_(FF)']
    data['Ehl-Ecoul16plusIntacts-EvdW-Etorsions-Eimprops-Eangles-Ebonds'] \
            = data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds'] - data['Ecoul_(FF)_16plusIntacts']

    if dict_keywords['fine_tune_only_f14_and_f15'] == True:
        reg.fit(data[predictors], data['Ehl-Ecoul16plusIntacts-EvdW-Etorsions-Eimprops-Eangles-Ebonds'])
    elif dict_keywords['fine_tune_only_f14_and_f15'] == False:
        reg.fit(data[predictors], data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds'])

    # get some infos
    if False:
        y_pred = reg.predict(data[predictors])
        rss = sum( (y_pred-data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds'])**2. )
        print('RSS = ', rss)
        print('intercept = ', reg.intercept_)
        print('\n====\n')

        #for i in range(len(reg.coef_)):
        #    print(predictors[i], reg.coef_[i])

    f14 = reg.coef_[0]
    f15 = reg.coef_[1]
    if dict_keywords['fine_tune_only_f14_and_f15'] == False:
        f16plus = reg.coef_[2]
    elif dict_keywords['fine_tune_only_f14_and_f15'] == True:
        f16plus = 1.0 # default value

    # get TINKER-coorect fudge factors and dielectric constant
    dielectric = 1./f16plus
    f14 = f14 * dielectric
    f15 = f15 * dielectric

    # write some info
    print('Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
    print('Estimated Coulomb fudge factor for 1-5-interactions: '+format(f15, '.3f'))
    if dict_keywords['fine_tune_only_f14_and_f15'] == False:
        print('Estimated Coulomb fudge factor for 1-6-interactions or higher: '+format(f16plus, '.3f'))
    print('\n====\n')

    if ( f14 < 0. ):
        print('Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
        sys.exit('== Error: Estimated Coulomb fudge factor for 1-4-interactions is negative. That seems strange. Exiting now...')
    if ( f15 < 0. ):
        print('Estimated Coulomb fudge factor for 1-5-interactions: '+format(f15, '.3f'))
        sys.exit('== Error: Estimated Coulomb fudge factor for 1-5-interactions is negative. That seems strange. Exiting now...')
    if ( f16plus < 0. ):
        print('Estimated Coulomb fudge factor for 1-6-interactions or higher: '+format(f16plus, '.3f'))
        sys.exit('== Error: Estimated Coulomb fudge factor for 1-6-interactions or higher is negative. That seems strange. Exiting now...')

    if ( f14 > 1.0 ):
        print('Warning! TINKER only uses fudge factors that do not exceed values of 1.0')
        print('         Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
        print('         --> Reset to 1.0')
        print('\n====\n')
        f14 = 1.0
    if ( f15 > 1.0 ):
        print('Warning! TINKER only uses fudge factors that do not exceed values of 1.0')
        print('         Estimated Coulomb fudge factor for 1-5-interactions: '+format(f15, '.3f'))
        print('         --> Reset to 1.0')
        print('\n====\n')
        f15 = 1.0

    return(f14, f15, dielectric)


###############################################################################
# do regression for vdW energy (TS or MBD) to estimate epsilon parameters
###############################################################################

def do_regression_epsilon_vdW():

    predictors = []

    for colname in data:
        if ( 'vdW' in colname ) and ( 'typepair' in colname ):
            predictors.append( colname )

    if dict_keywords['RegressionEpsilonMethod'] == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True, normalize=False)
    elif dict_keywords['RegressionEpsilonMethod'] == 'Ridge':
        reg = Ridge(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False)
    elif dict_keywords['RegressionEpsilonMethod'] == 'Lasso':
        if dict_keywords['RestrictRegressionEpsilonPositive'] == True:
            reg = Lasso(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False, positive=True)
        elif dict_keywords['RestrictRegressionEpsilonPositive'] == False:
            reg = Lasso(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False, positive=False)

    if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
        reg.fit(data[predictors], data[ 'E_MBD (Embd)' ])
    elif dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
        reg.fit(data[predictors], data[ 'E_vdW(TS) (EvdW)' ])

    # get some infos
    if True:
        y_pred = reg.predict(data[predictors])
        if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
            rss = sum( (y_pred-data['E_MBD (Embd)'])**2. )
        elif dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
            rss = sum( (y_pred-data['E_vdW(TS) (EvdW)'])**2. )
#        print('RSS = ', rss)
#        print('\n====\n')
#        print('intercept = ', reg.intercept_)

        #for i in range(len(reg.coef_)):
        #    print(predictors[i], reg.coef_[i])


    # get list of (already sorted) typepairs from DataFrame
    sortedlist_typepairs = []
    for colname in data:
        if ( 'vdW' in colname ) and ( 'typepair' in colname ):
            string_typepair = re.findall('\([0-9]*, [0-9]*\)',colname)
            typepair = ast.literal_eval(string_typepair[0])
            sortedlist_typepairs.append( typepair )

    typepairs__epsilon = {}

    for i in range(len(sortedlist_typepairs)):
        typepairs__epsilon[ sortedlist_typepairs[i] ] = reg.coef_[i]

    pairs__epsilon = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):
                if n_atom__type[i+1] <= n_atom__type[j+1]:
                    pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[i+1],n_atom__type[j+1] ) ]
                else:
                    pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[j+1],n_atom__type[i+1] ) ]

    return(pairs__epsilon, \
            typepairs__epsilon)


###############################################################################
# do regression for total energy to estimate epsilon parameters
###############################################################################

def do_regression_epsilon_Tot(origFF___typepairs__epsilon):

    predictors = []

    for colname in data:
        if ( 'vdW' in colname ) and ( 'typepair' in colname ):
            predictors.append( colname )

    if dict_keywords['RegressionEpsilonMethod'] == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True, normalize=False)
    elif dict_keywords['RegressionEpsilonMethod'] == 'Ridge':
        reg = Ridge(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False)
    elif dict_keywords['RegressionEpsilonMethod'] == 'Lasso':
        if dict_keywords['RestrictRegressionEpsilonPositive'] == True:
            reg = Lasso(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False, positive=True)
        elif dict_keywords['RestrictRegressionEpsilonPositive'] == False:
            reg = Lasso(alpha=dict_keywords['regularization_parameter_epsilon'], fit_intercept=True, normalize=False, positive=False)

    data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds'] = data['E_high-level (Ehl)'] \
                                                                     - data['Ecoul_(FF)'] \
                                                                     - data['Etorsions_(FF)'] \
                                                                     - data['Eimprops_(FF)'] \
                                                                     - data['Eangles_(FF)'] \
                                                                     - data['Ebonds_(FF)']

    reg.fit(data[predictors], data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds'])

    # get some infos
    if True:
        y_pred = reg.predict(data[predictors])
        rss = sum( (y_pred-data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds'])**2. )
#        print('RSS = ', rss)
#        print('\n====\n')
#        print('intercept = ', reg.intercept_)

        #for i in range(len(reg.coef_)):
        #    print(predictors[i], reg.coef_[i])


    # get list of (already sorted) typepairs from DataFrame
    sortedlist_typepairs = []
    for colname in data:
        if ( 'vdW' in colname ) and ( 'typepair' in colname ):
            string_typepair = re.findall('\([0-9]*, [0-9]*\)',colname)
            typepair = ast.literal_eval(string_typepair[0])
            sortedlist_typepairs.append( typepair )

    typepairs__epsilon = {}

    for i in range(len(sortedlist_typepairs)):
        typepairs__epsilon[ sortedlist_typepairs[i] ] = reg.coef_[i]

    pairs__epsilon = {}

    # loop over all atom pairs, exclude 1-2 and 1-3 interactions
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if ( not ( (i+1,j+1) in list_12_interacts ) ) and ( not ( (i+1,j+1) in list_13_interacts )):
                if n_atom__type[i+1] <= n_atom__type[j+1]:
                    pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[i+1],n_atom__type[j+1] ) ]
                else:
                    pairs__epsilon[ (i+1,j+1) ] = typepairs__epsilon[ ( n_atom__type[j+1],n_atom__type[i+1] ) ]

    return(pairs__epsilon, \
            typepairs__epsilon)


###################################################################
# do regression to estimate torsional parameters (V1, V2, V3)
###################################################################

def do_regression_torsionalV():

    predictors = []

    for colname in data:
        if ( 'torsionsV' in colname ) and ( 'classquadruple' in colname ):
            predictors.append( colname )

    if dict_keywords['Regression_torsionalV_Method'] == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True, normalize=False)
    elif dict_keywords['Regression_torsionalV_Method'] == 'Ridge':
        reg = Ridge(alpha=dict_keywords['regularization_parameter_torsionalV'], fit_intercept=True, normalize=False)
    elif dict_keywords['Regression_torsionalV_Method'] == 'Lasso':
        reg = Lasso(alpha=dict_keywords['regularization_parameter_torsionalV'], fit_intercept=True, normalize=False)

    data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds'] = data['E_high-level (Ehl)'] \
                                                      - data['Ecoul_(FF)'] \
                                                      - data['Evdw_(FF)'] \
                                                      - data['Eimprops_(FF)'] \
                                                      - data['Eangles_(FF)'] \
                                                      - data['Ebonds_(FF)']

    reg.fit(data[predictors], data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds'])

    # get some infos
    if False:
        y_pred = reg.predict(data[predictors])
        rss = sum( (y_pred-data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds'])**2. )
        print('RSS = ', rss)
        print('\n====\n')
#        print('intercept = ', reg.intercept_)

        #for i in range(len(reg.coef_)):
        #    print(predictors[i], reg.coef_[i])


    # get list of (already sorted) classquadruples from DataFrame
    sortedlist_classquadruplesV1 = []
    sortedlist_classquadruplesV2 = []
    sortedlist_classquadruplesV3 = []
    for colname in data:
        if ( 'torsionsV1' in colname ) and ( 'classquadruple' in colname ):
            string_classquadrupleV1 = re.findall('\([0-9]*, [0-9]*, [0-9]*, [0-9]*\)',colname)
            classquadrupleV1 = ast.literal_eval(string_classquadrupleV1[0])
            sortedlist_classquadruplesV1.append( classquadrupleV1 )
        if ( 'torsionsV2' in colname ) and ( 'classquadruple' in colname ):
            string_classquadrupleV2 = re.findall('\([0-9]*, [0-9]*, [0-9]*, [0-9]*\)',colname)
            classquadrupleV2 = ast.literal_eval(string_classquadrupleV2[0])
            sortedlist_classquadruplesV2.append( classquadrupleV2 )
        if ( 'torsionsV3' in colname ) and ( 'classquadruple' in colname ):
            string_classquadrupleV3 = re.findall('\([0-9]*, [0-9]*, [0-9]*, [0-9]*\)',colname)
            classquadrupleV3 = ast.literal_eval(string_classquadrupleV3[0])
            sortedlist_classquadruplesV3.append( classquadrupleV3 )

    classquadruple__V1 = {}
    classquadruple__V2 = {}
    classquadruple__V3 = {}

    for i in range(len(sortedlist_classquadruplesV1)):
        classquadruple__V1[ sortedlist_classquadruplesV1[i] ] = reg.coef_[i]
    for i in range(len(sortedlist_classquadruplesV2)):
        classquadruple__V2[ sortedlist_classquadruplesV2[i] ] = reg.coef_[i+len(sortedlist_classquadruplesV1)]
    for i in range(len(sortedlist_classquadruplesV3)):
        classquadruple__V3[ sortedlist_classquadruplesV3[i] ] = reg.coef_[i+len(sortedlist_classquadruplesV1)+len(sortedlist_classquadruplesV2)]

    return(classquadruple__V1, \
            classquadruple__V2, \
            classquadruple__V3)


###################################################################
# do regression to estimate torsional parameters (V1, V2, V3)
###################################################################

def do_regression_imptorsionalV():
    
    predictors = []
    
    for colname in data:
        if ( 'improps' in colname ) and ( 'classquadruple' in colname ):
            predictors.append( colname )
    
    if dict_keywords['Regression_imptorsionalV_Method'] == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True, normalize=False)
    elif dict_keywords['Regression_imptorsionalV_Method'] == 'Ridge':
        reg = Ridge(alpha=dict_keywords['regularization_parameter_imptorsionalV'], fit_intercept=True, normalize=False)
    elif dict_keywords['Regression_imptorsionalV_Method'] == 'Lasso':
        reg = Lasso(alpha=dict_keywords['regularization_parameter_imptorsionalV'], fit_intercept=True, normalize=False)
    
    data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds'] = data['E_high-level (Ehl)'] \
                                                         - data['Ecoul_(FF)'] \
                                                         - data['Evdw_(FF)'] \
                                                         - data['Etorsions_(FF)'] \
                                                         - data['Eangles_(FF)'] \
                                                         - data['Ebonds_(FF)']
    
    reg.fit(data[predictors], data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds'])
    
    # get some infos
    if False:
        y_pred = reg.predict(data[predictors])
        rss = sum( (y_pred-data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds'])**2. )
        print('RSS = ', rss)
        print('\n====\n')
    #       print('intercept = ', reg.intercept_)
    
           #for i in range(len(reg.coef_)):
           #    print(predictors[i], reg.coef_[i])
    
    
    # get list of (already sorted) classquadruples from DataFrame
    sortedlist_classquadruplesimpV2 = []
    for colname in data:
        if ( 'improps' in colname ) and ( 'classquadruple' in colname ):
            string_classquadrupleimpV2 = re.findall('\([0-9]*, [0-9]*, [0-9]*, [0-9]*\)',colname)
            classquadrupleimpV2 = ast.literal_eval(string_classquadrupleimpV2[0])
            sortedlist_classquadruplesimpV2.append( classquadrupleimpV2 )
               
    classquadruple__impV2 = {}
       
    for i in range(len(sortedlist_classquadruplesimpV2)):
        classquadruple__impV2[ sortedlist_classquadruplesimpV2[i] ] = reg.coef_[i]
           
    return(classquadruple__impV2)
    
	
############################################################
# print atom type/class overview
############################################################

def print_atom_type_class_overview():

    print('Atom type/class overview:')
    print('-------------------------')
    print('                         ')
    print('  Atom   Symbol   Type   Class   Atomic    Mass   Valence   Description')
    print('-----------------------------------------------------------------------')

    for i in range(n_atoms):
        printf("  %3d      %-3s     %3d    %3d      %3d    %6.3f     %3d    %s", i+1, class__symbol[n_atom__class[i+1]], n_atom__type[i+1], n_atom__class[i+1], n_atom__atomic[i+1], n_atom__mass[i+1], n_atom__valence[i+1], n_atom__description[i+1])

    print('\n====\n')
    
    return()


############################################################
# print original and new r0 parameters classpair-wise
############################################################

def print_r0_kB_params(origFF___classpair__r0, newFF___classpair__r0, origFF___classpair__Kb):

    print('r0 parameters:')
    print('--------------')
    print('              ')

    if dict_keywords['fine_tune_r0'] == False:
        print('>>> Original FF parameters have been used for new FF, i.e. r0 parameters have not been altered.')
    elif dict_keywords['fine_tune_r0'] == True:
        print('>>> An average distance r0 has been computed to estimate (fine-tune) r0.')

    print('                                      ')
    print('  Atom class pair   r0         r0     ')
    print('                    (origFF)   (newFF)')
    print('--------------------------------------')

    sortedlist_classpairs = []

    for classpair in origFF___classpair__r0:
        sortedlist_classpairs.append( classpair )
    sortedlist_classpairs = sorted(sortedlist_classpairs)

    for classpair in sortedlist_classpairs:
        printf("       %10s  %7.4f    %7.4f \n", classpair, origFF___classpair__r0[classpair], newFF___classpair__r0[classpair])

    print('\n====\n')

    print('kB parameters:')
    print('--------------')
    print('              ')

    print('>>> Original FF parameters have been used for new FF, i.e. kB parameters have not been altered.')

    print('                            ')
    print('  Atom class pair   kB      ')
    print('                    (origFF)')
    print('----------------------------')

    for classpair in sortedlist_classpairs:
        printf("       %10s   %7.3f \n", classpair, origFF___classpair__Kb[classpair])

    print('\n====\n')

    return()


############################################################
# print original and new theta0 parameters classtriple-wise, and Ktheta parameters (unaltered)
############################################################

def print_theta0_params(origFF___classtriple__theta0, newFF___classtriple__theta0, origFF___classtriple__Ktheta):

    print('theta0 parameters:')
    print('------------------')
    print('                  ')

    if dict_keywords['fine_tune_theta0'] == False:
        print('>>> Original FF parameters have been used for new FF, i.e. theta0 parameters have not been altered.')
    elif dict_keywords['fine_tune_theta0'] == True:
        print('>>> An average angle theta0 has been computed to estimate (fine-tune) theta0.')

    print('                                        ')
    print('  Atom class triple   theta0     theta0 ')
    print('                      (origFF)   (newFF)')
    print('----------------------------------------')

    sortedlist_classtriples = []

    for classtriple in origFF___classtriple__theta0:
        sortedlist_classtriples.append( classtriple )
    sortedlist_classtriples = sorted(sortedlist_classtriples, key=operator.itemgetter(1,2,0))

    for classtriple in sortedlist_classtriples:
        printf("    %15s   %8.4f   %8.4f \n", classtriple, origFF___classtriple__theta0[classtriple], newFF___classtriple__theta0[classtriple])

    print('\n====\n')

    print('Ktheta parameters:')
    print('------------------')
    print('                  ')

    print('>>> Original FF parameters have been used for new FF, i.e. Ktheta parameters have not been altered.')

    print('                              ')
    print('  Atom class triple   Ktheta  ')
    print('                      (origFF)')
    print('------------------------------')

    for classtriple in sortedlist_classtriples:
        printf("    %15s %8.3f\n", classtriple, origFF___classtriple__Ktheta[classtriple])

    print('\n====\n')

    return()


############################################################
# print torsions parameters 
############################################################

def print_torsions_params(origFF___classquadruple__V1, \
                           origFF___classquadruple__V2, \
                           origFF___classquadruple__V3, \
                           newFF___classquadruple__V1, \
                           newFF___classquadruple__V2, \
                           newFF___classquadruple__V3):

    sortedlist_classquadruples = []
    sortedlist_classquadruples_zeroVs = []

    for quadruple in list_1234_interacts:

        if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
            classquadruple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
        else:
            classquadruple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )

        if classquadruple in newFF___classquadruple__V1:
            sortedlist_classquadruples.append( classquadruple )
        else:
            if classquadruple in newFF___classquadruple__V2:
                sortedlist_classquadruples.append( classquadruple )
            else:
                if classquadruple in newFF___classquadruple__V3:
                    sortedlist_classquadruples.append( classquadruple )
                else:
                    sortedlist_classquadruples_zeroVs.append( classquadruple )

    sortedlist_classquadruples = sorted(set(sortedlist_classquadruples), key=operator.itemgetter(1,2,0,3))
    sortedlist_classquadruples_zeroVs = sorted(set(sortedlist_classquadruples_zeroVs), key=operator.itemgetter(1,2,0,3))
    
    print('torsions parameters V1, V2, V3')
    print('------------------------------')
    print('                              ')

    if dict_keywords['fine_tune_torsionalV'] == False:
        print('>>> Original FF parameters have been used for new FF, i.e. torsions parameters V1, V2, V3 have not been altered.')
    elif dict_keywords['fine_tune_torsionalV'] == True:
        if dict_keywords['Regression_torsionalV_Method'] == 'LinearRegression':
            print('>>> torsions parameters V1, V2, V3 have been assigned using Linear regression from total energies.')
        elif dict_keywords['Regression_torsionalV_Method'] == 'Ridge':
            print('>>> torsions parameters V1, V2, V3 have been assigned using Ridge regression from total energies.')
        elif dict_keywords['Regression_torsionalV_Method'] == 'Lasso':
            print('>>> torsions parameters V1, V2, V3 have been assigned using Lasso regression from total energies.')
        if dict_keywords['RegressionTorsionalVall'] == True:
            print('  >> All torsional parameters V1, V2, V3 are explicitly taken into consideration.')
            print('     (Dihedral angles with V1=V2=V3=0 in original FF are never taken into consideration.)')
        elif dict_keywords['RegressionTorsionalVall'] == False:
            print('  >> Only torsional parameters V1, V2, V3 are taken into consideration that are non-zero in original FF.')
            print('     (Dihedral angles with V1=V2=V3=0 in original FF are never taken into consideration.)')

    print('                                           ')
    print('  Atom class quadruple    V1         V1     ')
    print('                          (origFF)   (newFF)')
    print('--------------------------------------------')

    for classquadruple in sortedlist_classquadruples:
        if classquadruple in newFF___classquadruple__V1:
            if classquadruple not in origFF___classquadruple__V1:
                origFF___classquadruple__V1[classquadruple] = 0.
            printf("    %3d  %3d  %3d  %3d    %6.3f     %6.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V1[classquadruple], newFF___classquadruple__V1[classquadruple])

    print('                                           ')
    print('  Atom class quadruple    V2         V2     ')
    print('                          (origFF)   (newFF)')
    print('--------------------------------------------')

    for classquadruple in sortedlist_classquadruples:
        if classquadruple in newFF___classquadruple__V2:
            if classquadruple not in origFF___classquadruple__V2:
                origFF___classquadruple__V2[classquadruple] = 0.
            printf("    %3d  %3d  %3d  %3d    %6.3f     %6.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V2[classquadruple], newFF___classquadruple__V2[classquadruple])

    print('                                           ')
    print('  Atom class quadruple    V3         V3     ')
    print('                          (origFF)   (newFF)')
    print('--------------------------------------------')

    for classquadruple in sortedlist_classquadruples:
        if classquadruple in newFF___classquadruple__V3:
            if classquadruple not in origFF___classquadruple__V3:
                origFF___classquadruple__V3[classquadruple] = 0.
            printf("    %3d  %3d  %3d  %3d    %6.3f     %6.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V3[classquadruple], newFF___classquadruple__V3[classquadruple])

    print('\n\nDihedral angles with V1=V2=V3=0 in original FF (unaltered):')
    print('-----------------------------------------------------------')

    print('                      ')
    print('  Atom class quadruple')
    print('----------------------')

    for classquadruple in sortedlist_classquadruples_zeroVs:
        printf("    %3d  %3d  %3d  %3d\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3])

    print('\n====\n')

    return()


############################################################
# print improper torsions parameters (unaltered)
############################################################

def print_improps_params(origFF___classquadruple__impV2, newFF___classquadruple__impV2):

    print('improper V2 parameters:')
    print('-----------------------')
    print('                       ')

    if dict_keywords['fine_tune_imptorsionalV'] == False:
        print('>>> Original FF parameters have been used for new FF, i.e. improper V2 parameters have not been altered.')
    elif dict_keywords['fine_tune_imptorsionalV'] == True:
        if dict_keywords['Regression_imptorsionalV_Method'] == 'LinearRegression':
            print('>>> improper torsions parameters V2 have been assigned using Linear regression from total energies.')
        elif dict_keywords['Regression_imptorsionalV_Method'] == 'Ridge':
            print('>>> improper torsions parameters V2 have been assigned using Ridge regression from total energies.')
        elif dict_keywords['Regression_imptorsionalV_Method'] == 'Lasso':
            print('>>> improper torsions parameters V2 have been assigned using Lasso regression from total energies.')

    print('                              ')
    print('  Atom class quadruple    impV2      impV2  ')
    print('                          (origFF)   (newFF)')
    print('--------------------------------------------')

    sortedlist_classquadruples = []

    for classquadruple in newFF___classquadruple__impV2:
        sortedlist_classquadruples.append( classquadruple )
    sortedlist_classquadruples = sorted(sortedlist_classquadruples, key=operator.itemgetter(2,3,0,1))

    for classquadruple in sortedlist_classquadruples:
        printf("    %3d  %3d  %3d  %3d    %6.3f     %6.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__impV2[classquadruple], newFF___classquadruple__impV2[classquadruple])

    print('\n====\n')

    return()


############################################################
# print original and new sigma parameters typepair-wise
############################################################

def print_sigma_params(origFF___typepairs__sigma, newFF___typepairs__sigma):

    print('sigma parameters:')
    print('-----------------')
    print('                 ')

    if dict_keywords['fine_tune_sigma'] == 'False':
        print('>>> Original FF parameters have been used for new FF, i.e. sigma parameters have not been altered.')
    elif dict_keywords['fine_tune_sigma'] == 'TS':
        print('>>> sigma has been calculated (fine-tuned) using the Tkatchenko-Scheffler (TS) method.')
        if dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] == True:
            print('  >> sigma parameters are explicitly set to zero if already zero in original FF.')


    print('                                      ')
    print('   Atom type pair   sigma      sigma  ')
    print('                    (origFF)   (newFF)')
    print('--------------------------------------')

    sortedlist_typepairs = []

    for typepair in origFF___typepairs__sigma:
        sortedlist_typepairs.append( typepair )
    sortedlist_typepairs = sorted(sortedlist_typepairs)

    for typepair in sortedlist_typepairs:
        printf("       %10s  %7.4f    %7.4f \n", typepair, origFF___typepairs__sigma[typepair], newFF___typepairs__sigma[typepair])

    print('\n====\n')

    return()


############################################################
# print original and new epsilon parameters typepair-wise
############################################################

def print_epsilon_params(origFF___typepairs__epsilon, newFF___typepairs__epsilon):

    print('epsilon parameters:')
    print('-----------------')
    print('                 ')

    if dict_keywords['fine_tune_epsilon'] == 'False':
        print('>>> Original FF parameters have been used for new FF, i.e. epsilon parameters have not been altered.')
    elif dict_keywords['fine_tune_epsilon'] == 'TS':
        print('>>> epsilon has been calculated (fine-tuned) using the Tkatchenko-Scheffler (TS) method.')
        if dict_keywords['SetExplicitlyZero_vdW_SigmaEpsilon'] == True:
            print('  >> epsilon parameters are explicitly set to zero if already zero in original FF.')
    elif dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
        if dict_keywords['RegressionEpsilonMethod'] == 'LinearRegression':
            print('>>> epsilon has been assigned using Linear regression from Tkatchenko-Scheffler (TS) energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Ridge':
            print('>>> epsilon has been assigned using Ridge regression from Tkatchenko-Scheffler (TS) energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Lasso':
            print('>>> epsilon has been assigned using Lasso regression from Tkatchenko-Scheffler (TS) energies.')
            if dict_keywords['RestrictRegressionEpsilonPositive'] == True:
                print('  >> epsilon parameters are explicitly requested to be positive.')
            elif dict_keywords['RestrictRegressionEpsilonPositive'] == False:
                print('  >> epsilon parameters are requested to be either positive or negative.')
    elif dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
        if dict_keywords['RegressionEpsilonMethod'] == 'LinearRegression':
            print('>>> epsilon has been assigned using Linear regression from MBD energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Ridge':
            print('>>> epsilon has been assigned using Ridge regression from MBD energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Lasso':
            print('>>> epsilon has been assigned using Lasso regression from MBD energies.')
            if dict_keywords['RestrictRegressionEpsilonPositive'] == True:
                print('  >> epsilon parameters are explicitly requested to be positive.')
            elif dict_keywords['RestrictRegressionEpsilonPositive'] == False:
                print('  >> epsilon parameters are requested to be either positive or negative.')
    elif dict_keywords['fine_tune_epsilon'] == 'RegressionTot':
        if dict_keywords['RegressionEpsilonMethod'] == 'LinearRegression':
            print('>>> epsilon has been assigned using Linear regression from total energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Ridge':
            print('>>> epsilon has been assigned using Ridge regression from total energies.')
        elif dict_keywords['RegressionEpsilonMethod'] == 'Lasso':
            print('>>> epsilon has been assigned using Lasso regression from total energies.')
            if dict_keywords['RestrictRegressionEpsilonPositive'] == True:
                print('  >> epsilon parameters are explicitly requested to be positive.')
            elif dict_keywords['RestrictRegressionEpsilonPositive'] == False:
                print('  >> epsilon parameters are requested to be either positive or negative.')

    print('                                      ')
    print('   Atom type pair   epsilon    epsilon')
    print('                    (origFF)   (newFF)')
    print('--------------------------------------')

    sortedlist_typepairs = []

    for typepair in origFF___typepairs__epsilon:
        sortedlist_typepairs.append( typepair )
    sortedlist_typepairs = sorted(sortedlist_typepairs)

    for typepair in sortedlist_typepairs:
        printf("       %10s  %7.4f    %7.4f \n", typepair, origFF___typepairs__epsilon[typepair], newFF___typepairs__epsilon[typepair])

    print('\n====\n')

    return()


############################################################
# print original charge parameters type-wise
############################################################

def print_charge_params(origFF___type__charge, newFF___type__charge):

    print('Charge parameters:')
    print('------------------')
    print('                  ')

    if dict_keywords['fine_tune_charge'] == 'False':
        print('>>> Original FF parameters have been used for new FF, i.e. charge parameters have not been altered.')
    elif dict_keywords['fine_tune_charge'] == 'Hirshfeld':
        print('>>> Partial charge parameters have been assigned using Hirshfeld charges.')
    elif dict_keywords['fine_tune_charge'] == 'ESP':
        print('>>> Partial charge parameters have been assigned using ESP charges.')
    elif dict_keywords['fine_tune_charge'] == 'RESP':
        print('>>> Partial charge parameters have been assigned using RESP charges.')

    print('                  ')
    print('   Atom type   Charge q   Charge q   Corresponding atoms')
    print('               (origFF)   (newFF)                       ')
    print('--------------------------------------------------------')

    sortedlist_types = []
    charge_sum1 = 0.
    charge_sum2 = 0.

    for atype in origFF___type__charge:
        sortedlist_types.append( atype )
    sortedlist_types = sorted(sortedlist_types)

    for atype in sortedlist_types:
        alist = []
        for i in range(n_atoms):
            if n_atom__type[i+1] == atype:
                alist.append(i+1)
        printf("   %9d   %8.4f   %8.4f   ", atype, origFF___type__charge[atype], newFF___type__charge[atype]); print(alist)
        charge_sum1 += origFF___type__charge[atype] * len(alist)
        charge_sum2 += newFF___type__charge[atype] * len(alist)

    print('-------------------------------------------------------------------')
    printf("Sum of charge: %8.4f   %8.4f\n", charge_sum1, charge_sum2)

    print('\n====\n')

    return()


############################################################
# print original and new fudge factors
############################################################

def print_fudge_factors(f14, f15, dielectric):

    print('Fudge factors:')
    print('--------------')
    print('              ')

    if dict_keywords['fine_tune_Coulomb_fudge_factors'] == False:
        print('>>> Original Coulomb fudge factors have been used for new FF, i.e. fudge factors have not been altered.')
    elif dict_keywords['fine_tune_Coulomb_fudge_factors'] == True:
        print('>>> Fudge factors have been assigned using linear regression from total energies.')
        if dict_keywords['fine_tune_only_f14_and_f15'] == True:
            print('  >> Only fudge factors for 1-4-interactions and 1-5-interactions have been altered.')
        elif dict_keywords['fine_tune_only_f14_and_f15'] == False:
            print('  >> In addition to the fudge factors for 1-4-interactions and 1-5-interactions, the fudge factor for 1-6-interactions or higher has been altered as well by making use of the dielectric constant.')

    print('                  ')
    print('   Fudge factor     origFF      newFF')
    print('-------------------------------------')
    printf('            f14      0.500    %7.3f\n', f14)
    printf('            f15      1.000    %7.3f\n', f15)
    printf('        f16plus      1.000    %7.3f\n', 1./dielectric)
    printf('     dielectric      1.000    %7.3f   ... dielectric = 1 / f16plus\n', dielectric)

    print('\n====\n')


############################################################
############################################################

def write_TINKER_ff_params_file(newFF___classpair__r0, \
                                 origFF___classpair__Kb, \
                                 newFF___classtriple__theta0, \
                                 origFF___classtriple__Ktheta, \
                                 newFF___classquadruple__V1, \
                                 newFF___classquadruple__V2, \
                                 newFF___classquadruple__V3, \
                                 origFF___classquadruple__impV2, \
                                 newFF___classquadruple__impV2, \
                                 newFF___typepairs__sigma, \
                                 newFF___typepairs__epsilon, \
                                 newFF___type__charge, \
                                 f14, f15, dielectric):

    FFfile = open("oplsaa-ffaffurr.prm", 'w')

    FFfile.write("""
      ##############################
      ##                          ##
      ##  Force Field Definition  ##
      ##                          ##
      ##############################


forcefield              OPLS-AA

vdwindex                TYPE
vdwtype                 LENNARD-JONES
radiusrule              GEOMETRIC
radiustype              SIGMA
radiussize              DIAMETER
epsilonrule             GEOMETRIC
torsionunit             0.5
imptorunit              0.5
vdw-14-scale            0.5\n""")

    FFfile.write("chg-14-scale            %.3f\n" % f14)
    FFfile.write("chg-15-scale            %.3f\n" % f15)
    FFfile.write("dielectric              %.3f\n" % dielectric)

    FFfile.write("""
electric                332.06


      #############################
      ##                         ##
      ##  Atom Type Definitions  ##
      ##                         ##
      #############################
\n\n""")

    sortedlist_types = []
    for n_atom,atype in n_atom__type.items():
        sortedlist_types.append( atype )
    sortedlist_types = sorted(set(sortedlist_types))

    for atype in sortedlist_types:
        FFfile.write("atom        %3d  %3d    %-2s    %-26s    %2d   %7.3f    %1d\n" % (atype, type__class[atype], class__symbol[type__class[atype]], str("\"")+type__description[atype]+str("\""), class__atomic[type__class[atype]], class__mass[type__class[atype]], class__valence[type__class[atype]]) )

    FFfile.write("""\n
      #####################################
      ##                                 ##
      ##  Van der Waals Pair Parameters  ##
      ##                                 ##
      #####################################
\n\n""")

    sortedlist_typepairs = []

    for typepair in newFF___typepairs__sigma:
        sortedlist_typepairs.append( typepair )
    sortedlist_typepairs = sorted(sortedlist_typepairs)

    for typepair in sortedlist_typepairs:
        FFfile.write("vdwpr       %3d   %3d         %6.4f     %6.4f\n" % (typepair[0], typepair[1], newFF___typepairs__sigma[typepair], newFF___typepairs__epsilon[typepair]) )

    FFfile.write("""\n
      ##################################
      ##                              ##
      ##  Bond Stretching Parameters  ##
      ##                              ##
      ##################################
\n\n""")

    sortedlist_classpairs = []

    for classpair in newFF___classpair__r0:
        sortedlist_classpairs.append( classpair )
    sortedlist_classpairs = sorted(sortedlist_classpairs)

    for classpair in sortedlist_classpairs:
        FFfile.write("bond        %3d  %3d          %6.2f     %6.4f\n" % (classpair[0], classpair[1], origFF___classpair__Kb[classpair], newFF___classpair__r0[classpair]) )

    FFfile.write("""\n
      ################################
      ##                            ##
      ##  Angle Bending Parameters  ##
      ##                            ##
      ################################
\n\n""")

    sortedlist_classtriples = []

    for classtriple in newFF___classtriple__theta0:
        sortedlist_classtriples.append( classtriple )
    sortedlist_classtriples = sorted(sortedlist_classtriples, key=operator.itemgetter(1,2,0))

    for classtriple in sortedlist_classtriples:
        FFfile.write("angle       %3d  %3d  %3d     %6.2f     %6.2f\n" % (classtriple[0], classtriple[1], classtriple[2], origFF___classtriple__Ktheta[classtriple], newFF___classtriple__theta0[classtriple]) )

    FFfile.write("""\n
      #####################################
      ##                                 ##
      ##  Improper Torsional Parameters  ##
      ##                                 ##
      #####################################
\n\n""")

    sortedlist_classquadruples = []

    for classquadruple in newFF___classquadruple__impV2:
        sortedlist_classquadruples.append( classquadruple )
    sortedlist_classquadruples = sorted(sortedlist_classquadruples, key=operator.itemgetter(2,3,0,1))

    for classquadruple in sortedlist_classquadruples:
        FFfile.write("imptors     %3d  %3d  %3d  %3d          %7.3f  180.0  2\n" % (classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], newFF___classquadruple__impV2[classquadruple]) )

    FFfile.write("""\n
      ############################
      ##                        ##
      ##  Torsional Parameters  ##
      ##                        ##
      ############################
\n\n""")

    sortedlist_classquadruples = []

    for quadruple in list_1234_interacts:

        if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
            classquadruple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
        else:
            classquadruple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )

        sortedlist_classquadruples.append( classquadruple )

    sortedlist_classquadruples = sorted(set(sortedlist_classquadruples), key=operator.itemgetter(1,2,0,3))

    for classquadruple in sortedlist_classquadruples:

        V1 = 0.
        V2 = 0.
        V3 = 0.

        if classquadruple in newFF___classquadruple__V1:
            V1 = newFF___classquadruple__V1[classquadruple]
        if classquadruple in newFF___classquadruple__V2:
            V2 = newFF___classquadruple__V2[classquadruple]
        if classquadruple in newFF___classquadruple__V3:
            V3 = newFF___classquadruple__V3[classquadruple]

        FFfile.write("torsion     %3d  %3d  %3d  %3d    %7.3f 0.0 1 %7.3f 180.0 2 %7.3f 0.0 3\n" % (classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], V1, V2, V3) )

    FFfile.write("""\n
      ########################################
      ##                                    ##
      ##  Atomic Partial Charge Parameters  ##
      ##                                    ##
      ########################################
\n\n""")

    for atype in sortedlist_types:
        FFfile.write("charge      %3d              %7.4f\n" % (atype, newFF___type__charge[atype]) )

    FFfile.close()

    return()


############################################################
############################################################

if __name__ == "__main__":
    main()

