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
import time
from simtk.openmm.app import *
from simtk.openmm.app.internal import *
from simtk.openmm import *
from simtk.unit import *

from scipy.optimize import curve_fit, minimize

from xml.etree import ElementTree as ET

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from pyswarm import pso

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
	
	# get original FF parameters from user provided 'input.pdb'
	#   here ONLY the file input.pdb is used
	#   (OpenMM is called)
	global n_atoms              # number of atoms
	global residues__atoms      # dictionary: residues --> atoms
	global n_atom__type         # dictionary: atom number --> atom type
	global n_atom__class        # dictionary: atom number --> atom class
	global type__class          # dictionary: atom type --> atom class
	global n_atom__atomic       # dictionary: atom number --> atomic number
	global n_atom__mass         # dictionary: atom number --> mass
	global class__atomic        # dictionary: atom class --> atomic number
	global class__mass          # dictionary: atom class --> mass
	global class__element       # dictionary: atom class --> element
	global list_123_interacts   # list of 123 interactions (needed for angle bending calculations)
	global list_1234_interacts  # list of 1234 interactions (needed for torsional angle calculations)
	global list_imp1234_interacts  # list of 1234 improper interactions (needed for improper torsional angle calculations)
	global list_12_interacts
	global list_13_interacts
	global list_14_interacts
	n_atoms, \
	 residues__atoms, \
	 n_atom__type, \
	 n_atom__class, \
	 type__class, \
	 n_atom__atomic, \
	 n_atom__mass, \
	 class__atomic, \
	 class__mass, \
	 class__element, \
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
	 origFF___classquadruple__V4, \
	 list_1234_interacts, \
	 origFF___classquadruple__impV2, \
	 list_imp1234_interacts, \
	 list_12_interacts, \
	 list_13_interacts, \
	 list_14_interacts, \
	 origFF___CN_factor, \
	 origFF___type__zero_transfer_distance, \
	 origFF___type__ChargeTransfer_parameters, \
	 origFF___type__averageAlpha0eff, \
	 origFF___gamma = get_originalFF_params()
	
	# get pair-wise charge parameters (original FF) for Coulomb interactions
	origFF___pairs__charge = get_charge_pairs_params(origFF___type__charge)
	
	# get (type)pair-wise vdW parameters (sigma & epsilon) for original FF
	#   - original FF --> sigma = sqrt(sigma1*sigma2)
	#                     epsilon = sqrt(epsilon1*epsilon2)
	origFF___pairs__sigma, \
	 origFF___pairs__epsilon, \
	 origFF___typepairs__sigma, \
	 origFF___typepairs__epsilon = get_origFF_vdW_pairs_params(origFF___type__sigma, \
																origFF___type__epsilon)

	# test function only used for cross-ckecking with OpenMM
	# --> all energy contributions (bonds, angles, torsions, vdW, Coulomb)
	#     are calculated using a file called 'input.pdb'
	#     --> energies can then be cross-checked with OpenMM
	#cross_check_OpenMM(origFF___classpair__Kb, \
	#					origFF___classpair__r0, \
	#					origFF___classtriple__Ktheta, \
	#					origFF___classtriple__theta0, \
	#					origFF___classquadruple__V1, \
	#					origFF___classquadruple__V2, \
	#					origFF___classquadruple__V3, \
	#					origFF___classquadruple__V4, \
	#					origFF___classquadruple__impV2, \
	#					origFF___pairs__sigma, \
	#					origFF___pairs__epsilon, \
	#					origFF___pairs__charge)

	# get logfiles (FHI-aims output files)
	# from input file 'ffaffurr.input.FHI-aims-logfiles'
	global list_logfiles
	list_logfiles = get_logfiles()
	
	# get xyz for all logfiles
	global listofdicts_logfiles___n_atom__xyz
	global index__listofdicts_logfiles___n_atom__xyz
	listofdicts_logfiles___n_atom__xyz, \
	 index__listofdicts_logfiles___n_atom__xyz= get_logfiles_xyz()
	
	# polarizibility from FHI-aims
	#origFF___type__averageAlpha0eff, origFF___n_atom__alpha0eff = get_alpha0eff()
	
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
	if dict_keywords['fine_tune_charge_transfer'] == False:
		if dict_keywords['fine_tune_charge'] == 'False':
			newFF___type__charge = copy.deepcopy(origFF___type__charge)
		elif ( dict_keywords['fine_tune_charge'] == 'Hirshfeld' ) or ( dict_keywords['fine_tune_charge'] == 'ESP') or ( dict_keywords['fine_tune_charge'] == 'RESP') :
			newFF___type__charge = get_average_charge()
	elif dict_keywords['fine_tune_charge_transfer'] == True:
		newFF___type__charge = get_type__isolated_charge(origFF___type__charge)
	
	
	# get pair-wise charge parameters (new FF) for Coulomb interactions, No charge transfer
	newFF___pairs__charge = get_charge_pairs_params(newFF___type__charge)
	
	# get newFF___type__averageAlpha0eff
	if dict_keywords['fine_tune_polarization_energy'] == True:
		if dict_keywords['fine_tune_polarizabilities'] == False:
			if origFF___type__averageAlpha0eff:
				newFF___type__averageAlpha0eff = copy.deepcopy(origFF___type__averageAlpha0eff)
				newFF___gamma = copy.deepcopy(origFF___gamma)
			else:
				newFF___type__averageAlpha0eff = get_alpha0eff()
				newFF___gamma = 0.92
	elif dict_keywords['fine_tune_polarization_energy'] == False:
		newFF___type__averageAlpha0eff = {}
		newFF___gamma = 0.92
	
	# fine-tune charge transfer (or not)
	if dict_keywords['fine_tune_charge_transfer'] == False:
		newFF___CN_factor = copy.deepcopy(origFF___CN_factor)
		newFF___type__zero_transfer_distance = copy.deepcopy(origFF___type__zero_transfer_distance)
		newFF___type__ChargeTransfer_parameters = copy.deepcopy(origFF___type__ChargeTransfer_parameters)
		fopt = []
	elif dict_keywords['fine_tune_charge_transfer'] == True:
		newFF___CN_factor, \
		 newFF___type__zero_transfer_distance, \
		 newFF___type__ChargeTransfer_parameters, \
		 fopt = PSO_optimize(origFF___classpair__Kb, \
							   newFF___classpair__r0, \
							   origFF___classtriple__Ktheta, \
							   newFF___classtriple__theta0, \
							   origFF___classquadruple__V1, \
							   origFF___classquadruple__V2, \
							   origFF___classquadruple__V3, \
							   origFF___classquadruple__V4, \
							   origFF___classquadruple__impV2, \
							   origFF___pairs__sigma, \
							   origFF___pairs__epsilon, \
							   newFF___pairs__charge, \
							   newFF___type__charge, \
							   newFF___type__averageAlpha0eff, \
							   newFF___gamma)
	
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
	
	# include polarization energy (or not)
	# tune polarizabilities (or not)												  
	if dict_keywords['fine_tune_polarization_energy'] == True:
		if dict_keywords['fine_tune_polarizabilities'] == True:
			newFF___type__averageAlpha0eff, newFF___gamma, fopt_pol = get_polarizabilities_PSO(origFF___classpair__Kb, \
																		newFF___classpair__r0, \
																		origFF___classtriple__Ktheta, \
																		newFF___classtriple__theta0, \
																		origFF___classquadruple__V1, \
																		origFF___classquadruple__V2, \
																		origFF___classquadruple__V3, \
																		origFF___classquadruple__V4, \
																		origFF___classquadruple__impV2, \
																		origFF___pairs__sigma, \
																		origFF___pairs__epsilon, \
																		newFF___pairs__charge, \
																		newFF___type__charge, \
																		newFF___CN_factor, \
																		newFF___type__zero_transfer_distance, \
																		newFF___type__ChargeTransfer_parameters)
	
													  
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
	# bonding terms are of form: 0.5 * kB * (r-r0)**2
	get_bonding_energies(origFF___classpair__Kb, newFF___classpair__r0)
	
	# get angles energies with newFF parameters
	# angles bending terms are of form: 0.5 * Ktheta * (theta-theta0)**2
	get_angles_energies(origFF___classtriple__Ktheta, newFF___classtriple__theta0)
	
	# get torsions energies with origFF parameters
	#   (not really useful at the moment but I put it in just for completeness
	#    and in case I need this function in the future for some reason)
	# -> torsion (dihedral angles) terms are of form:
	#       V1  * ( 1+cos(  phi − θ0) )
	#       V2  * ( 1+cos(2*phi − θ0) )
	#       V3  * ( 1+cos(3*phi − θ0) )
	# θ0 is the phase offset, and k is the force constant
	get_torsions_energies(origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3, origFF___classquadruple__V4, 'Etorsions_(origFF)')
	
	# get improper torsions energies with origFF parameters
	# -> improper torsion terms are of form:
	#       impv2 * (1 + cos (2θ − θ0 ))
	get_improper_energies(origFF___classquadruple__impV2)
	
	# get polarization energies with alpha from FHI-aims
	# -> polarization terms are of form:
	#       -0.5 * dipole * E0
	get_polarization_energies(newFF___type__averageAlpha0eff, newFF___type__charge, newFF___gamma, newFF___CN_factor, newFF___type__zero_transfer_distance, newFF___type__ChargeTransfer_parameters)
	
	# get Coulomb energies with newFF parameters
	# -> Coulomb terms are of form: f * q1*q2 / r12
	#          {   0 for 1-2-interactions and 1-3-interactions
	#       f ={ 0.5 for 1-4-interactions
	#          {   1 for 1-5-interactions and higher
	get_Coulomb_energies(newFF___pairs__charge, newFF___type__charge, newFF___CN_factor, newFF___type__zero_transfer_distance, newFF___type__ChargeTransfer_parameters)
	
	# get Coulomb energy contributions per 1-X-interaction with newFF parameters
	# -> terms are of form: q1*q2 / r12
	get_Coulomb_energyContribsPer_1_X_interaction(newFF___pairs__charge, newFF___type__charge, newFF___CN_factor, newFF___type__zero_transfer_distance, newFF___type__ChargeTransfer_parameters)
	
	# get vdW energies with origFF parameters
	# -> van der Waals (vdW; Lennard-Jones) terms are of form:
	#       4*epsilon * f * [ (sigma/r)**12 - (sigma/r)**6 ]
	#            {   0 for 1-2-interactions and 1-3-interactions
	#         f ={ 1/2 for 1-4-interactions
	#            {   1 for 1-5-interactions and higher
	get_vdW_energies(origFF___pairs__sigma, origFF___pairs__epsilon, 'Evdw_(origFF)')
	
	# do regression for estimating fudge factors in Coulomb energies
	f14 = 0.5 # default
	f15 = 1.0 # default
	if ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == True ):
		f14, f15 = do_regression_Coulomb_fudge_factors() 
		# reset Coulomb energies with new fudge factors
		data['Ecoul_(FF)'] =  (   f14 * data['Ecoul_(FF)_14Intacts'] \
								+ f15 * data['Ecoul_(FF)_15plusIntacts'] )
	
	#    with pandas.option_context('display.max_rows', 999, 'display.max_columns', 999):
	#        print(data.head())
	#        print(data)
    
	# get bonding energy contributions classpair-wise
	#   (not really useful at the moment but I put it in just for completeness
	#    and in case I need this function in the future for some reason)
	# -> bonding contributions are of form: 0.5 * (r-r0)**2
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
	#       ( 1+cos(2*phi − θ0) )
	if dict_keywords['fine_tune_imptorsionalV'] == True:
	    get_impropsEnergyContribs()
	
	# get torsions energy contributions classquadruple-wise
	# torsion (dihedral angles) contributions are of form:
	#       V1  * ( 1+cos(  phi − θ0) )
	#       V2  * ( 1+cos(2*phi − θ0) )
	#       V3  * ( 1+cos(3*phi − θ0) )
	# θ0 is the phase offset, and k is the force constant
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
		 newFF___classquadruple__V3 = do_regression_torsionalV(origFF___classquadruple__V1,origFF___classquadruple__V2, origFF___classquadruple__V3) 
		newFF___classquadruple__V4 = {}
	elif dict_keywords['fine_tune_torsionalV'] == False:
		newFF___classquadruple__V1 = copy.deepcopy(origFF___classquadruple__V1)
		newFF___classquadruple__V2 = copy.deepcopy(origFF___classquadruple__V2)
		newFF___classquadruple__V3 = copy.deepcopy(origFF___classquadruple__V3)
		newFF___classquadruple__V4 = copy.deepcopy(origFF___classquadruple__V4)
		
	# get torsions energies with newFF parameters
	#   (not really useful at the moment but I put it in just for completeness
	#    and in case I need this function in the future for some reason)
	# -> torsion (dihedral angles) terms are of form:
	#       V1  * ( 1+cos(  phi − θ0) )
	#       V2  * ( 1+cos(2*phi − θ0) )
	#       V3  * ( 1+cos(3*phi − θ0) )
	# θ0 is the phase offset, and k is the force constant
	get_torsions_energies(newFF___classquadruple__V1, newFF___classquadruple__V2, newFF___classquadruple__V3, newFF___classquadruple__V4, 'Etorsions_(FF)')	
		
    
	# do regression to estimate improper torsional parameters (impV2)
	if dict_keywords['fine_tune_imptorsionalV'] == True:
		newFF___collect_classquadruple__impV2, \
		  newFF___classquadruple__impV2 = do_regression_imptorsionalV(origFF___classquadruple__impV2) 
	elif dict_keywords['fine_tune_imptorsionalV'] == False:
		newFF___collect_classquadruple__impV2 = {}
		newFF___classquadruple__impV2 = copy.deepcopy(origFF___classquadruple__impV2)
		for atuple in newFF___classquadruple__impV2:
			if atuple[2] == 3:
				newFF___collect_classquadruple__impV2[ (0, 0, atuple[2], atuple[3]) ] = newFF___classquadruple__impV2[ atuple ]
			elif atuple[2] == 24 or atuple[2] == 47 or atuple[2] == 48:
				newFF___collect_classquadruple__impV2[ (0, 0, atuple[2], 0) ] = newFF___classquadruple__impV2[ atuple ]
	
	#print(newFF___classquadruple__impV2)
	#energy_list = get_FF_energy(origFF___classpair__Kb, \
	#				newFF___classpair__r0, \
	#				origFF___classtriple__Ktheta, \
	#				newFF___classtriple__theta0, \
	#				newFF___classquadruple__V1, \
	#				newFF___classquadruple__V2, \
	#				newFF___classquadruple__V3, \
	#				newFF___classquadruple__V4, \
	#				newFF___classquadruple__impV2, \
	#				newFF___pairs__sigma, \
	#				newFF___pairs__epsilon, \
	#				newFF___pairs__charge, \
	#				newFF___type__charge, \
	#				newFF___CN_factor, \
	#				newFF___type__zero_transfer_distance, \
	#				newFF___type__ChargeTransfer_parameters, \
	#				newFF___type__averageAlpha0eff, \
	#				newFF___gamma)
	#print(energy_list)
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
	print_improps_params(origFF___classquadruple__impV2, \
						   newFF___classquadruple__impV2)
	
	# print original and new sigma parameters typepair-wise
	print_sigma_params(origFF___typepairs__sigma, newFF___typepairs__sigma)
	
	# print original and new epsilon parameters typepair-wise
	print_epsilon_params(origFF___typepairs__epsilon, newFF___typepairs__epsilon)
	
	# print original and new charge parameters type-wise
	print_charge_params(origFF___type__charge, newFF___type__charge)
	
	# print original and new fudge factors
	print_fudge_factors(f14, f15)
	
	# print charge transfer parameters
	print_charge_transfer_params(newFF___CN_factor, newFF___type__zero_transfer_distance, newFF___type__ChargeTransfer_parameters, fopt)
	
	# print polarization parameters
	print_polarization_params(newFF___type__averageAlpha0eff, newFF___gamma)
	
	# write OpenMM force field parameter file
	write_OpenMM_ff_params_file(newFF___classpair__r0, \
								 origFF___classpair__Kb, \
								 newFF___classtriple__theta0, \
								 origFF___classtriple__Ktheta, \
								 newFF___classquadruple__V1, \
								 newFF___classquadruple__V2, \
								 newFF___classquadruple__V3, \
								 newFF___classquadruple__V4, \
								 newFF___collect_classquadruple__impV2, \
								 newFF___classquadruple__impV2, \
								 newFF___typepairs__sigma, \
								 newFF___typepairs__epsilon, \
								 newFF___type__charge, \
								 f14, f15)
    
	# write Custombond force parameter file
	write_custombondforce_para(newFF___pairs__sigma, \
								newFF___pairs__epsilon, \
								newFF___CN_factor, \
								f15, \
                               newFF___type__zero_transfer_distance, \
		                        newFF___type__ChargeTransfer_parameters, \
	                            newFF___type__averageAlpha0eff, \
	                            newFF___gamma)
								
	
	#for atupe in list_imp1234_interacts:
	#	print(atupe)
	#	print(n_atom__class[atupe[0]],n_atom__class[atupe[1]], n_atom__class[atupe[2]],n_atom__class[atupe[3]])
	#print(newFF___classquadruple__impV2)
    
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
		astring = get_input_loop_lines(lines, 'fine_tune_only_f14')
		if astring in ['True', 'true']:
			dict_keywords['fine_tune_only_f14'] = True
		elif astring in ['False', 'false']:
			dict_keywords['fine_tune_only_f14'] = False
		else:
			sys.exit('== Error: keyword \'fine_tune_only_f14\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')
			
	astring = get_input_loop_lines(lines, 'fine_tune_charge_transfer')
	if astring in ['True', 'true']:
		dict_keywords['fine_tune_charge_transfer'] = True
	elif astring in ['False', 'false']:
		dict_keywords['fine_tune_charge_transfer'] = False
	else:
		sys.exit('== Error: keyword \'fine_tune_charge_transfer\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')
		
			
	if ( dict_keywords['fine_tune_charge_transfer'] == True ):
		astring = get_input_loop_lines(lines, 'fine_tune_isolated_charges')
		if astring in ['False', 'false']:
			dict_keywords['fine_tune_isolated_charges'] = 'False'
		elif astring in ['Hirshfeld', 'hirshfeld']:
			dict_keywords['fine_tune_isolated_charges'] = 'Hirshfeld'
		elif astring in ['ESP', 'esp']:
			dict_keywords['fine_tune_isolated_charges'] = 'ESP'
		elif astring in ['RESP', 'resp']:
			dict_keywords['fine_tune_isolated_charges'] = 'RESP'
		else:
			sys.exit('== Error: keyword \'fine_tune_isolated_charges\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')
				
	if ( dict_keywords['fine_tune_charge'] != 'False' ) and ( dict_keywords['fine_tune_charge_transfer'] == True ):
		sys.exit('== Error: Average charges and charge transfer can not be used at the same time. Check input file \'ffaffurr.input\'. Exiting now...')
		
	astring = get_input_loop_lines(lines, 'fine_tune_polarization_energy')
	if astring in ['True', 'true']:
		dict_keywords['fine_tune_polarization_energy'] = True
	elif astring in ['False', 'false']:
		dict_keywords['fine_tune_polarization_energy'] = False
	else:
		sys.exit('== Error: keyword \'fine_tune_polarization_energy\' not set correctly. Check input file \'ffaffurr.input\'. Exiting now...')
		
	if ( dict_keywords['fine_tune_polarization_energy'] == True ):
		astring = get_input_loop_lines(lines, 'fine_tune_polarizabilities')
		if astring in ['False', 'false']:
			dict_keywords['fine_tune_polarizabilities'] = False
		elif astring in ['True', 'true']:
			dict_keywords['fine_tune_polarizabilities'] = True
		
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
# get original FF parameters from user provided 'input.pdb'
#   here ONLY the file input.pdb is used
#   (OpenMM is called)
############################################################

def get_originalFF_params():
	n_atoms    = -1
	n_bonds    = -1
	n_angles   = -1
	n_torsions = -1
	n_vdws     = -1
	
	if dict_keywords['readparamsfromffaffurr'] == False:
		choose_forcefield = 'OPLS-AA.xml'
	elif dict_keywords['readparamsfromffaffurr'] == True:
		choose_forcefield = 'ffaffurr-oplsaa.xml'
	# build system with OpenMM
	pdb = PDBFile('input.pdb')
	forcefield = ForceField(choose_forcefield)
	system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
	
	forces = { force.__class__.__name__ : force for force in system.getForces() }
	bondforce     = forces['HarmonicBondForce']
	angleforce    = forces['HarmonicAngleForce']
	torsforce     = forces['PeriodicTorsionForce']                 # include improper
	nonbondforce  = forces['NonbondedForce']
	
	n_atoms    =  system.getNumParticles()
	if not n_atoms > 0:
		sys.exit('== Error: Number of atoms should be a positive number. Exiting now...')
		
	n_bonds    =  bondforce.getNumBonds()
	if not n_bonds > -1:
		sys.exit('== Error: Number of bonds should be a natural number. Exiting now...')
		
	n_angles   =  angleforce.getNumAngles()
	if not n_angles > -1:
		sys.exit('== Error: Number of angles should be a natural number. Exiting now...')
		
	n_torsions = torsforce.getNumTorsions()
	if not n_torsions > -1:
		sys.exit('== Error: Number of torsional angles should be a natural number. Exiting now...')
	
	# data.atoms, data.atomType, data.atomType[atom]
	data = forcefield._SystemData()
	
	# Make a list of all atoms ('name', 'element', 'index', 'residue', 'id')
	data.atoms = list(pdb.topology.atoms())
	
	# Make a list of all bonds
	bondedToAtom = forcefield._buildBondedToAtomList(pdb.topology)
	
	for bond in pdb.topology.bonds():
		data.bonds.append(ForceField._BondData(bond[0].index, bond[1].index))		
	
	# Make a list of all unique angles
	uniqueAngles = set()
	list_123_interacts = []
	for bond in data.bonds:
		for atom in bondedToAtom[bond.atom1]:
			if atom != bond.atom2:
				if atom < bond.atom2:
					uniqueAngles.add((atom, bond.atom1, bond.atom2))
				else:
					uniqueAngles.add((bond.atom2, bond.atom1, atom))
		for atom in bondedToAtom[bond.atom2]:
			if atom != bond.atom1:
				if atom > bond.atom1:
					uniqueAngles.add((bond.atom1, bond.atom2, atom))
				else:
					uniqueAngles.add((atom, bond.atom2, bond.atom1))
	data.angles = sorted(list(uniqueAngles))
	list_123_interacts = sorted(list(uniqueAngles))
	
	# Make a list of all unique proper torsions
	uniquePropers = set()
	list_1234_interacts = []
	for angle in data.angles:
		for atom in bondedToAtom[angle[0]]:
			if atom not in angle:
				if atom < angle[2]:
					uniquePropers.add((atom, angle[0], angle[1], angle[2]))
				else:
					uniquePropers.add((angle[2], angle[1], angle[0], atom))
		for atom in bondedToAtom[angle[2]]:
			if atom not in angle:
				if atom > angle[0]:
					uniquePropers.add((angle[0], angle[1], angle[2], atom))
				else:
					uniquePropers.add((atom, angle[2], angle[1], angle[0]))
	data.propers = sorted(list(uniquePropers))
	list_1234_interacts = sorted(list(uniquePropers))
	
	# data.atomType, residues__atoms
	residues__atoms = {}
	for chain in pdb.topology.chains():
		for res in chain.residues():      
			[template, matches] = forcefield._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=False)
			
			if matches is None:
				raise Exception('User-supplied template does not match the residue %d (%s)' % (res.index+1, res.name)) 
			else:
				data.recordMatchedAtomParameters(res, template, matches)			
			residues__atoms[template.__dict__['name']] = template.__dict__['atoms']
	
	# n_atom__type		
	n_atom__type = {}
	for atom in data.atoms:
		n_atom__type[atom.__dict__['index']] = data.atomType[atom]
	
	# type__class
	type__class = {}
	for atom in data.atoms:
		atomtype = data.atomType[atom]	
		type__class[atomtype] = forcefield._atomTypes[atomtype].__dict__['atomClass']
				
	# n_atom__class
	n_atom__class = {}
	for index in n_atom__type.keys():
		n_atom__class[index] = int(type__class[n_atom__type[index]])
	
	#n_atom__atomic, n_atom__mass, class__atomic, class__mass, class__element
	n_atom__atomic = {}
	n_atom__mass = {}
	class__atomic = {}
	class__mass = {}
	class__element = {}
	for atom in data.atoms:
		n_atom = atom.__dict__['index']
		classes = n_atom__class[n_atom] 
		element = atom.__dict__['element']
		atomic = element.__dict__['_atomic_number']
		mass = element.__dict__['_mass']
		n_atom__atomic[n_atom] = atomic 
		n_atom__mass[n_atom] = mass.__dict__['_value']
		class__atomic[classes] = atomic
		class__mass[classes] = mass.__dict__['_value']
		class__element[classes] = element.__dict__['_symbol']
	
	#if dict_keywords['readparamsfromffaffurr'] == False:
	# origFF___type__charge, origFF___type__sigma(nm), origFF___type__epsilon(kJ/mol)
	type__charge = {}
	type__sigma = {}
	type__epsilon = {} 
	for index in range(nonbondforce.getNumParticles()):
		[charge, sigma, epsilon] = nonbondforce.getParticleParameters(index)
		type = n_atom__type[index]
		type__charge[type] = charge.__dict__['_value']
		type__sigma[type] = sigma.__dict__['_value']
		type__epsilon[type] = epsilon.__dict__['_value']
	
	# origFF___classpair__Kb(kJ/(nm**2 mol)), origFF___classpair__r0(nm)	
	classpair__Kb = {}
	classpair__r0 = {}
	list_12_interacts = []
	for index in range(bondforce.getNumBonds()):
		[particle1, particle2, r0, Kb]= bondforce.getBondParameters(index)
		if n_atom__class[int(particle1)] <= n_atom__class[int(particle2)]:
			aclasspair = ( n_atom__class[int(particle1)],n_atom__class[int(particle2)] )
		else:
			aclasspair = ( n_atom__class[int(particle2)],n_atom__class[int(particle1)] )
		classpair__Kb[ aclasspair ] = float(Kb.__dict__['_value'])
		classpair__r0[ aclasspair ] = float(r0.__dict__['_value'])
		
		if int(particle1) <= int(particle2):
			atuple = (int(particle1),int(particle2))
		else:
			atuple = (int(particle2),int(particle1))
		list_12_interacts.append( atuple )
	
	# origFF___classtriple__Ktheta(kJ/(mol rad**2)), origFF___classtriple__theta0(rad), list_123_interacts
	classtriple__Ktheta = {}
	classtriple__theta0 = {}
	
	for index in range(angleforce.getNumAngles()):
		[particle1, particle2, particle3, theta0, Ktheta]= angleforce.getAngleParameters(index)
		if n_atom__class[int(particle1)] <= n_atom__class[int(particle3)]:
			aclasstriple = ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)] )
		else:
			aclasstriple = ( n_atom__class[int(particle3)],n_atom__class[int(particle2)],n_atom__class[int(particle1)] )
		classtriple__Ktheta[ aclasstriple ] = float(Ktheta.__dict__['_value'])
		classtriple__theta0[ aclasstriple ] = float(theta0.__dict__['_value'])
	
	# origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3, classquadruple__V4, list_imp1234_interacts
	list_imp1234_interacts = []
	classquadruple__V1 = {}
	classquadruple__V2 = {}
	classquadruple__V3 = {}
	classquadruple__V4 = {}
	classquadruple__impV2 = {}
	for index in range(torsforce.getNumTorsions()):
		[particle1, particle2, particle3, particle4, periodicity, phase, k ] = torsforce.getTorsionParameters(index)           # phase(rad), k(kJ/mol)
		# improper
		if particle1 <= particle2:
			atuple1 = (particle1, particle2)
		else:
			atuple1 = (particle2, particle1)
		if particle2 <= particle3:
			atuple2 = (particle2, particle3)
		else:
			atuple2 = (particle3, particle2)
		if particle3 <= particle4:
			atuple3 = (particle3, particle4)
		else:
			atuple3 = (particle4, particle3)
		if (atuple1 or atuple2 or atuple3) not in list_12_interacts:		
			atuple = (int(particle1), int(particle2), int(particle3), int(particle4))		
			list_imp1234_interacts.append( atuple )
			classquadruple__impV2[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
		# proper
		else:
			if periodicity == 1:
				if n_atom__class[int(particle2)] <= n_atom__class[int(particle3)]:
					classquadruple__V1[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
				else:
					classquadruple__V1[ ( n_atom__class[int(particle4)],n_atom__class[int(particle3)],n_atom__class[int(particle2)],n_atom__class[int(particle1)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
			elif periodicity == 2:
				if n_atom__class[int(particle2)] <= n_atom__class[int(particle3)]:
					classquadruple__V2[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
				else:
					classquadruple__V2[ ( n_atom__class[int(particle4)],n_atom__class[int(particle3)],n_atom__class[int(particle2)],n_atom__class[int(particle1)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
			elif periodicity == 3:
				if n_atom__class[int(particle2)] <= n_atom__class[int(particle3)]:
					classquadruple__V3[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
				else:
					classquadruple__V3[ ( n_atom__class[int(particle4)],n_atom__class[int(particle3)],n_atom__class[int(particle2)],n_atom__class[int(particle1)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
			elif periodicity == 4:
				if n_atom__class[int(particle2)] <= n_atom__class[int(particle3)]:
					classquadruple__V4[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
				else:
					classquadruple__V4[ ( n_atom__class[int(particle4)],n_atom__class[int(particle3)],n_atom__class[int(particle2)],n_atom__class[int(particle1)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
	#elif dict_keywords['readparamsfromffaffurr'] == True:
	#	tree = ET.ElementTree(file='ffaffurr-oplsaa.xml')
    #
	#	# origFF___type__charge, origFF___type__sigma(nm), origFF___type__epsilon(kJ/mol)
	#	type__charge = {}
	#	type__sigma = {}
	#	type__epsilon = {}
	#	if tree.getroot().find('NonbondedForce') is not None:
	#		for atom in tree.getroot().find('NonbondedForce').findall('Atom'):
	#			type__charge[atom.attrib['type']] = float(atom.attrib['charge'])
	#			type__sigma[atom.attrib['type']] = float(atom.attrib['sigma'])
	#			type__epsilon[atom.attrib['type']] = float(atom.attrib['epsilon'])
	#	
	#	# origFF___classpair__Kb(kJ/(nm**2 mol)), origFF___classpair__r0(nm)
	#	classpair__Kb = {}
	#	classpair__r0 = {}
	#	list_12_interacts = []
	#	for index in range(bondforce.getNumBonds()):
	#		[particle1, particle2, r0, Kb]= bondforce.getBondParameters(index)
	#		if int(particle1) <= int(particle2):
	#			atuple = (int(particle1),int(particle2))
	#		else:
	#			atuple = (int(particle2),int(particle1))
	#		list_12_interacts.append( atuple )
	#		
	#	if tree.getroot().find('HarmonicBondForce') is not None:
	#		for bond in tree.getroot().find('HarmonicBondForce').findall('Bond'):
	#			if int(bond.attrib['class1']) <= int(bond.attrib['class2']):
	#				aclasspair = ( int(bond.attrib['class1']),int(bond.attrib['class2']) )
	#			else:
	#				aclasspair = ( int(bond.attrib['class2']),int(bond.attrib['class1']) )
	#			classpair__Kb[ aclasspair ] = float(bond.attrib['k'])
	#			classpair__r0[ aclasspair ] = float(bond.attrib['length'])
	#	
	#	# origFF___classtriple__Ktheta(kJ/(mol rad**2)), origFF___classtriple__theta0(rad), list_123_interacts
	#	classtriple__Ktheta = {}
	#	classtriple__theta0 = {}		
	#	if tree.getroot().find('HarmonicAngleForce') is not None:
	#		for angle in tree.getroot().find('HarmonicAngleForce').findall('Angle'):
	#			
	#			if int(angle.attrib['class1']) <= int(angle.attrib['class3']):
	#				aclasstriple = ( int(angle.attrib['class1']), int(angle.attrib['class2']), int(angle.attrib['class3']) )
	#			else:
	#				aclasstriple = ( int(angle.attrib['class3']), int(angle.attrib['class2']), int(angle.attrib['class1']) )
	#			classtriple__Ktheta[aclasstriple] = float(angle.attrib['k'])
	#			classtriple__theta0[aclasstriple] = float(angle.attrib['angle'])
	#	
	#	# origFF___classquadruple__V1, origFF___classquadruple__V2, origFF___classquadruple__V3, classquadruple__V4, list_imp1234_interacts
	#	list_imp1234_interacts = []
	#	classquadruple__V1 = {}
	#	classquadruple__V2 = {}
	#	classquadruple__V3 = {}
	#	classquadruple__V4 = {}
	#	classquadruple__impV2 = {}		
	#	for index in range(torsforce.getNumTorsions()):
	#		[particle1, particle2, particle3, particle4, periodicity, phase, k ] = torsforce.getTorsionParameters(index)           # phase(rad), k(kJ/mol)
	#		# improper
	#		if particle1 <= particle2:
	#			atuple1 = (particle1, particle2)
	#		else:
	#			atuple1 = (particle2, particle1)
	#		if particle2 <= particle3:
	#			atuple2 = (particle2, particle3)
	#		else:
	#			atuple2 = (particle3, particle2)
	#		if particle3 <= particle4:
	#			atuple3 = (particle3, particle4)
	#		else:
	#			atuple3 = (particle4, particle3)
	#		if (atuple1 or atuple2 or atuple3) not in list_12_interacts:		
	#			atuple = (int(particle1), int(particle2), int(particle3), int(particle4))		
	#			list_imp1234_interacts.append( atuple )
	#			#classquadruple__impV2[ ( n_atom__class[int(particle1)],n_atom__class[int(particle2)],n_atom__class[int(particle3)],n_atom__class[int(particle4)] ) ] = [phase.__dict__['_value'], k.__dict__['_value']]
	#		
	#				
	#	if tree.getroot().find('PeriodicTorsionForce') is not None:
	#		for quadruple in tree.getroot().find('PeriodicTorsionForce').findall('Improper'):
	#			classquadruple__impV2[( quadruple.attrib['class2'], quadruple.attrib['class3'], quadruple.attrib['class1'], quadruple.attrib['class4'])] = [ float(quadruple.attrib['phase1']), float(quadruple.attrib['k1'])]
	#		for quadruple in tree.getroot().find('PeriodicTorsionForce').findall('Proper'):
	#			if int(quadruple.attrib['class2']) <= int(quadruple.attrib['class3']):
	#				atuple = ( int(quadruple.attrib['class1']), int(quadruple.attrib['class2']), int(quadruple.attrib['class3']), int(quadruple.attrib['class4']))
	#			else:
	#				atuple = ( int(quadruple.attrib['class4']), int(quadruple.attrib['class3']), int(quadruple.attrib['class2']), int(quadruple.attrib['class1']))
	#				
	#			for key in quadruple.attrib.keys():
	#				if 'periodicity' in key:
	#					num = key[-1]
	#					
	#					if int(quadruple.attrib[key]) == 1:
	#						classquadruple__V1[atuple] = [ float(quadruple.attrib['phase' + str(num)]), float(quadruple.attrib['k' + str(num)])]
	#					elif int(quadruple.attrib[key]) == 2:
	#						classquadruple__V2[atuple] = [ float(quadruple.attrib['phase' + str(num)]), float(quadruple.attrib['k' + str(num)])]
	#					elif int(quadruple.attrib[key]) == 3:
	#						classquadruple__V3[atuple] = [ float(quadruple.attrib['phase' + str(num)]), float(quadruple.attrib['k' + str(num)])]
	#					elif int(quadruple.attrib[key]) == 4:
	#						classquadruple__V4[atuple] = [ float(quadruple.attrib['phase' + str(num)]), float(quadruple.attrib['k' + str(num)])]		
	#
	list_13_interacts = []
	list_14_interacts = []
	for index in range(nonbondforce.getNumExceptions()):
		[particle1, particle2, chargeprod, sigma, epsilon] =  nonbondforce.getExceptionParameters(index)
		if particle1 <= particle2:
			atuple = (particle1, particle2)
		else:
			atuple = (particle2, particle1)
		if (chargeprod.__dict__['_value'] == 0) and (epsilon.__dict__['_value'] == 0) and (atuple not in list_12_interacts):          # list_12, list_13
			list_13_interacts.append(atuple)
		else: #(chargeprod.__dict__['_value'] != 0) or (epsilon.__dict__['_value'] != 0):
			list_14_interacts.append(atuple)
			
	if dict_keywords['readparamsfromffaffurr'] == False:
		CN_factor = 0
		type__zero_transfer_distance = {}
		type__ChargeTransfer_parameters = {}
		type__averageAlpha0eff = {}
		gamma = 0.92
	elif dict_keywords['readparamsfromffaffurr'] == True:
		type__ChargeTransfer_parameters = {}
		type__zero_transfer_distance = {}
		type__averageAlpha0eff = {}
		gamma = 0.92
		
		tree = ET.ElementTree(file='CustomForce.xml')
		
		if tree.getroot().find('CustomChargeTransfer') is not None:
			CN_factor = str(tree.getroot().find('CustomChargeTransfer').attrib['CN_factor'])
		for atom in tree.getroot().find('CustomChargeTransfer').findall('Atom'):
			type__ChargeTransfer_parameters[atom.attrib['type']] = [float(atom.attrib['a']), float(atom.attrib['b'])]
			type__zero_transfer_distance[atom.attrib['type']] = float(atom.attrib['r'])
			
		if tree.getroot().find('CustomPoleForce') is not None:
			gamma = float(tree.getroot().find('CustomPoleForce').attrib['gamma'])
			for atom in tree.getroot().find('CustomPoleForce').findall('Polarize'):
				type__averageAlpha0eff[ atom.attrib['type'] ] = float( atom.attrib['polarizability'] )	
	
	return(n_atoms, \
			residues__atoms, \
			n_atom__type, \
			n_atom__class, \
			type__class, \
			n_atom__atomic, \
			n_atom__mass, \
			class__atomic, \
			class__mass, \
			class__element, \
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
			classquadruple__V4, \
			list_1234_interacts, \
			classquadruple__impV2, \
			list_imp1234_interacts, \
			list_12_interacts, \
			list_13_interacts, \
			list_14_interacts, \
			CN_factor, \
			type__zero_transfer_distance, \
			type__ChargeTransfer_parameters,\
			type__averageAlpha0eff, \
			gamma)

############################################################
# get pair-wise charge parameters for Coulomb interactions
############################################################

def get_charge_pairs_params(type__charge):

	pairs__charge = {}
	
	# loop over all atom pairs, exclude 1-2 and 1-3 interactions
	for i in range(n_atoms):
		for j in range(i+1, n_atoms):
			if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):
				pairs__charge[ (i,j) ] = type__charge[ n_atom__type[i] ] * type__charge[ n_atom__type[j] ]
	
	return(pairs__charge)

############################################################
# get (type)pair-wise vdW parameters (sigma & epsilon) for original FF
#   - original FF --> sigma = sqrt(sigma1*sigma2)
#                     epsilon = sqrt(epsilon1*epsilon2)
############################################################
	
def get_origFF_vdW_pairs_params(type__sigma, type__epsilon):

	pairs__sigma = {}
	pairs__epsilon = {}
	typepairs__sigma = {}
	typepairs__epsilon = {}
	
	if dict_keywords['readparamsfromffaffurr'] == False:
		# loop over all atom pairs, exclude 1-2 and 1-3 interactions
		for i in range(n_atoms):
			for j in range(i+1, n_atoms):
				if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):					
						pairs__sigma[ (i,j) ] = math.sqrt( type__sigma[ n_atom__type[i] ] * type__sigma[ n_atom__type[j] ] )
						pairs__epsilon[ (i,j) ] = math.sqrt( type__epsilon[ n_atom__type[i] ] * type__epsilon[ n_atom__type[j] ] )
						if n_atom__type[i] <= n_atom__type[j]:
							typepairs__sigma[ ( int(n_atom__type[i]),int(n_atom__type[j]) ) ] = pairs__sigma[ (i,j) ]
							typepairs__epsilon[ ( int(n_atom__type[i]),int(n_atom__type[j]) ) ] = pairs__epsilon[ (i,j) ]
						else:
							typepairs__sigma[ ( int(n_atom__type[j]),int(n_atom__type[i]) ) ] = pairs__sigma[ (i,j) ]
							typepairs__epsilon[ ( int(n_atom__type[j]),int(n_atom__type[i]) ) ] = pairs__epsilon[ (i,j) ]
							
	elif dict_keywords['readparamsfromffaffurr'] == True:
		tree = ET.ElementTree(file='CustomForce.xml')
	
		if tree.getroot().find('CustomBondForce') is not None:
			for bond14 in tree.getroot().find('CustomBondForce').findall('Bond14'):
				if int(bond14.attrib['atom1']) <= int(bond14.attrib['atom2']):
					pair = ( int(bond14.attrib['atom1']), int(bond14.attrib['atom2']) )
				else:
					pair = ( int(bond14.attrib['atom2']), int(bond14.attrib['atom1']) )
				pairs__sigma[ pair ] = float(bond14.attrib['sigma'])
				pairs__epsilon [ pair ] = float(bond14.attrib['epsilon'])
			for bond15 in tree.getroot().find('CustomBondForce').findall('Bond15'):
				if int(bond15.attrib['atom1']) <= int(bond15.attrib['atom2']):
					pair = ( int(bond15.attrib['atom1']), int(bond15.attrib['atom2']) )
				else:
					pair = ( int(bond15.attrib['atom2']), int(bond15.attrib['atom1']) )
				pairs__sigma[ pair ] = float(bond15.attrib['sigma'])
				pairs__epsilon [ pair ] = float(bond15.attrib['epsilon'])
		for pair in pairs__sigma.keys():		
			if n_atom__type[pair[0]] <= n_atom__type[pair[1]]:
				typepairs__sigma[ ( int(n_atom__type[pair[0]]),int(n_atom__type[pair[1]]) ) ] = pairs__sigma[ pair ]
				typepairs__epsilon[ ( int(n_atom__type[pair[0]]),int(n_atom__type[pair[1]]) ) ] = pairs__epsilon[ pair ]
			else:
				typepairs__sigma[ ( int(n_atom__type[pair[1]]),int(n_atom__type[pair[0]]) ) ] = pairs__sigma[ pair ]
				typepairs__epsilon[ ( int(n_atom__type[pair[1]]),int(n_atom__type[pair[0]]) ) ] = pairs__epsilon[ pair ]	
						
	return(pairs__sigma, \
			pairs__epsilon, \
			typepairs__sigma, \
			typepairs__epsilon)

############################################################
# get xyz from one pdf file called 'input.pdb'
############################################################

def get_openmm_xyz():

	# check if file is there
	if not os.path.exists('input.pdb'):
		sys.exit('== Error: Input file \'input.pdb\' does not exist. Exiting now...')
	else:
		print('Now reading from input file \'input.pdb\'...\n')
	
	# read file
	in_file = open("input.pdb", 'r')
	file_lines = in_file.readlines()
	lines = list(file_lines)
	in_file.close()
	
	n_atom__xyz = {}
	
	for line in lines:
		coords_found = re.match(r'(\s*(\w+)\s*(\d+)\s*(\w+\d*)\s*(\w+)\s*(\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*?)', line)
		if coords_found:
			stringa, n_atom, element, residue, n_residue, x, y, z, int1, int2, symbol = line.split(None)
			n_atom__xyz[int(n_atom)-1] = [float(x), float(y), float(z)]
	
	return(n_atom__xyz)

############################################################
# get distances between any two atoms
############################################################
	
def get_distances(n_atom__xyz):

	pairs__distances = {}
	
	for i in range(n_atoms):
		for j in range(i+1, n_atoms):
			distance = math.sqrt(   ( n_atom__xyz[i][0] - n_atom__xyz[j][0] ) ** 2. \
								+ ( n_atom__xyz[i][1] - n_atom__xyz[j][1] ) ** 2. \
								+ ( n_atom__xyz[i][2] - n_atom__xyz[j][2] ) ** 2. )
			pairs__distances[ (i,j) ] = distance * 0.1                     # angstrom to nm
	return(pairs__distances)

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
	
	theta =  math.acos(costheta)
	
	return(theta)

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
	
	phi =  math.acos(cosphi)
	
	sinphi = ( x32*x1234 + y32*y1234 + z32*z1234 ) / ( r32*r123*r234 )
	
	if sinphi < 0.: phi = -phi
	
	return(phi)
 
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
		
		Ebonds += 0.5 * Kb * (r-r0)**2.
		
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
		
		Eangles += 0.5 * Ktheta  * (theta-theta0)**2.
		
	return(Eangles)
	
############################################################
# get torsions bending energy
############################################################

def get_torsions_energy(quadruples__torsions, \
						 classquadruple__V1, classquadruple__V2, classquadruple__V3, classquadruple__V4):

	Etorsions = 0.
	
	for quadruple in list_1234_interacts:

		phi = quadruples__torsions[quadruple]
		
		if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
			atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
		else:
			atuple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )
		
		if atuple in classquadruple__V1:
			[phase, V1] = classquadruple__V1[atuple]
			Etorsions += V1  * ( 1. + math.cos(  1 * phi - phase ) )
		if atuple in classquadruple__V2:
			[phase, V2] = classquadruple__V2[atuple]
			Etorsions += V2 * ( 1. + math.cos(  2 * phi - phase ) )
		if atuple in classquadruple__V3:
			[phase, V3] = classquadruple__V3[atuple]
			Etorsions += V3 * ( 1. + math.cos(  3 * phi - phase ) )
		if atuple in classquadruple__V4:
			[phase, V4] = classquadruple__V3[atuple]
			Etorsions += V4 * ( 1. + math.cos(  4 * phi - phase ) )
			
	return(Etorsions)
	
############################################################
# get improper torsions energy
############################################################

def get_improps_energy(quadruples__improps, classquadruple__impV2):

	Eimprops = 0.
	
	for quadruple in list_imp1234_interacts:

		phi = quadruples__improps[quadruple]
		
		atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
		
		[phase, impV2] = classquadruple__impV2[atuple]
		
		Eimprops += impV2 * ( 1. + math.cos(  2 * phi - phase ))
		
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

def get_Coulomb_energy(pairs__distances, pairs__charge, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):
	
	qqr2kjpermole = 138.935456
	
	Ecoul = 0.
	
	if type__ChargeTransfer_parameters:
		# ion_index, polar_index
		polar_index = []
		n_atom__charge = {}
		for i in range(n_atoms):
			n_atom__charge[i] = type__charge[ n_atom__type[i] ]
			if class__element[n_atom__class[i]] == 'Zn' or class__element[n_atom__class[i]] == 'Na':
				ion_index = i
			elif class__element[n_atom__class[i]] == 'S' or class__element[n_atom__class[i]] == 'O' or class__element[n_atom__class[i]] == 'N':
				polar_index.append(i)
		
		cation_pairs = []		
		for i in polar_index:
			if i < ion_index:
				cation_pairs.append((i, ion_index))
			else:
				cation_pairs.append((ion_index, i))
				
		# get CN
		CN = get_CN(pairs__distances, cation_pairs, type__zero_transfer_distance)
		
		total_transq = 0	
		for pair in cation_pairs:
			if n_atom__type[pair[0]] in type__ChargeTransfer_parameters.keys():
				if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[0]]]:
					if (CN_factor == 0):
						transq = type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1]
					else:
						transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
					total_transq += transq
					n_atom__charge[pair[0]] += transq
			elif n_atom__type[pair[1]] in type__ChargeTransfer_parameters.keys():
				if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[1]]]:
					if (CN_factor == 0):
						transq = type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1]
					else:
						transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
					total_transq += transq
					n_atom__charge[pair[1]] += transq
		
		n_atom__charge[ion_index] = n_atom__charge[ion_index] - total_transq
		
		for pair,qq in pairs__charge.items():
			if pair in list_14_interacts:
				f = 0.5
			else:
				f = 1.
			r = pairs__distances[pair]
			
			Ecoul += f * qqr2kjpermole * n_atom__charge[pair[0]] * n_atom__charge[pair[1]] / r
		
	else:
		for pair,qq in pairs__charge.items():
			if pair in list_14_interacts:
				f = 0.5
			else:
				f = 1.
			r = pairs__distances[pair]
			Ecoul += f * qqr2kjpermole * qq / r
		
	return(Ecoul)

#########################################################################
# connvert amber combination rule of vdW pair-wise parameters to OPLS-AA
#########################################################################

def OPLS_LJ(system):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    lorentz = CustomNonbondedForce(
        '4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    system.addForce(lorentz)
    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        LJset[index] = (sigma, epsilon)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(
            index, charge, sigma, epsilon * 0)
    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED
        # FORCE
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            #print (p1,p2,sig,eps)
            sig14 = sqrt(LJset[p1][0] * LJset[p2][0])
            eps14 = sqrt(LJset[p1][1] * LJset[p2][1])
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)  #eps = fudge_factor * esp14
    #for i in range(nonbonded_force.getNumExceptions()):
    #    (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
    #    print(p1, p2, q, sig, eps)
    return system


############################################################
# test function only used for cross-ckecking with OpenMM
# --> all energy contributions (bonds, angles, torsions, impropers, vdW, Coulomb)
#     are calculated using a file called 'input.pdb'
#     --> energies can then be cross-checked with OpenMM
############################################################

def get_energy_decomposition(pdbfile):
	
	# build system with OpenMM
	pdb = PDBFile(pdbfile)
	forcefield = ForceField('amber03.xml')
	system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
	
	for i in range(system.getNumForces()):
		force = system.getForce(i)
		force.setForceGroup(i)
			
	integrator = VerletIntegrator(0.002*picoseconds)
	simulation = Simulation(pdb.topology, system, integrator)
	
	simulation.context.setPositions(pdb.positions)
	
	energy__decomposition = {}
	for i in range(system.getNumForces()):
		force = system.getForce(i)
		energy = simulation.context.getState(getEnergy=True, groups=1<<i).getPotentialEnergy()
		energy__decomposition[force.__class__.__name__ ] = energy
	return(energy__decomposition)

#############################################################
# get energies list of training conformers from force field(kcal/mol)
#############################################################

def get_FF_energy(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					CN_factor, \
					type__zero_transfer_distance, \
					type__ChargeTransfer_parameters, \
					type__averageAlpha0eff, \
					gamma):
	
	type__averageR0eff = get_R0eff()
						
	index__Energies_FF = {}
	for index, n_atom__xyz in index__listofdicts_logfiles___n_atom__xyz.items():
		
		index = '{:04d}'.format(int(index))
		#print(index)
		
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
		#print('Ebonds: ', Ebonds/4.184)
		
		# get angle bending energy
		Eangles = get_angles_energy(triples__angles, classtriple__Ktheta, classtriple__theta0)
		#print('Eangles: ', Eangles/4.184)
		
		# get torsions bending energy
		Etorsions = get_torsions_energy(quadruples__torsions, \
										classquadruple__V1, classquadruple__V2, classquadruple__V3, classquadruple__V4)
										
		#print('Etorsions: ', Etorsions/4.184)
		
		# get improper torsions energy
		Eimprops = get_improps_energy(quadruples__improps, classquadruple__impV2)
		#print('Eimprops: ', Eimprops/4.184)
		
		# get total torsions energy
		Etotorsion = Etorsions + Eimprops
		#print('Etottorsion: ', Etotorsion/4.184)
		
		# get vdW energy
		Evdw = get_vdW_energy(pairs__distances, pairs__sigma, pairs__epsilon)
		#print('Evdw: ', Evdw/4.184)
		
		# get Coulomb energy
		Ecoul = get_Coulomb_energy(pairs__distances, pairs__charge, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
		#print('Ecoul', Ecoul/4.184)
		
		# get Nonbonded energy
		Enonb = Evdw + Ecoul
		
		# get polarization energy
		Epol = get_polarization_energy(n_atom__xyz, pairs__distances, type__averageAlpha0eff, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
		#print('Epol: ', Epol/4.184)
		
		Energy_FF = Ebonds + Eangles + Etotorsion + Enonb + Epol
		#print('FF : ', Energy_FF/4.184)
		
		index__Energies_FF[index] = Energy_FF/4.184          #kJ/mol > kcal/mol
		
	return(index__Energies_FF)

##########################################################
#  get MAE of training data between DFT energie and FF energies
###########################################################
def get_MAE(classpair__Kb, \
			 classpair__r0, \
			 classtriple__Ktheta, \
			 classtriple__theta0, \
			 classquadruple__V1, \
			 classquadruple__V2, \
			 classquadruple__V3, \
			 classquadruple__V4, \
			 classquadruple__impV2, \
			 pairs__sigma, \
			 pairs__epsilon, \
			 pairs__charge, \
			 type__charge, \
			 CN_factor, \
			 type__zero_transfer_distance, \
			 type__ChargeTransfer_parameters, \
			 type__averageAlpha0eff, \
			 gamma):
				 
	index__Energies_FF = get_FF_energy(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					CN_factor, \
					type__zero_transfer_distance, \
					type__ChargeTransfer_parameters,\
					type__averageAlpha0eff, \
					gamma)
	
	index__Energies_DFT = {}
	with open(os.path.join(os.getcwd(), 'energies.pbe+vdw_training.kcal'), 'r') as DFT_energy:
		lines = DFT_energy.readlines()
		
		for line in lines:
			index = line.split(None, 1)[0]
			index__Energies_DFT[index] = float(line.split(None, 1)[1])
		
	#MAE
	x_data = []
	y_data = []
	
	for key in index__Energies_FF.keys():
		x_data.append(index__Energies_FF[key])
		y_data.append(index__Energies_DFT[key])
		
	# l1-norm error function
	def fs(a):
		return numpy.sum(abs(x-y+a[0]) for x,y in zip(x_data,y_data))
		
	x_data = numpy.array(x_data)
	y_data = numpy.array(y_data)
	
	#l1-norm
	B = minimize(fs,numpy.array([0.0]),method='Nelder-Mead')
	rtmp = x_data[:] - y_data[:] + B.x 
	
	MAE = float(fs([B.x])/rtmp.shape[0])
	return(MAE)

  
def cross_check_OpenMM(classpair__Kb, \
						classpair__r0, \
						classtriple__Ktheta, \
						classtriple__theta0, \
						classquadruple__V1, \
						classquadruple__V2, \
						classquadruple__V3, \
						classquadruple__V4, \
						classquadruple__impV2, \
						pairs__sigma, \
						pairs__epsilon, \
						pairs__charge):

	print('Cross-checking all energy contributions on a provided file called \'input.pdb\'.')
	
	# get xyz from a pdb file called 'input.pdb'
	n_atom__xyz = get_openmm_xyz()
	
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
									classquadruple__V1, classquadruple__V2, classquadruple__V3, classquadruple__V4)
	printf("  Torsional Angle:       %9.4f\n", Etorsions)
	
	# get improper torsions energy
	Eimprops = get_improps_energy(quadruples__improps, classquadruple__impV2)
	printf(" Improper Torsion:       %9.4f\n", Eimprops)
	
	# get total torsions energy
	Etotorsion = Etorsions + Eimprops
	printf("    Total Torsion:       %9.4f\n", Etotorsion)
	
	# get vdW energy
	Evdw = get_vdW_energy(pairs__distances, pairs__sigma, pairs__epsilon)
	printf("    Van der Waals:       %9.4f\n", Evdw)
	
	# get Coulomb energy
	Ecoul = get_Coulomb_energy(pairs__distances, pairs__charge)
	printf("    Charge-Charge:       %9.4f\n", Ecoul)
	
	# get Nonbonded energy
	Enonb = Evdw + Ecoul
	printf("        Nonbonded:       %9.4f\n", Enonb)
	
	print('\n====\n')
	
	# energy decomposition caculated from OpenMM 
	# build system with OpenMM
	pdb = PDBFile('input.pdb')
	forcefield = ForceField('OPLS-AA.xml')
	system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
	
	system = OPLS_LJ(system)
	
	for i in range(system.getNumForces()):
		force = system.getForce(i)
		force.setForceGroup(i)
			
	integrator = VerletIntegrator(0.002*picoseconds)
	simulation = Simulation(pdb.topology, system, integrator)
	
	simulation.context.setPositions(pdb.positions)
	
	energy__decomposition = {}
	for i in range(system.getNumForces()):
		force = system.getForce(i)
		energy = simulation.context.getState(getEnergy=True, groups=1<<i).getPotentialEnergy()
		#energy__decomposition[force.__class__.__name__ ] = energy
		print(force.__class__.__name__, energy )
	
	return()

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
		n_atom__xyz[int(n_atom.rstrip(":"))-1] = [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
	
	return(n_atom__xyz)

############################################################
# get xyz for all logfiles
############################################################

def get_logfiles_xyz():

	listofdicts_logfiles___n_atom__xyz = []
	index__listofdicts_logfiles___n_atom__xyz = {}
	
	for logfile in list_logfiles:
		n_atom__xyz = get_fhiaims_xyz(logfile)
		listofdicts_logfiles___n_atom__xyz.append( n_atom__xyz )
		
		index = logfile.rsplit('/', 2)[1]
		index__listofdicts_logfiles___n_atom__xyz[index] = n_atom__xyz
	
	return(listofdicts_logfiles___n_atom__xyz, \
	       index__listofdicts_logfiles___n_atom__xyz)

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
		
		for line in lines:
			if string in line:
				charge = float(line.rsplit(None, 1)[-1])
				list_charges.append( charge )
		
		# FIXBUG in FHI-aims
		# ESP charges are of opposite charge in some FHI-aims versions, please check
		#if dict_keywords['fine_tune_charge'] == 'ESP': list_charges = [ -x for x in list_charges ]
		
		if len(list_charges) == 0:
			sys.exit('== Error: No charge information found in '+logfile+'. Exiting now...')
		elif len(list_charges) > n_atoms:
			#sys.exit('== Error in '+logfile+': Too much charge information found. Single point energy calculations only! Exiting now...')
			list_charges = list_charges[-n_atoms:]
		#else:
	for i in range(n_atoms):
		n_atom__charges[i] = list_charges[i]
	
	return(n_atom__charges)

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
			type__collectedCharges[ n_atom__type[i] ].append( n_atom__charges[i] )
	
	type__averageCharge = {}
	for atype in sortedlist_types:
		type__averageCharge[atype] = numpy.mean(type__collectedCharges[ atype ])
	
	return(type__averageCharge)
	

############################################################
# get R0free atom-wise
############################################################

def get_R0free():

	n_atom__R0free = {}
	
	for i in range(n_atoms):
		if n_atom__atomic[i] == 1:
			n_atom__R0free[i] = 3.1000 * 0.5291772
		elif n_atom__atomic[i] == 2:     
			n_atom__R0free[i] = 2.6500 * 0.5291772
		elif n_atom__atomic[i] == 3:  
			n_atom__R0free[i] = 4.1600 * 0.5291772
		elif n_atom__atomic[i] == 4:  
			n_atom__R0free[i] = 4.1700 * 0.5291772
		elif n_atom__atomic[i] == 5:  
			n_atom__R0free[i] = 3.8900 * 0.5291772
		elif n_atom__atomic[i] == 6:  
			n_atom__R0free[i] = 3.5900 * 0.5291772
		elif n_atom__atomic[i] == 7:  
			n_atom__R0free[i] = 3.3400 * 0.5291772
		elif n_atom__atomic[i] == 8:  
			n_atom__R0free[i] = 3.1900 * 0.5291772
		elif n_atom__atomic[i] == 9:  
			n_atom__R0free[i] = 3.0400 * 0.5291772
		elif n_atom__atomic[i] == 10: 
			n_atom__R0free[i] = 2.9100 * 0.5291772
		elif n_atom__atomic[i] == 11: 
			n_atom__R0free[i] = 3.7300 * 0.5291772
		elif n_atom__atomic[i] == 12: 
			n_atom__R0free[i] = 4.2700 * 0.5291772
		elif n_atom__atomic[i] == 13: 
			n_atom__R0free[i] = 4.3300 * 0.5291772
		elif n_atom__atomic[i] == 14: 
			n_atom__R0free[i] = 4.2000 * 0.5291772
		elif n_atom__atomic[i] == 15: 
			n_atom__R0free[i] = 4.0100 * 0.5291772
		elif n_atom__atomic[i] == 16: 
			n_atom__R0free[i] = 3.8600 * 0.5291772
		elif n_atom__atomic[i] == 17: 
			n_atom__R0free[i] = 3.7100 * 0.5291772
		elif n_atom__atomic[i] == 18: 
			n_atom__R0free[i] = 3.5500 * 0.5291772
		elif n_atom__atomic[i] == 19: 
			n_atom__R0free[i] = 3.7100 * 0.5291772
		elif n_atom__atomic[i] == 20: 
			n_atom__R0free[i] = 4.6500 * 0.5291772
		elif n_atom__atomic[i] == 21: 
			n_atom__R0free[i] = 4.5900 * 0.5291772
		elif n_atom__atomic[i] == 22: 
			n_atom__R0free[i] = 4.5100 * 0.5291772
		elif n_atom__atomic[i] == 23: 
			n_atom__R0free[i] = 4.4400 * 0.5291772
		elif n_atom__atomic[i] == 24: 
			n_atom__R0free[i] = 3.9900 * 0.5291772
		elif n_atom__atomic[i] == 25: 
			n_atom__R0free[i] = 3.9700 * 0.5291772
		elif n_atom__atomic[i] == 26: 
			n_atom__R0free[i] = 4.2300 * 0.5291772
		elif n_atom__atomic[i] == 27: 
			n_atom__R0free[i] = 4.1800 * 0.5291772
		elif n_atom__atomic[i] == 28: 
			n_atom__R0free[i] = 3.8200 * 0.5291772
		elif n_atom__atomic[i] == 29: 
			n_atom__R0free[i] = 3.7600 * 0.5291772
		elif n_atom__atomic[i] == 30: 
			n_atom__R0free[i] = 4.0200 * 0.5291772
		elif n_atom__atomic[i] == 31: 
			n_atom__R0free[i] = 4.1900 * 0.5291772
		elif n_atom__atomic[i] == 32: 
			n_atom__R0free[i] = 4.2000 * 0.5291772
		elif n_atom__atomic[i] == 33: 
			n_atom__R0free[i] = 4.1100 * 0.5291772
		elif n_atom__atomic[i] == 34: 
			n_atom__R0free[i] = 4.0400 * 0.5291772
		elif n_atom__atomic[i] == 35: 
			n_atom__R0free[i] = 3.9300 * 0.5291772
		elif n_atom__atomic[i] == 36: 
			n_atom__R0free[i] = 3.8200 * 0.5291772
		else:
			sys.exit('== Error: Current implementation of free vdW radii only for elements up to Kr. Exiting now...')
	# bohr > angstrom
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
		n_atom__free_atom_volume[i] = alist_free_atom_volumes[i]
		n_atom__hirshfeld_volume[i] = alist_hirshfeld_volumes[i]
	
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
# get R0eff
############################################################

def get_R0eff():
	
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
			R0eff = ( ( n_atom__hirshfeld_volume[i] / n_atom__free_atom_volume[i] )**(1./3.) ) * n_atom__R0free[i]
			type__collectedR0eff[ n_atom__type[i] ].append( R0eff )
	
	type__averageR0eff = {}
	
	# average R0eff
	for atype in sortedlist_types:
		type__averageR0eff[atype] = numpy.mean(type__collectedR0eff[ atype ])
	
	return(type__averageR0eff)

############################################################
# fine-tune and get (type)pair-wise sigmas from TS (using R0eff)
############################################################

def get_sigmas_TS():

	type__averageR0eff = get_R0eff()
	
	pairs__sigma = {}
	typepairs__sigma = {}
	
	# loop over all atom pairs, exclude 1-2 and 1-3 interactions
	for i in range(n_atoms):
		for j in range(i+1, n_atoms):
			if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):
	
				pairs__sigma[ (i,j) ] = ( type__averageR0eff[ n_atom__type[i] ] + type__averageR0eff[ n_atom__type[j] ] ) * (2.**(-1./6.)) * (0.1)   # angstrom to nm
	
				if n_atom__type[i] <= n_atom__type[j]:
					typepairs__sigma[ ( int(n_atom__type[i]),int(n_atom__type[j]) ) ] = pairs__sigma[ (i,j) ]
				else:
					typepairs__sigma[ ( int(n_atom__type[j]),int(n_atom__type[i]) ) ] = pairs__sigma[ (i,j) ]
	
	return(pairs__sigma, \
			typepairs__sigma)

############################################################
# get C6free, alpha0free atom-wise
############################################################

def get_C6free_alpha0free():

	n_atom__C6free = {}
	n_atom__alpha0free = {}
	
	for i in range(n_atoms):
		if n_atom__atomic[i] == 1:
			n_atom__C6free[i] =    6.5000 ; n_atom__alpha0free[i] =   4.5000
		elif n_atom__atomic[i] == 2:                                                                            
			n_atom__C6free[i] =    1.4600 ; n_atom__alpha0free[i] =   1.3800
		elif n_atom__atomic[i] == 3:                                                                            
			n_atom__C6free[i] = 1387.0000 ; n_atom__alpha0free[i] = 164.2000
		elif n_atom__atomic[i] == 4:                                                                          
			n_atom__C6free[i] =  214.0000 ; n_atom__alpha0free[i] =  38.0000
		elif n_atom__atomic[i] == 5:                                                                          
			n_atom__C6free[i] =   99.5000 ; n_atom__alpha0free[i] =  21.0000
		elif n_atom__atomic[i] == 6:                                                                          
			n_atom__C6free[i] =   46.6000 ; n_atom__alpha0free[i] =  12.0000
		elif n_atom__atomic[i] == 7:                                                                          
			n_atom__C6free[i] =   24.2000 ; n_atom__alpha0free[i] =   7.4000
		elif n_atom__atomic[i] == 8:                                                                          
			n_atom__C6free[i] =   15.6000 ; n_atom__alpha0free[i] =   5.4000
		elif n_atom__atomic[i] == 9:                                                                          
			n_atom__C6free[i] =    9.5200 ; n_atom__alpha0free[i] =   3.8000
		elif n_atom__atomic[i] == 10:                                                                         
			n_atom__C6free[i] =    6.3800 ; n_atom__alpha0free[i] =   2.6700
		elif n_atom__atomic[i] == 11:                                                                         
			n_atom__C6free[i] = 1556.0000 ; n_atom__alpha0free[i] = 162.7000
		elif n_atom__atomic[i] == 12:                                                                         
			n_atom__C6free[i] =  627.0000 ; n_atom__alpha0free[i] =  71.0000
		elif n_atom__atomic[i] == 13:                                                                         
			n_atom__C6free[i] =  528.0000 ; n_atom__alpha0free[i] =  60.0000
		elif n_atom__atomic[i] == 14:                                                                         
			n_atom__C6free[i] =  305.0000 ; n_atom__alpha0free[i] =  37.0000
		elif n_atom__atomic[i] == 15:                                                                         
			n_atom__C6free[i] =  185.0000 ; n_atom__alpha0free[i] =  25.0000
		elif n_atom__atomic[i] == 16:                                                                         
			n_atom__C6free[i] =  134.0000 ; n_atom__alpha0free[i] =  19.6000
		elif n_atom__atomic[i] == 17:                                                                         
			n_atom__C6free[i] =   94.6000 ; n_atom__alpha0free[i] =  15.0000
		elif n_atom__atomic[i] == 18:                                                                         
			n_atom__C6free[i] =   64.3000 ; n_atom__alpha0free[i] =  11.1000
		elif n_atom__atomic[i] == 19:                                                                         
			n_atom__C6free[i] = 3897.0000 ; n_atom__alpha0free[i] = 292.9000
		elif n_atom__atomic[i] == 20:                                                                         
			n_atom__C6free[i] = 2221.0000 ; n_atom__alpha0free[i] = 160.0000
		elif n_atom__atomic[i] == 21:                                                                         
			n_atom__C6free[i] = 1383.0000 ; n_atom__alpha0free[i] = 120.0000
		elif n_atom__atomic[i] == 22:                                                                         
			n_atom__C6free[i] = 1044.0000 ; n_atom__alpha0free[i] =  98.0000
		elif n_atom__atomic[i] == 23:                                                                         
			n_atom__C6free[i] =  832.0000 ; n_atom__alpha0free[i] =  84.0000
		elif n_atom__atomic[i] == 24:                                                                         
			n_atom__C6free[i] =  602.0000 ; n_atom__alpha0free[i] =  78.0000
		elif n_atom__atomic[i] == 25:                                                                         
			n_atom__C6free[i] =  552.0000 ; n_atom__alpha0free[i] =  63.0000
		elif n_atom__atomic[i] == 26:                                                                         
			n_atom__C6free[i] =  482.0000 ; n_atom__alpha0free[i] =  56.0000
		elif n_atom__atomic[i] == 27:                                                                         
			n_atom__C6free[i] =  408.0000 ; n_atom__alpha0free[i] =  50.0000
		elif n_atom__atomic[i] == 28:                                                                         
			n_atom__C6free[i] =  373.0000 ; n_atom__alpha0free[i] =  48.0000
		elif n_atom__atomic[i] == 29:                                                                         
			n_atom__C6free[i] =  253.0000 ; n_atom__alpha0free[i] =  42.0000
		elif n_atom__atomic[i] == 30:                                                                         
			n_atom__C6free[i] =  284.0000 ; n_atom__alpha0free[i] =  40.0000
		elif n_atom__atomic[i] == 31:                                                                         
			n_atom__C6free[i] =  498.0000 ; n_atom__alpha0free[i] =  60.0000
		elif n_atom__atomic[i] == 32:                                                                         
			n_atom__C6free[i] =  354.0000 ; n_atom__alpha0free[i] =  41.0000
		elif n_atom__atomic[i] == 33:                                                                         
			n_atom__C6free[i] =  246.0000 ; n_atom__alpha0free[i] =  29.0000
		elif n_atom__atomic[i] == 34:                                                                         
			n_atom__C6free[i] =  210.0000 ; n_atom__alpha0free[i] =  25.0000
		elif n_atom__atomic[i] == 35:                                                                         
			n_atom__C6free[i] =  162.0000 ; n_atom__alpha0free[i] =  20.0000
		elif n_atom__atomic[i] == 36:                                                                         
			n_atom__C6free[i] =  129.6000 ; n_atom__alpha0free[i] =  16.8000
		else:
			sys.exit('== Error: Current implementation of free C6 and alpha0 only for elements up to Kr. Exiting now...')
	
	return(n_atom__C6free, n_atom__alpha0free)

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
	
			R0eff = ( ( n_atom__hirshfeld_volume[i] / n_atom__free_atom_volume[i] )**(1./3.) ) * n_atom__R0free[i]
			type__collectedR0eff[ n_atom__type[i] ].append( R0eff )
	
			C6eff = ( n_atom__hirshfeld_volume[i] / n_atom__free_atom_volume[i] )**2. * n_atom__C6free[i]
			type__collectedC6eff[ n_atom__type[i] ].append( C6eff )
	
			alpha0eff = ( n_atom__hirshfeld_volume[i] / n_atom__free_atom_volume[i] ) * n_atom__alpha0free[i]
			type__collectedAlpha0eff[ n_atom__type[i] ].append( alpha0eff )
	
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
			if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):
	
				sigma = ( type__averageR0eff[ n_atom__type[i] ] + type__averageR0eff[ n_atom__type[j] ] ) * (2.**(-1./6.))
	
				C6i = type__averageC6eff[ n_atom__type[i] ]
				C6j = type__averageC6eff[ n_atom__type[j] ]
	
				alpha0i = type__averageAlpha0eff[ n_atom__type[i] ]
				alpha0j = type__averageAlpha0eff[ n_atom__type[j] ]
	
				C6 = ( 2. * C6i * C6j ) / ( (alpha0j/alpha0i)*C6i + (alpha0i/alpha0j)*C6j )
	
				pairs__epsilon[ (i,j) ] = C6 / ( 4.* sigma**6. ) * 96.485309# eV to kJ/mol
	
				if n_atom__type[i] <= n_atom__type[j]:
					typepairs__epsilon[ ( int(n_atom__type[i]),int(n_atom__type[j]) ) ] = pairs__epsilon[ (i,j) ]
				else:
					typepairs__epsilon[ ( int(n_atom__type[j]),int(n_atom__type[i]) ) ] = pairs__epsilon[ (i,j) ]
	
	return(pairs__epsilon, \
			typepairs__epsilon)

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
				energy = float(energy) *  96.485309 # eV to kJ/mol
				list_Ehl.append( energy )
	
		if dict_keywords['fine_tune_epsilon'] == 'RegressionMBD':
			energylist = []
			for line in lines:
				if '| MBD@rsSCS energy              :' in line:
					astring1, energy, astring2 = line.rsplit(None, 2)
					energylist.append( float(energy) * 96.485309 ) # eV to kJ/mol
			if not energylist:
				sys.exit('== Error: No MBD information found in '+logfile+'. Exiting now...')
			energy = energylist[-1] # last element
			list_Embd.append( energy )
	
		if dict_keywords['fine_tune_epsilon'] == 'RegressionTS':
			energylist = []
			for line in lines:
				if '| vdW energy correction         :' in line:
					astring1, energy, astring2 = line.rsplit(None, 2)
					energylist.append( float(energy) * 96.485309 ) # eV to kJ/mol
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
# bonding terms are of form: 0.5 * kB * (r-r0)**2
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
# angles bending terms are of form: 0.5 * Ktheta * (theta-theta0)**2
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
#       V1  * ( 1+cos(  phi − θ0) )
#       V2  * ( 1+cos(2*phi − θ0) )
#       V3  * ( 1+cos(3*phi − θ0) )
# θ0 is the phase offset, and k is the force constant
############################################################

def get_torsions_energies(classquadruple__V1, classquadruple__V2, classquadruple__V3, classquadruple__V4, dataString):

	list_Etorsions = []
	
	for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:
	
		quadruples__torsions = get_torsions(n_atom__xyz)
	
		Etorsions = get_torsions_energy(quadruples__torsions, \
										classquadruple__V1, classquadruple__V2, classquadruple__V3, classquadruple__V4)
	
		list_Etorsions.append( Etorsions )
	
	data[dataString] = list_Etorsions
	
	return()

############################################################
# get improper torsions energies with FF parameters
# -> improper torsion terms are of form:
#       impv2 * (1 + cos (2θ − θ0 ))
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
# get Coulomb energies with FF parameters
# -> Coulomb terms are of form: f * q1*q2 / r12
#          {   0 for 1-2-interactions and 1-3-interactions
#       f ={ 0.5 for 1-4-interactions
#          {   1 for 1-5-interactions and higher
############################################################

def get_Coulomb_energies(pairs__charge, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):

	list_Ecoul = []
	
	for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:
	
		pairs__distances = get_distances(n_atom__xyz)
	
		Ecoul = get_Coulomb_energy(pairs__distances, pairs__charge, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
	
		list_Ecoul.append( Ecoul )
	
	data['Ecoul_(FF)'] = list_Ecoul
	
	return()

############################################################
# get Coulomb energy contributions per 1-X-interaction with newFF parameters
# -> terms of form: q1*q2 / r12
############################################################

def get_Coulomb_energyContribsPer_1_X_interaction(pairs__charge, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):

	qqr2kjpermole = 138.935456
	
	list_Ecoul_14Intacts = []
	list_Ecoul_15plusIntacts = []
	
	for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:
	
		Ecoul_14Intacts = 0.
		Ecoul_15plusIntacts = 0.
	
		pairs__distances = get_distances(n_atom__xyz)
	
		# 1-2- and 1-3-interactions are already excluded in pairs__charge dictionary
		# (see function get_charge_pairs_params())
		if type__ChargeTransfer_parameters:
			# ion_index, polar_index
			polar_index = []
			n_atom__charge = {}
			for i in range(n_atoms):
				n_atom__charge[i] = type__charge[ n_atom__type[i] ]
				if class__element[n_atom__class[i]] == 'Zn' or class__element[n_atom__class[i]] == 'Na':
					ion_index = i
				elif class__element[n_atom__class[i]] == 'S' or class__element[n_atom__class[i]] == 'O' or class__element[n_atom__class[i]] == 'N':
					polar_index.append(i)
			
			cation_pairs = []		
			for i in polar_index:
				if i < ion_index:
					cation_pairs.append((i, ion_index))
				else:
					cation_pairs.append((ion_index, i))
					
			# get CN
			CN = get_CN(pairs__distances, cation_pairs, type__zero_transfer_distance)
			
			total_transq = 0	
			for pair in cation_pairs:
				if n_atom__type[pair[0]] in type__ChargeTransfer_parameters.keys():
					if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[0]]]:
						if (CN_factor == 0):
							transq = type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1]
						else:
							transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
						total_transq += transq
						n_atom__charge[pair[0]] += transq
				elif n_atom__type[pair[1]] in type__ChargeTransfer_parameters.keys():
					if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[1]]]:
						if CN_factor == 0:
							transq = type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1]
						else:
							transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
						total_transq += transq
						n_atom__charge[pair[1]] += transq
			
			n_atom__charge[ion_index] = n_atom__charge[ion_index] - total_transq
		
			for pair,qq in pairs__charge.items():
				r = pairs__distances[pair]
				if pair in list_14_interacts:
					Ecoul_14Intacts += qqr2kjpermole * n_atom__charge[pair[0]] * n_atom__charge[pair[1]] / r
				else:
					Ecoul_15plusIntacts += qqr2kjpermole * n_atom__charge[pair[0]] * n_atom__charge[pair[1]] / r
		
		else:
			for pair,qq in pairs__charge.items():
				r = pairs__distances[pair]
				if pair in list_14_interacts:
					Ecoul_14Intacts += qqr2kjpermole * qq / r
				else:
					Ecoul_15plusIntacts += qqr2kjpermole * qq / r
	
		list_Ecoul_14Intacts.append(Ecoul_14Intacts)
		list_Ecoul_15plusIntacts.append(Ecoul_15plusIntacts)
	
	data['Ecoul_(FF)_14Intacts'] = list_Ecoul_14Intacts
	data['Ecoul_(FF)_15plusIntacts'] = list_Ecoul_15plusIntacts
	
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

###############################################################################
# do regression for total energy to estimate Coulomb fudge factors
###############################################################################

def do_regression_Coulomb_fudge_factors():

	predictors = []
	
	for colname in data:
		if ( 'Ecoul_(FF)_14Intacts' in colname ):
			predictors.append( colname )
		if dict_keywords['fine_tune_only_f14'] == False:
			if ( 'Ecoul_(FF)_15plusIntacts' in colname ):
				predictors.append( colname )
	
	reg = LinearRegression(fit_intercept=True, normalize=False)
	
	# note that the vdW energy of the origFF is used here
	data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'] = data['E_high-level (Ehl)'] \
														- data['Evdw_(origFF)'] \
														- data['Etorsions_(origFF)'] \
														- data['Eimprops_(FF)'] \
														- data['Eangles_(FF)'] \
														- data['Ebonds_(FF)'] \
														- data['Epol_(FF)']
	data['Ehl-Ecoul15plusIntacts-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'] \
			= data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'] - data['Ecoul_(FF)_15plusIntacts']
	
	if dict_keywords['fine_tune_only_f14'] == True:
		reg.fit(data[predictors], data['Ehl-Ecoul15plusIntacts-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'])
	elif dict_keywords['fine_tune_only_f14'] == False:
		reg.fit(data[predictors], data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'])
	
	# get some infos
	if False:
		y_pred = reg.predict(data[predictors])
		rss = sum( (y_pred-data['Ehl-EvdW-Etorsions-Eimprops-Eangles-Ebonds-Epol'])**2. )
		print('RSS = ', rss)
		print('intercept = ', reg.intercept_)
		print('\n====\n')
	
		#for i in range(len(reg.coef_)):
		#    print(predictors[i], reg.coef_[i])
	
	f14 = reg.coef_[0]
	if dict_keywords['fine_tune_only_f14'] == False:
		f15 = reg.coef_[1]
	elif dict_keywords['fine_tune_only_f14'] == True:
		f15 = 1.0 # default value
	
	
	# write some info
	print('Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
	if dict_keywords['fine_tune_only_f14'] == False:
		print('Estimated Coulomb fudge factor for 1-5-interactions or higher: '+format(f15, '.3f'))
	print('\n====\n')
	
	if ( f14 < 0. ):
		print('Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
		sys.exit('== Error: Estimated Coulomb fudge factor for 1-4-interactions is negative. That seems strange. Exiting now...')
	if ( f15 < 0. ):
		print('Estimated Coulomb fudge factor for 1-5-interactions or higher: '+format(f15, '.3f'))
		sys.exit('== Error: Estimated Coulomb fudge factor for 1-5-interactions or higher is negative. That seems strange. Exiting now...')
	
	if ( f14 > 1.0 ):
		print('Warning! We only uses fudge factors that do not exceed values of 1.0')
		print('         Estimated Coulomb fudge factor for 1-4-interactions: '+format(f14, '.3f'))
		print('         --> Reset to 1.0')
		print('\n====\n')
		f14 = 1.0
	
	return(f14, f15)

############################################################
# get bonding energy contributions classpair-wise
# -> bonding contributions are of form: 0.5 * (r-r0)**2
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
				classpairs__bondEnergyContribs[ atuple ] += 0.5 * (r-r0)**2.
			else:
				classpairs__bondEnergyContribs[ atuple ] = 0.5 * (r-r0)**2.
	
		listofdicts_logfiles___classpairs__bondEnergyContribs.append( classpairs__bondEnergyContribs )
	
	sortedlist_classpairs = []
	
	for classpair in listofdicts_logfiles___classpairs__bondEnergyContribs[0]:
		sortedlist_classpairs.append( classpair )
	sortedlist_classpairs = sorted(sortedlist_classpairs)
	
	for classpair in sortedlist_classpairs:
	
		list_classpair_contribs = []
	
		for classpairs__bondEnergyContribs in listofdicts_logfiles___classpairs__bondEnergyContribs:
			list_classpair_contribs.append( classpairs__bondEnergyContribs[ classpair ] )
	
		data['bonds_0.5*(r-r0)**2_classpair'+str(classpair)] = list_classpair_contribs
	
	return()

############################################################
# get angles energy contributions classtriple-wise
# -> angles bending contributions are of form: 0.5 * (theta-theta0)**2
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
				classtriples__anglesEnergyContribs[ atuple ] += 0.5 * (theta-theta0)**2.
			else:
				classtriples__anglesEnergyContribs[ atuple ] = 0.5 * (theta-theta0)**2.
	
		listofdicts_logfiles___classtriples__anglesEnergyContribs.append( classtriples__anglesEnergyContribs )
	
	sortedlist_classtriples = []
	
	for classtriple in listofdicts_logfiles___classtriples__anglesEnergyContribs[0]:
		sortedlist_classtriples.append( classtriple )
	sortedlist_classtriples = sorted(sortedlist_classtriples, key=operator.itemgetter(1,2,0))
	
	for classtriple in sortedlist_classtriples:
	
		list_classtriple_contribs = []
	
		for classtriples__anglesEnergyContribs in listofdicts_logfiles___classtriples__anglesEnergyContribs:
			list_classtriple_contribs. append( classtriples__anglesEnergyContribs[ classtriple ] )
	
		data['angles_0.5_*_(theta-theta0)**2_classtriple'+str(classtriple)] = list_classtriple_contribs
	
	return()

############################################################
# get improper torsions energy contributions classquadruple-wise
# -> improper torsion contributions are of form:
#       impV2  * ( 1+cos(2*phi − θ0) )
############################################################

def get_impropsEnergyContribs():

    listofdicts_logfiles___classquadruples__impropsEnergyContribs = []

    for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:

        quadruples__improps = get_improps(n_atom__xyz)

        classquadruples__impropsEnergyContribs = {}
        
        collect_classquadruples__impropsEnergyContribs = {}

        for quadruple in list_imp1234_interacts:

            phi = quadruples__improps[quadruple]
            
            phase = 3.14159265359

            atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )

            if atuple in classquadruples__impropsEnergyContribs:
                classquadruples__impropsEnergyContribs[ atuple ] +=   1. + math.cos(  2 * phi - phase ) 
            else:
                classquadruples__impropsEnergyContribs[ atuple ] = 1. + math.cos(  2 * phi - phase ) 
                
        for atuple in classquadruples__impropsEnergyContribs:
            if atuple[2] == 3:
                if (0, 0, atuple[2], atuple[3]) in collect_classquadruples__impropsEnergyContribs:
                    collect_classquadruples__impropsEnergyContribs[ (0, 0, atuple[2], atuple[3]) ] += classquadruples__impropsEnergyContribs[ atuple ]
                else:
                    collect_classquadruples__impropsEnergyContribs[ (0, 0, atuple[2], atuple[3]) ] = classquadruples__impropsEnergyContribs[ atuple ]
            elif atuple[2] == 24 or atuple[2] == 47 or atuple[2] == 48 :
                if (0, 0, atuple[2], 0) in collect_classquadruples__impropsEnergyContribs:
                    collect_classquadruples__impropsEnergyContribs[ (0, 0, atuple[2], 0) ] += classquadruples__impropsEnergyContribs[ atuple ]
                else:
                    collect_classquadruples__impropsEnergyContribs[ (0, 0, atuple[2], 0) ] = classquadruples__impropsEnergyContribs[ atuple ]
        #print(collect_classquadruples__impropsEnergyContribs)
        #print(classquadruples__impropsEnergyContribs)

        listofdicts_logfiles___classquadruples__impropsEnergyContribs.append( collect_classquadruples__impropsEnergyContribs )

    sortedlist_imps_classquadruples = []

    for classquadruple in listofdicts_logfiles___classquadruples__impropsEnergyContribs[0]:
        sortedlist_imps_classquadruples.append( classquadruple )
    sortedlist_imps_classquadruples = sorted(sortedlist_imps_classquadruples, key=operator.itemgetter(2,3,0,1))

    for classquadruple in sortedlist_imps_classquadruples:

        list_classquadruple_contribs = []

        for collect_classquadruples__impropsEnergyContribs in listofdicts_logfiles___classquadruples__impropsEnergyContribs:
            list_classquadruple_contribs.append( collect_classquadruples__impropsEnergyContribs[ classquadruple ] )

        data['improps_1+cos(2*phi − θ0)_classquadruple'+str(classquadruple)] = list_classquadruple_contribs

    return()


############################################################
# get torsions energy contributions classquadruple-wise
# torsion (dihedral angles) contributions are of form:
#       V1  * ( 1+cos(  phi − θ0) )
#       V2  * ( 1+cos(2*phi − θ0) )
#       V3  * ( 1+cos(3*phi − θ0) )
# θ0 is the phase offset, and k is the force constant
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
					
					if ( atuple in origFF___classquadruple__V1 ):
						phase = origFF___classquadruple__V1[atuple][0]
					else:
						phase = 0.0
						
					if atuple in classquadruples__torsionsV1EnergyContribs:
						classquadruples__torsionsV1EnergyContribs[ atuple ] +=  1. + math.cos(  1 * phi - phase ) 
					else:
						classquadruples__torsionsV1EnergyContribs[ atuple ] =   1. + math.cos(  1 * phi - phase ) 
				
				if ( atuple in origFF___classquadruple__V2 ) or ( dict_keywords['RegressionTorsionalVall'] == True ):
					
					if ( atuple in origFF___classquadruple__V2 ):
						phase = origFF___classquadruple__V2[atuple][0]
					else:
						phase = 3.14159265359
					
					if atuple in classquadruples__torsionsV2EnergyContribs:
						classquadruples__torsionsV2EnergyContribs[ atuple ] += 1. + math.cos(  2 * phi - phase ) 
					else:
						classquadruples__torsionsV2EnergyContribs[ atuple ] =  1. + math.cos(  2 * phi - phase ) 
				
				if ( atuple in origFF___classquadruple__V3 ) or ( dict_keywords['RegressionTorsionalVall'] == True ):
					
					if ( atuple in origFF___classquadruple__V3 ):
						phase = origFF___classquadruple__V3[atuple][0]
					else:
						phase = 0.0
					
					if atuple in classquadruples__torsionsV3EnergyContribs:
						classquadruples__torsionsV3EnergyContribs[ atuple ] += 1. + math.cos(  3 * phi - phase )
					else:
						classquadruples__torsionsV3EnergyContribs[ atuple ] = 1. + math.cos(  3 * phi - phase )
	
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
	
		data['torsionsV1_1+math.cos(1*phi-phase)_classquadruple'+str(V1classquadruple)] = list_V1classquadruple_contribs
		
	for V2classquadruple in sortedlist_V2classquadruples:
	
		list_V2classquadruple_contribs = []
	
		for classquadruples__torsionsV2EnergyContribs in listofdicts_logfiles___classquadruples__torsionsV2EnergyContribs:
			list_V2classquadruple_contribs.append( classquadruples__torsionsV2EnergyContribs[ V2classquadruple ] )
	
		data['torsionsV2_1+math.cos(2*phi-phase)_classquadruple'+str(V2classquadruple)] = list_V2classquadruple_contribs
	
	for V3classquadruple in sortedlist_V3classquadruples:
	
		list_V3classquadruple_contribs = []
	
		for classquadruples__torsionsV3EnergyContribs in listofdicts_logfiles___classquadruples__torsionsV3EnergyContribs:
			list_V3classquadruple_contribs.append( classquadruples__torsionsV3EnergyContribs[ V3classquadruple ] )
	
		data['torsionsV3_1+math.cos(3*phi-phase)_classquadruple'+str(V3classquadruple)] = list_V3classquadruple_contribs
	
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
				atuple = ( int(n_atom__type[pair[0]]),int(n_atom__type[pair[1]]) )
			else:
				atuple = ( int(n_atom__type[pair[1]]),int(n_atom__type[pair[0]]) )

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
		#print('RSS = ', rss)
		#print('\n====\n')
		#print('intercept = ', reg.intercept_)
		
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
			if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):
				if n_atom__type[i] <= n_atom__type[j]:
					pairs__epsilon[ (i,j) ] = typepairs__epsilon[ ( int(n_atom__type[i]),int(n_atom__type[j]) ) ]
				else:
					pairs__epsilon[ (i,j) ] = typepairs__epsilon[ ( int(n_atom__type[j]),int(n_atom__type[i]) ) ]
	
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
	
	data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds-Epol'] = data['E_high-level (Ehl)'] \
																	- data['Ecoul_(FF)'] \
																	- data['Etorsions_(origFF)'] \
																	- data['Eimprops_(FF)'] \
																	- data['Eangles_(FF)'] \
																	- data['Ebonds_(FF)'] \
																	- data['Epol_(FF)']
	
	reg.fit(data[predictors], data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds-Epol'])
	
	# get some infos
	if True:
		y_pred = reg.predict(data[predictors])
		rss = sum( (y_pred-data['Ehl-Ecoul-Etorsions-Eimprops-Eangles-Ebonds-Epol'])**2. )
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
			if ( not ( (i,j) in list_12_interacts ) ) and ( not ( (i,j) in list_13_interacts )):
				if n_atom__type[i] <= n_atom__type[j]:
					pairs__epsilon[ (i,j) ] = typepairs__epsilon[ ( int(n_atom__type[i]), int(n_atom__type[j]) ) ]
				else:
					pairs__epsilon[ (i,j) ] = typepairs__epsilon[ ( int(n_atom__type[j]), int(n_atom__type[i]) ) ]
	
	return(pairs__epsilon, \
			typepairs__epsilon)


###################################################################
# do regression to estimate torsional parameters (V1, V2, V3)
###################################################################

def do_regression_torsionalV(origFF___classquadruple__V1,origFF___classquadruple__V2, origFF___classquadruple__V3 ):

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
	
	data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds-Epol'] = data['E_high-level (Ehl)'] \
													- data['Ecoul_(FF)'] \
													- data['Evdw_(FF)'] \
													- data['Eimprops_(FF)'] \
													- data['Eangles_(FF)'] \
													- data['Ebonds_(FF)'] \
													- data['Epol_(FF)']
	
	reg.fit(data[predictors], data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds-Epol'])
	
	# get some infos
	if False:
		y_pred = reg.predict(data[predictors])
		rss = sum( (y_pred-data['Ehl-Ecoul-Evdw-Eimprops-Eangles-Ebonds-Epol'])**2. )
		print('RSS = ', rss)
		print('\n====\n')
		print('intercept = ', reg.intercept_)
	
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
		if ( sortedlist_classquadruplesV1[i] in origFF___classquadruple__V1 ):
			phase = origFF___classquadruple__V1[sortedlist_classquadruplesV1[i]][0]
		else:
			phase = 0.0
		classquadruple__V1[ sortedlist_classquadruplesV1[i] ] = [phase, reg.coef_[i]]
	for i in range(len(sortedlist_classquadruplesV2)):
		if ( sortedlist_classquadruplesV2[i] in origFF___classquadruple__V2 ):
			phase = origFF___classquadruple__V2[sortedlist_classquadruplesV2[i]][0]
		else:
			phase = 3.14159265359
		classquadruple__V2[ sortedlist_classquadruplesV2[i] ] = [phase, reg.coef_[i+len(sortedlist_classquadruplesV1)]]
	for i in range(len(sortedlist_classquadruplesV3)):
		if ( sortedlist_classquadruplesV3[i] in origFF___classquadruple__V3 ):
			phase = origFF___classquadruple__V3[sortedlist_classquadruplesV3[i]][0]
		else:
			phase = 0.0
		classquadruple__V3[ sortedlist_classquadruplesV3[i] ] = [phase, reg.coef_[i+len(sortedlist_classquadruplesV1)+len(sortedlist_classquadruplesV2)]]
	
	return(classquadruple__V1, \
			classquadruple__V2, \
			classquadruple__V3)

def do_regression_imptorsionalV(origFF___classquadruple__impV2):
    
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
		#elif dict_keywords['RestrictRegressionimpVPositive'] == False:
			#reg = Lasso(alpha=dict_keywords['regularization_parameter_imptorsionalV'], fit_intercept=True, normalize=False, positive=False)        
	
	data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds-Epol'] = data['E_high-level (Ehl)'] \
														- data['Ecoul_(FF)'] \
														- data['Evdw_(FF)'] \
														- data['Etorsions_(FF)'] \
														- data['Eangles_(FF)'] \
														- data['Ebonds_(FF)'] \
														- data['Epol_(FF)']
	
	reg.fit(data[predictors], data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds-Epol'])
	
	# get some infos
	if False:
		y_pred = reg.predict(data[predictors])
		rss = sum( (y_pred-data['Ehl-Ecoul-Evdw-Etorsions-Eangles-Ebonds-Epol'])**2. )
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
			
	collect_classquadruple__impV2 = {}
	
	for i in range(len(sortedlist_classquadruplesimpV2)):
		phase = 3.14159265359
		collect_classquadruple__impV2[ sortedlist_classquadruplesimpV2[i] ] = [phase, reg.coef_[i]]
	
	classquadruple__impV2 = {}
	
	for atuple in origFF___classquadruple__impV2.keys():
		if atuple[2] == 3:
			classquadruple__impV2[ atuple ] = collect_classquadruple__impV2[ (0, 0, atuple[2], atuple[3]) ]
		elif atuple[2] == 24 or atuple[2] == 47 or atuple[2] == 48:
			classquadruple__impV2[ atuple ] = collect_classquadruple__impV2[ (0, 0, atuple[2], 0) ]
		
	#print(collect_classquadruple__impV2)
	#print(classquadruple__impV2)
		
	return(collect_classquadruple__impV2, \
	          classquadruple__impV2)

############################################################
# get charges of isolated atoms for charge transfer 
############################################################

def get_type__isolated_charge(type_charge):
	
	type__isolated_charge = {}
	
	if dict_keywords['fine_tune_isolated_charges'] == 'False':
		type__isolated_charge = copy.deepcopy(type_charge)
	elif dict_keywords['fine_tune_isolated_charges'] == 'Hirshfeld':
		type__isolated_charge = {'1085' :  0.0070, \
                                 '1086' :  0.0263, \
                                 '1142' : -0.4852, \
                                 '1148' : -0.1157, \
                                 '1166' : -0.0026, \
                                 '1177' :  0.1355, \
                                 '1178' : -0.3085, \
                                 '1180' : -0.1098, \
                                 '1183' :  0.0864, \
		                         '166'  :  0.0158, \
                                 '177'  :  0.1489, \
                                 '178'  : -0.2692, \
                                 '180'  : -0.0928, \
                                 '183'  :  0.1211, \
                                 '184'  : -0.0561, \
                                 '80'   : -0.1146, \
                                 '85'   :  0.0460, \
                                 '880'  :  1.0, \
								 '834'  :  2.0}
	elif dict_keywords['fine_tune_isolated_charges'] == 'ESP':
		type__isolated_charge = {'1085' :  0.4480, \
                                 '1086' :  0.0728, \
                                 '1142' : -0.5128, \
                                 '1148' : -1.3652, \
                                 '1166' :  0.3430, \
                                 '1177' :  0.4503, \
                                 '1178' : -0.5607, \
                                 '1180' : -0.5630, \
                                 '1183' :  0.2803, \
		                         '166'  :  0.4286, \
                                 '177'  :  0.4568, \
                                 '178'  : -0.5130, \
                                 '180'  : -0.4445, \
                                 '183'  :  0.2832, \
                                 '184'  : -0.3360, \
                                 '80'   : -0.5690, \
                                 '85'   :  0.1313, \
                                 '880'  :  1.0, \
								 '834'  :  2.0}
	elif dict_keywords['fine_tune_isolated_charges'] == 'RESP':
		type__isolated_charge = {'1085' :  0.1923, \
                                 '1086' :  0.0507, \
                                 '1142' : -0.6638, \
                                 '1148' : -0.5493, \
                                 '1166' :  0.2946, \
                                 '1177' :  0.4281, \
                                 '1178' : -0.5693, \
                                 '1180' : -0.6335, \
                                 '1183' :  0.2927, \
		                         '166'  :  0.3207, \
                                 '177'  :  0.4970, \
                                 '178'  : -0.5218, \
                                 '180'  : -0.4561, \
                                 '183'  :  0.2951, \
                                 '184'  : -0.2954, \
                                 '80'   : -0.5680, \
                                 '85'   :  0.1368, \
                                 '880'  :  1.0, \
								 '834'  :  2.0}
	
	atom_type__charge = {}
	for i in range(n_atoms):
		atom_type__charge [str(n_atom__type[i])] = type__isolated_charge[str(n_atom__type[i])]
		
	return(atom_type__charge)

############################################################
# get charge transfer parameters with Partical Swarm Optimization
############################################################

# the target function that to be minimized
def PSO_objective_ChargTrans(x, *args):
	
	type__polar = []
		
	for i in range(n_atoms):
		if n_atom__atomic[i] == 8 or n_atom__atomic[i] == 16 or n_atom__atomic[i] == 7: 
			type__polar.append(n_atom__type[i])
	type__polar = list(set(type__polar))
	
	type__zero_transfer_distance = {}
	type__ChargeTransfer_parameters = {}
	for i in type__polar:
		type__zero_transfer_distance[i] = 0
		
		slope = 0
		offset = 0
		
		type__ChargeTransfer_parameters[i] = [slope, offset]
	
	type__averageAlpha0eff = get_alpha0eff()
	
	a = -1
		
	a += 1
	CN_factor = x[a]
	
	pso_type__zero_transfer_distance = {}
	for key in type__zero_transfer_distance.keys():
		a += 1
		pso_type__zero_transfer_distance[key] = x[a]
		
	pso_type__ChargeTransfer_parameters = {}
	for key in type__ChargeTransfer_parameters.keys():
		a += 1
		
		slope = x[a]
		offset = -1 * slope * pso_type__zero_transfer_distance[key]
		pso_type__ChargeTransfer_parameters[key] = [ slope, offset]
	
	classpair__Kb, \
	classpair__r0, \
	classtriple__Ktheta, \
	classtriple__theta0, \
	classquadruple__V1, \
	classquadruple__V2, \
	classquadruple__V3, \
	classquadruple__V4, \
	classquadruple__impV2, \
	pairs__sigma, \
	pairs__epsilon, \
	pairs__charge, \
	type__charge, \
	type__averageAlpha0eff, \
	gamma = args
	
	MAE = get_MAE(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					CN_factor, \
					pso_type__zero_transfer_distance, \
					pso_type__ChargeTransfer_parameters, \
					type__averageAlpha0eff, \
					gamma)
	
	return(MAE)

def PSO_optimize(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					type__averageAlpha0eff, \
					gamma):
	
	type__polar   = []
		
	for i in range(n_atoms):
		if n_atom__atomic[i] == 8 or n_atom__atomic[i] == 16 or n_atom__atomic[i] == 7: 
			type__polar.append(n_atom__type[i])
	type__polar = list(set(type__polar))
	
	type__zero_transfer_distance = {}
	type__ChargeTransfer_parameters = {}
	for i in type__polar:
		type__zero_transfer_distance[i] = 0
		
		slope = 0
		offset = 0
		
		type__ChargeTransfer_parameters[i] = [slope, offset]
	
		
	args = (classpair__Kb, \
			 classpair__r0, \
			 classtriple__Ktheta, \
			 classtriple__theta0, \
			 classquadruple__V1, \
			 classquadruple__V2, \
			 classquadruple__V3, \
			 classquadruple__V4, \
			 classquadruple__impV2, \
			 pairs__sigma, \
			 pairs__epsilon, \
			 pairs__charge, \
			 type__charge, \
			 type__averageAlpha0eff, \
			 gamma)
	
	# Define the lower and upper bounds respectively
	lb = []
	ub = []
	
	# CN_factor
	lb.append( 1 )
	ub.append( 100 )	
		
	for key in type__zero_transfer_distance.keys():
		lb.append( 0.24 )
		ub.append( 0.4 )
	
	for key in type__ChargeTransfer_parameters.keys():
		lb.append( -15 )
		ub.append( 0 )
			
	xopt, fopt = pso(PSO_objective_ChargTrans, lb, ub, swarmsize = 40, maxiter=10000,  processes=4, args=args) 
	
	b = -1
	
	b += 1
	new___CN_factor = xopt[b]
	
	new___type__zero_transfer_distance = {}
	for key in type__zero_transfer_distance.keys():
		b += 1
		new___type__zero_transfer_distance[key] = xopt[b]
	
	new___type__ChargeTransfer_parameters = {}
	for key in type__ChargeTransfer_parameters.keys():
		b += 1
		
		slope = xopt[b]
		offset = -1 * slope * new___type__zero_transfer_distance[key]
		new___type__ChargeTransfer_parameters[key] = [ slope, offset]
	
	return(new___CN_factor, new___type__zero_transfer_distance, new___type__ChargeTransfer_parameters, fopt)

############################################################
# get atomic polarizabilities with Partical Swarm Optimization
############################################################

# Define the objective (to be minimize)
def PSO_objective_polarizabilities(x, *args):
	
	type__averageAlpha0eff = get_alpha0eff()
	
	a = -1
		
	pso_type__averageAlpha0eff = {}
	for key in type__averageAlpha0eff.keys():
		a += 1
		pso_type__averageAlpha0eff[key] = x[a]
	
	a += 1
	pso_gamma = x[a]
	
	classpair__Kb, \
	 classpair__r0, \
	 classtriple__Ktheta, \
	 classtriple__theta0, \
	 classquadruple__V1, \
	 classquadruple__V2, \
	 classquadruple__V3, \
	 classquadruple__V4, \
	 classquadruple__impV2, \
	 pairs__sigma, \
	 pairs__epsilon, \
	 pairs__charge, \
	 type__charge, \
	 CN_factor, \
	 type__zero_transfer_distance, \
	 type__ChargeTransfer_parameters = args
	
	MAE = get_MAE(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					CN_factor, \
					type__zero_transfer_distance, \
					type__ChargeTransfer_parameters, \
					pso_type__averageAlpha0eff, \
					pso_gamma)
			
	return(MAE)

def get_polarizabilities_PSO(classpair__Kb, \
					classpair__r0, \
					classtriple__Ktheta, \
					classtriple__theta0, \
					classquadruple__V1, \
					classquadruple__V2, \
					classquadruple__V3, \
					classquadruple__V4, \
					classquadruple__impV2, \
					pairs__sigma, \
					pairs__epsilon, \
					pairs__charge, \
					type__charge, \
					CN_factor, \
					type__zero_transfer_distance, \
					type__ChargeTransfer_parameters):
	
	type__averageAlpha0eff = get_alpha0eff()
		
	args = (classpair__Kb, \
			 classpair__r0, \
			 classtriple__Ktheta, \
			 classtriple__theta0, \
			 classquadruple__V1, \
			 classquadruple__V2, \
			 classquadruple__V3, \
			 classquadruple__V4, \
			 classquadruple__impV2, \
			 pairs__sigma, \
			 pairs__epsilon, \
			 pairs__charge, \
			 type__charge, \
			 CN_factor, \
			 type__zero_transfer_distance, \
			 type__ChargeTransfer_parameters)	
		
	# Define the lower and upper bounds respectively
	lb = []
	ub = []
		
	for key in type__averageAlpha0eff.keys():
		lb.append( type__averageAlpha0eff[key] * 0.2 )
		ub.append( type__averageAlpha0eff[key] * 3 )
		
	# gamma	
	lb.append( 0.8 )
	ub.append( 1.2 )
	
			
	xopt, fopt = pso(PSO_objective_polarizabilities, lb, ub, swarmsize = 40, maxiter=10000, processes=4, args=args)
	
	b = -1
	
	new___type__averageAlpha0eff = {}
	for key in type__averageAlpha0eff.keys():
		b += 1
		new___type__averageAlpha0eff[key] = xopt[b]
		
	b += 1
	new___gamma = xopt[b]
	
	return(new___type__averageAlpha0eff, new___gamma, fopt)
	
############################################################
# get coordination number(CN) of cation
############################################################

def get_CN(pairs__distances, cation_pairs, type__zero_transfer_distance):
	
	CN = 0
	for pair in cation_pairs:
		distance = pairs__distances[pair]
		if n_atom__type[pair[0]] in type__zero_transfer_distance.keys():
			if distance <= type__zero_transfer_distance[n_atom__type[pair[0]]]:
				CN += 1
		elif n_atom__type[pair[1]] in type__zero_transfer_distance.keys():
			if distance <= type__zero_transfer_distance[n_atom__type[pair[1]]]:
				CN += 1
	
	return(CN)

############################################################
# get alpha0eff for polarization energy
############################################################

def get_alpha0eff():
	
	n_atom_nohyd = []
	for i in range(n_atoms):
		if class__element[n_atom__class[i]] != 'H':
			n_atom_nohyd.append(i)
	
	sortedlist_types = []
	for n_atom in n_atom_nohyd:
		sortedlist_types.append( n_atom__type[n_atom] )
	#for n_atom,atype in n_atom__type.items():
	#	sortedlist_types.append( atype )
	sortedlist_types = sorted(set(sortedlist_types))
	
	
	type__collectedAlpha0eff = {}
	for atype in sortedlist_types:
		type__collectedAlpha0eff[atype] = []
	
	# get C6free, alpha0free atom-wise
	n_atom__C6free, n_atom__alpha0free = get_C6free_alpha0free()
	
	for logfile in list_logfiles:
	
		# get free atom volumes & Hirshfeld volumes atom-wise
		n_atom__free_atom_volume, n_atom__hirshfeld_volume = get_fhiaims_hirshfeld_volume(logfile)
	
		# calculate alpha0eff atom-wise and collect type-wise
		for i in n_atom_nohyd:
			
			alpha0eff = ( n_atom__hirshfeld_volume[i] / n_atom__free_atom_volume[i] ) * n_atom__alpha0free[i]
			type__collectedAlpha0eff[ n_atom__type[i] ].append( alpha0eff )
	
	type__averageAlpha0eff = {}
	
	# average alpha0eff
	for atype in sortedlist_types:
		type__averageAlpha0eff[atype] = numpy.mean(type__collectedAlpha0eff[ atype ]) * 0.5291772 ** 3 * 0.1 ** 3   # bohr ** 3 > angstrom ** 3 > nm ** 3
		
	return(type__averageAlpha0eff)

############################################################
# get Ei0 of logfile for polarization energy
############################################################

def get_Ei0(n_atom__xyz, pairs__distances, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):
	
	# ion_index, polar_index
	polar_index = []
	n_atom_nohyd = []
	n_atom__charge = {}
	for i in range(n_atoms):
		n_atom__charge[i] = type__charge[ n_atom__type[i] ]
		if class__element[n_atom__class[i]] == 'Zn' or class__element[n_atom__class[i]] == 'Na':
			ion_index = i
		elif class__element[n_atom__class[i]] == 'S' or class__element[n_atom__class[i]] == 'O' or class__element[n_atom__class[i]] == 'N':
			polar_index.append(i)
			
		if class__element[n_atom__class[i]] != 'H':
			n_atom_nohyd.append(i)
	
	if type__ChargeTransfer_parameters:
		
		cation_pairs = []		
		for i in polar_index:
			if i < ion_index:
				cation_pairs.append((i, ion_index))
			else:
				cation_pairs.append((ion_index, i))
				
		# get CN
		CN = get_CN(pairs__distances, cation_pairs, type__zero_transfer_distance)
		
		total_transq = 0	
		for pair in cation_pairs:
			if n_atom__type[pair[0]] in type__ChargeTransfer_parameters.keys():
				if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[0]]]:
					if (CN_factor == 0):
						transq = type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1]
					else:
						transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[0]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[0]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
					total_transq += transq
					n_atom__charge[pair[0]] += transq
			elif n_atom__type[pair[1]] in type__ChargeTransfer_parameters.keys():
				if pairs__distances[pair] <= type__zero_transfer_distance[n_atom__type[pair[1]]]:
					if (CN_factor == 0):
						transq = type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1]
					else:
						transq = ( type__ChargeTransfer_parameters[n_atom__type[pair[1]]][0] * pairs__distances[pair] + type__ChargeTransfer_parameters[n_atom__type[pair[1]]][1] ) / math.pow( CN, 1.0/float(CN_factor) ) 
					total_transq += transq
					n_atom__charge[pair[1]] += transq
		
		n_atom__charge[ion_index] = n_atom__charge[ion_index] - total_transq
	
	n_atom__E0 = {}
	n_atom__E0[ion_index] = 0
	
	for i in n_atom_nohyd:
		if i != ion_index:
			vec_r_iMe = numpy.array( n_atom__xyz[ ion_index ] ) * 0.1 - numpy.array( n_atom__xyz[ i ] ) * 0.1         # angstrom > nm
			vec_r_Mei = numpy.array( n_atom__xyz[ i ] ) * 0.1 - numpy.array( n_atom__xyz[ ion_index ] ) * 0.1
			
			
			if i < ion_index:
				pair = ( i, ion_index )
			else:
				pair = ( ion_index, i )
				
			r_iMe = pairs__distances[pair]
			
			# add r_cutoff
			r_vdw_radius = type__averageR0eff[ n_atom__type[i] ] * 0.1 + type__averageR0eff[ n_atom__type[ion_index] ] * 0.1   # angstrom > nm
			r_cutoff = gamma * r_vdw_radius
			
			if r_iMe <= r_cutoff:
				r_iMe = r_cutoff
			
			factor_i = n_atom__charge[ ion_index ] / ( r_iMe ** 3 )
			n_atom__E0[i] = factor_i * vec_r_iMe
			
			factor_Me = n_atom__charge[ i ] / ( r_iMe ** 3 )
			n_atom__E0[ion_index] += factor_Me * vec_r_Mei
	
	#print(n_atom__E0)
	return(n_atom__E0)
	
############################################################
# get Tij of logfile for polarization energy
############################################################

def get_Tij(n_atom__xyz, pairs__distances, i, j, gamma, type__averageR0eff):
	
	vec_r_ij = numpy.array( n_atom__xyz[ j ] ) * 0.1 - numpy.array( n_atom__xyz[ i ] ) * 0.1         # angstrom > nm

	#vec_r_ij = numpy.matrix( n_atom__xyz[ j ] ) * 0.1 - numpy.matrix( n_atom__xyz[ i ] ) * 0.1         # angstrom > nm
	#vec_r_ij_T = vec_r_ij.transpose()
	
	if i < j:
		pair = ( i, j )
	else:
		pair = ( j, i )
		
	r_ij = pairs__distances[pair]
	
	# add r_cutoff
	r_vdw_radius = type__averageR0eff[ n_atom__type[i] ] * 0.1 + type__averageR0eff[ n_atom__type[j] ] * 0.1   # angstrom > nm
	r_cutoff = gamma * r_vdw_radius
	
	if r_ij <= r_cutoff:
		r_ij = r_cutoff
		
	I = numpy.eye(3)
	
	rij_rij = numpy.dot(vec_r_ij.reshape(-1,1), vec_r_ij.reshape(1,-1))
	#Tij = ( 3 * vec_r_ij_T * vec_r_ij / r_ij ** 2 - I ) / r_ij ** 3
	Tij = ( 3  / r_ij ** 2 * rij_rij  - I ) / r_ij ** 3
	
	return(Tij)

############################################################
# get induced dipole of logfile for polarization energy
############################################################

def get_induced_dipole(n_atom__xyz, pairs__distances, type__averageAlpha0eff, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):
	
	for i in range(n_atoms):
		if class__element[n_atom__class[i]] == 'Zn' or class__element[n_atom__class[i]] == 'Na':
			ion_index = i
	
	n_atom_nohyd = []
	for i in range(n_atoms):
		if class__element[n_atom__class[i]] != 'H':
			n_atom_nohyd.append(i)

	# get alpha list
	n_atom__alpha0eff = {}
	for i in n_atom_nohyd:
		n_atom__alpha0eff[i] = type__averageAlpha0eff[ n_atom__type[i] ]
	
	# get Ei0
	n_atom__E0 = get_Ei0(n_atom__xyz, pairs__distances, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
	
	# initial guess of the induced dipole of cation
	dipole_Me_initial = numpy.array( [0., 0., 0.] ) 
	
	dipole_Me_1 = dipole_Me_initial

	# induced dipole
	n_atom__dipole ={}
	dipole_Me_2 = n_atom__alpha0eff[ion_index] *  n_atom__E0[ion_index]
	for i in n_atom__alpha0eff.keys():
		if i != ion_index:
			TiMe = get_Tij(n_atom__xyz, pairs__distances, i, ion_index, gamma, type__averageR0eff)
			TMei = get_Tij(n_atom__xyz, pairs__distances, ion_index, i, gamma, type__averageR0eff)
			
			n_atom__dipole[i] = n_atom__alpha0eff[i] * n_atom__E0[i] +  n_atom__alpha0eff[i] * numpy.dot(dipole_Me_1 , TiMe)
			dipole_Me_2 += n_atom__alpha0eff[ion_index] * numpy.dot(n_atom__dipole[i] , TMei)
			
	N = 0
	while numpy.linalg.norm( dipole_Me_2 - dipole_Me_1 ) > 2.0819434e-8:           # 1D = 0.020819434 e*nm   10**(-6) D > e*nm        
		#print('N', N)
		#print(numpy.linalg.norm( dipole_Me_2 - dipole_Me_1 ))
		dipole_Me_1 = dipole_Me_2
		#print(dipole_Me_1, 'print(dipole_Me_1)')
		n_atom__dipole ={}
		dipole_Me_2 = n_atom__alpha0eff[ion_index] *  n_atom__E0[ion_index]
		for i in n_atom__alpha0eff.keys():
			if i != ion_index:
				TiMe = get_Tij(n_atom__xyz, pairs__distances, i, ion_index, gamma, type__averageR0eff)
				TMei = get_Tij(n_atom__xyz, pairs__distances, ion_index, i, gamma, type__averageR0eff)
				
				n_atom__dipole[i] = n_atom__alpha0eff[i] *  n_atom__E0[i] + n_atom__alpha0eff[i] * numpy.dot(dipole_Me_1 , TiMe)
				
				dipole_Me_2 += n_atom__alpha0eff[ion_index] * numpy.dot(n_atom__dipole[i] , TMei)
		#print('dipole_Me_2', dipole_Me_2)
		
		N += 1
	
	n_atom__dipole[ion_index] = dipole_Me_2
	
	return(n_atom__dipole, n_atom__E0)
	
############################################################
# get polarization energy of logfile
############################################################
		
def get_polarization_energy(n_atom__xyz, pairs__distances, type__averageAlpha0eff, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):
	
	if dict_keywords['fine_tune_polarization_energy'] == True:
		
		n_atom__dipole, n_atom__E0 = get_induced_dipole(n_atom__xyz, pairs__distances, type__averageAlpha0eff, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
		
		Epol = 0.
		for i in n_atom__E0.keys():
			
			Epol += -0.5 * float(numpy.dot(n_atom__dipole[i], n_atom__E0[i]))
			
			
		Epol = Epol * 96.485309# eV to kJ/mol
		
	elif dict_keywords['fine_tune_polarization_energy'] == False:
		Epol = 0.
		
	return(Epol)

############################################################
# get polarization energies
############################################################

def get_polarization_energies(type__averageAlpha0eff, type__charge, gamma, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters):

	list_Epol = []
	
	type__averageR0eff = get_R0eff()
	
	for n_atom__xyz in listofdicts_logfiles___n_atom__xyz:
		
		
		pairs__distances = get_distances(n_atom__xyz)
		
		Epol = get_polarization_energy(n_atom__xyz, pairs__distances, type__averageAlpha0eff, gamma, type__averageR0eff, type__charge, CN_factor, type__zero_transfer_distance, type__ChargeTransfer_parameters)
		
		list_Epol.append( Epol )
			
	
	data['Epol_(FF)'] = list_Epol
	
	return()


############################################################
# print atom type/class overview
############################################################

def print_atom_type_class_overview():

	print('Atom type/class overview:')
	print('-------------------------')
	print('                         ')
	print('  Atom    Type   Class   Atomic    Mass  ')
	print('------------------------------------------')
	
	for i in range(n_atoms):
		printf("  %3d    %4d   %4s     %3d     %-8f\n", i,  int(n_atom__type[i]), n_atom__class[i], n_atom__atomic[i], n_atom__mass[i])
	
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
		printf(" %15s   %7.4f    %7.4f \n", classpair, origFF___classpair__r0[classpair], newFF___classpair__r0[classpair])
	
	print('\n====\n')
	
	print('kB parameters:')
	print('--------------')
	print('              ')
	
	print('>>> Original FF parameters have been used for new FF, i.e. kB parameters have not been altered.')
	
	print('                            ')
	print('  Atom class pair   kB      ')
	print('                    (origFF)')
	print('------------------------------')
	
	for classpair in sortedlist_classpairs:
		printf(" %15s   %7.1f \n", classpair, origFF___classpair__Kb[classpair])
	
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
	
	print('                                               ')
	print('  Atom class triple      theta0          theta0 ')
	print('                         (origFF)        (newFF)')
	print('----------------------------------------------------')
	
	sortedlist_classtriples = []
	
	for classtriple in origFF___classtriple__theta0:
		sortedlist_classtriples.append( classtriple )
	sortedlist_classtriples = sorted(sortedlist_classtriples, key=operator.itemgetter(1,2,0))
	
	for classtriple in sortedlist_classtriples:
		printf(" %18s  %14.11f   %14.11f \n", classtriple, origFF___classtriple__theta0[classtriple], newFF___classtriple__theta0[classtriple])
	
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
		printf(" %18s  %8.3f\n", classtriple, origFF___classtriple__Ktheta[classtriple])
	
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
		print('>>> Original FF parameters have been used for new FF, i.e. torsions parameters V1, V2, V3 and phase parameters have not been altered.')
	elif dict_keywords['fine_tune_torsionalV'] == True:
		print('>>> Original phase parameters have not been changed.')
		if dict_keywords['RegressionTorsionalVall'] == True:
			print('  >> For phase parameters that are not assigned by original FF, phase1 = 0.0, phase2 = 3.14159265359, phase3 = 0.0. ')
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
	
	print('                                                              ')
	print('  Atom class quadruple    phase1            V1         V1     ')
	print('                          (origFF)          (origFF)   (newFF)')
	print('--------------------------------------------------------------')
	
	for classquadruple in sortedlist_classquadruples:
		if classquadruple in newFF___classquadruple__V1:
			if classquadruple not in origFF___classquadruple__V1:
				origFF___classquadruple__V1[classquadruple] = [0.0, 0]
			printf("    %3s  %3s  %3s  %3s    %13.11f    %7.3f     %7.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V1[classquadruple][0], origFF___classquadruple__V1[classquadruple][1], newFF___classquadruple__V1[classquadruple][1])
	
	print('                                                              ')
	print('  Atom class quadruple    phase2            V2         V2     ')
	print('                          (origFF)          (origFF)   (newFF)')
	print('--------------------------------------------------------------')
	
	for classquadruple in sortedlist_classquadruples:
		if classquadruple in newFF___classquadruple__V2:
			if classquadruple not in origFF___classquadruple__V2:
				origFF___classquadruple__V2[classquadruple] = [3.14159265359, 0]
			printf("    %3s  %3s  %3s  %3s    %13.11f    %7.3f     %7.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V2[classquadruple][0], origFF___classquadruple__V2[classquadruple][1], newFF___classquadruple__V2[classquadruple][1])
	
	print('                                                              ')
	print('  Atom class quadruple    phase3            V3         V3     ')
	print('                          (origFF)          (origFF)   (newFF)')
	print('--------------------------------------------------------------')
	
	for classquadruple in sortedlist_classquadruples:
		if classquadruple in newFF___classquadruple__V3:
			if classquadruple not in origFF___classquadruple__V3:
				origFF___classquadruple__V3[classquadruple] = [0.0, 0]
			printf("    %3s  %3s  %3s  %3s    %13.11f    %7.3f     %7.3f\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3], origFF___classquadruple__V3[classquadruple][0], origFF___classquadruple__V3[classquadruple][1], newFF___classquadruple__V3[classquadruple][1])
	
	print('\n\nDihedral angles with V1=V2=V3=0 in original FF (unaltered):')
	print('-----------------------------------------------------------')
	
	print('                      ')
	print('  Atom class quadruple')
	print('----------------------')
	
	for classquadruple in sortedlist_classquadruples_zeroVs:
		printf("    %3s  %3s  %3s  %3s\n", classquadruple[0], classquadruple[1], classquadruple[2], classquadruple[3])
	
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
		print('>>> Original FF parameters have been used for new FF, i.e. improper torsions parameters impV2 have not been altered.')
	elif dict_keywords['fine_tune_imptorsionalV'] == True:
		print('>>> Original phase parameters phase = 3.14159265359 have not been changed.')
		if dict_keywords['Regression_imptorsionalV_Method'] == 'LinearRegression':
			print('>>> improper torsions parameters impV2 have been assigned using Linear regression from total energies.')
		elif dict_keywords['Regression_imptorsionalV_Method'] == 'Ridge':
			print('>>> improper torsions parameters impV2 have been assigned using Ridge regression from total energies.')
		elif dict_keywords['Regression_imptorsionalV_Method'] == 'Lasso':
			print('>>> improper torsions parameters impV2 have been assigned using Lasso regression from total energies.')
			
	print('                              ')
	print('  Atom class quadruple    impV2      impV2  ')
	print('                          (origFF)   (newFF)')
	print('--------------------------------------------')
	
	sortedlist_classquadruples = []
	
	for classquadruple in newFF___classquadruple__impV2:
		sortedlist_classquadruples.append( classquadruple )
	sortedlist_classquadruples = sorted(sortedlist_classquadruples, key=operator.itemgetter(2,3,0,1))

	for classquadruple in sortedlist_classquadruples:
		printf("    %3s  %3s  %3s  %3s    %6.3f     %6.3f\n", classquadruple[2], classquadruple[0], classquadruple[1], classquadruple[3], origFF___classquadruple__impV2[classquadruple][1], newFF___classquadruple__impV2[classquadruple][1])
	
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
		printf("%17s  %7.4f    %7.4f \n", typepair, origFF___typepairs__sigma[typepair], newFF___typepairs__sigma[typepair])
	
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
		printf("%17s  %7.4f    %7.4f \n", typepair, origFF___typepairs__epsilon[typepair], newFF___typepairs__epsilon[typepair])
	
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
			if n_atom__type[i] == atype:
				alist.append(i)
		printf("   %9d   %8.4f   %8.4f   ", int(atype), origFF___type__charge[atype], newFF___type__charge[atype]); print(alist)
		charge_sum1 += origFF___type__charge[atype] * len(alist)
		charge_sum2 += newFF___type__charge[atype] * len(alist)
	
	print('-------------------------------------------------------------------')
	printf("Sum of charge: %8.4f   %8.4f\n", charge_sum1, charge_sum2)
	
	print('\n====\n')
	
	return()


############################################################
# print original and new fudge factors
############################################################

def print_fudge_factors(f14, f15):

	print('Fudge factors:')
	print('--------------')
	print('              ')
	
	if dict_keywords['fine_tune_Coulomb_fudge_factors'] == False:
		print('>>> Original Coulomb fudge factors have been used for new FF, i.e. fudge factors have not been altered.')
	elif dict_keywords['fine_tune_Coulomb_fudge_factors'] == True:
		print('>>> Fudge factors have been assigned using linear regression from total energies.')
		if dict_keywords['fine_tune_only_f14'] == True:
			print('  >> Only fudge factor for 1-4-interactions have been altered.')
		elif dict_keywords['fine_tune_only_f14'] == False:
			print('  >> In addition to the fudge factor for 1-4-interactions, the fudge factor for 1-5-interactions or higher has been altered as well.')
	
	print('                  ')
	print('   Fudge factor     origFF      newFF')
	print('-------------------------------------')
	printf('            f14   0.5000       %7.6f\n', f14)
	printf('            f15   1.0000       %7.6f\n', f15)
	
	print('\n====\n')
	
############################################################
# print charge transfer parameters
############################################################

def print_charge_transfer_params(CN_factor, \
		                          type__zero_transfer_distance, \
		                          type__ChargeTransfer_parameters, \
		                          fopt):
	
	print('Charge transfer parameters:')
	print('---------------------------')
	print('                  ')
	
	if dict_keywords['fine_tune_charge_transfer'] == False:
		print('>>> Charge transfer have not been altered in new FF.')
	elif dict_keywords['fine_tune_charge_transfer'] == True:
		print('>>> Charge transfer have been altered in new FF.')
		printf('  >> Mean absolute error from PSO: %6f\n', float(fopt))
	if ( CN_factor != 'No_Charge_transfer' ) and ( CN_factor != 0 ):
		printf('>> Coordination number factor of cation: %4f\n', float(CN_factor))
	
	if (dict_keywords['fine_tune_charge_transfer'] == True) or (dict_keywords['readparamsfromffaffurr'] == True):
		print('                  ')
		print('   Atom type   Param a   Param b   distance r   Corresponding atoms')
		print('---------------------------------------------------------------------------------')
		
		sortedlist_types = []
	
		for atype in type__ChargeTransfer_parameters.keys():
			sortedlist_types.append( int(atype) )
		sortedlist_types = sorted(sortedlist_types)
		
		for key in sortedlist_types: #type__transChg_params.keys():
			printf("   %9d  %8.4f  %8.4f  %8.4f        %2s\n", int(key), type__ChargeTransfer_parameters[str(key)][0], type__ChargeTransfer_parameters[str(key)][1], type__zero_transfer_distance[str(key)], class__element[int(type__class[str(key)])])
	
	
	print('\n====\n')

############################################################
# print polarization energy parameters
############################################################	
def print_polarization_params(newFF___type__averageAlpha0eff, newFF___gamma):
	
	print('Polarization parameters:')
	print('---------------------------')
	print('                  ')
	
	if dict_keywords['fine_tune_polarization_energy'] == False:
		print('>>> Polarization_energy have not been included in new FF.')
	elif dict_keywords['fine_tune_polarization_energy'] == True:
		print('>>> Polarization_energy have been included in new FF.')
		printf('>>> Scale factor of cutoff distance: %6f\n', float(newFF___gamma))
		
	if dict_keywords['fine_tune_polarization_energy'] == True:
		print('                  ')
		print('   Atom type   Param Alpha   Corresponding atoms')
		print('------------------------------------------------')
		
		sortedlist_types = []
	
		for atype in newFF___type__averageAlpha0eff.keys():
			sortedlist_types.append( int(atype) )
		sortedlist_types = sorted(sortedlist_types)
		
		for key in sortedlist_types:
			printf("   %9d  %8.5f        %2s\n", int(key), newFF___type__averageAlpha0eff[str(key)], class__element[int(type__class[str(key)])] )
	
	
############################################################
# define the format of output xml file
############################################################

def indent(elem, level=0):
	i = "\n" + level*"  "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "  "
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
		for elem in elem:
			indent(elem, level+1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i

############################################################
############################################################

def write_OpenMM_ff_params_file(newFF___classpair__r0, \
							 origFF___classpair__Kb, \
							 newFF___classtriple__theta0, \
							 origFF___classtriple__Ktheta, \
							 newFF___classquadruple__V1, \
							 newFF___classquadruple__V2, \
							 newFF___classquadruple__V3, \
							 newFF___classquadruple__V4, \
							 newFF___collect_classquadruple__impV2, \
							 newFF___classquadruple__impV2, \
							 newFF___typepairs__sigma, \
							 newFF___typepairs__epsilon, \
							 newFF___type__charge, \
							 f14, f15):
	# creat root
	root = ET.Element("Forcefield")
	
	# son1 atomtype
	atomtype = ET.SubElement(root, "AtomTypes")
	
	# grandson(type) in atomtype
	sortedlist_types = []
	for n_atom,atype in n_atom__type.items():
		sortedlist_types.append( int(atype) )
	sortedlist_types = list(set(sortedlist_types))
	sortedlist_types.sort()
	
	for i in range(len(sortedlist_types)):
		types = str(sortedlist_types[i])
		classes = str(type__class[types])
		element =  str(class__element[int(classes)])
		mass = str(class__mass[int(classes)])
		type_name = types+str(i)
		type_name  = ET.SubElement(atomtype, "Type", attrib={'name': types, 'class': classes, 'element': element, 'mass': mass})
		
	# son2 residue
	residues = ET.SubElement(root, "Residues")	
	
	# grandson(residue) in residues
	for key in residues__atoms.keys():
		atom__list = {}
		atom__name = {}
		atom__type = {}
		atom__bondedto = {}
		atom__externalBonds = {}
		bonds = []
		externalBonds = []
		
		atoms = residues__atoms[key]
		a = -1
		for atom in atoms:
			a +=1
			atom__list[a] = atom		
			atom__name[a] = atom.name
			atom__type[a] = atom.type
			atom__bondedto[a] = atom.bondedTo
			atom__externalBonds[a] = atom.externalBonds
		for bond in atom__bondedto.keys():
			for i in atom__bondedto[bond]:
				if i >= bond:
					bonds.append((bond,i))
		for index in atom__externalBonds.keys():
			if atom__externalBonds[index] == 1:
				externalBonds.append(index)
	
		residue = ET.SubElement(residues, "Residue", attrib={'name': key})
		for key1 in atom__name.keys():
			atom = ET.SubElement(residue, "Atom", attrib={'name': atom__name[key1], 'type': atom__type[key1]} )
		for i in bonds:
			bonds = ET.SubElement(residue, "Bond", attrib={'from': str(i[0]), 'to': str(i[1])})
		for j in externalBonds:
			external = ET.SubElement(residue, "ExternalBond", attrib={'from': str(j)})
	
	# son3 HarmonicBondForce
	bondforces = ET.SubElement(root, "HarmonicBondForce") 	
	for key in origFF___classpair__Kb.keys():
		bondforce = ET.SubElement(bondforces, "Bond", attrib={'class1': str(key[0]), 'class2': str(key[1]), 'length': str(newFF___classpair__r0[key]), 'k': str(origFF___classpair__Kb[key]) })
	
	# son4 HarmonicAngleForce
	angleforces = ET.SubElement(root, "HarmonicAngleForce") 
	for key in origFF___classtriple__Ktheta.keys():
		angleforce = ET.SubElement(angleforces, "Angle", attrib={'class1': str(key[0]),'class2': str(key[1]), 'class3': str(key[2]), 'angle': str(newFF___classtriple__theta0[key]), 'k': str(origFF___classtriple__Ktheta[key])})
	
	# son5 PeriodicTorsionForce
	torsionforces = ET.SubElement(root, "PeriodicTorsionForce")
	
	# grandson(proper) in PeriodicTorsionForce
	for quadruple in list_1234_interacts:
		
		if n_atom__class[quadruple[1]] <= n_atom__class[quadruple[2]]:
			atuple = ( n_atom__class[quadruple[0]],n_atom__class[quadruple[1]],n_atom__class[quadruple[2]],n_atom__class[quadruple[3]] )
		else:
			atuple = ( n_atom__class[quadruple[3]],n_atom__class[quadruple[2]],n_atom__class[quadruple[1]],n_atom__class[quadruple[0]] )
			
		if (atuple in newFF___classquadruple__V1.keys()) or (atuple in newFF___classquadruple__V2.keys()) or (atuple in newFF___classquadruple__V3.keys()) or (atuple in newFF___classquadruple__V4.keys()):
			torsionforce = ET.SubElement(torsionforces, "Proper", attrib={'class1': str(atuple[0]),'class2': str(atuple[1]), 'class3': str(atuple[2]), 'class4': str(atuple[3])})
			a = 0
			if atuple in newFF___classquadruple__V1.keys():
				a += 1
				perio_name = 'periodicity' + str(a)
				phase_name = 'phase' + str(a)
				k_name = 'k' + str(a)
				torsionforce.set(perio_name, '1')
				torsionforce.set(phase_name , str(newFF___classquadruple__V1[atuple][0]))
				torsionforce.set(k_name, str(newFF___classquadruple__V1[atuple][1]))
			if atuple in newFF___classquadruple__V2.keys():
				a += 1
				perio_name = 'periodicity' + str(a)
				phase_name = 'phase' + str(a)
				k_name = 'k' + str(a)
				torsionforce.set(perio_name, '2')
				torsionforce.set(phase_name , str(newFF___classquadruple__V2[atuple][0]))
				torsionforce.set(k_name, str(newFF___classquadruple__V2[atuple][1]))
			if atuple in newFF___classquadruple__V3.keys():
				a += 1
				perio_name = 'periodicity' + str(a)
				phase_name = 'phase' + str(a)
				k_name = 'k' + str(a)
				torsionforce.set(perio_name, '3')
				torsionforce.set(phase_name , str(newFF___classquadruple__V3[atuple][0]))
				torsionforce.set(k_name, str(newFF___classquadruple__V3[atuple][1]))
			if atuple in newFF___classquadruple__V4.keys():
				a += 1
				perio_name = 'periodicity' + str(a)
				phase_name = 'phase' + str(a)
				k_name = 'k' + str(a)
				torsionforce.set(perio_name, '4')
				torsionforce.set(phase_name , str(newFF___classquadruple__V4[atuple][0]))
				torsionforce.set(k_name, str(newFF___classquadruple__V4[atuple][1]))
		
	# grandson(improper) in PeriodicTorsionForce
	for atuple in newFF___collect_classquadruple__impV2.keys():
		#if dict_keywords['fine_tune_imptorsionalV'] == False and dict_keywords['readparamsfromffaffurr'] == False:
		if atuple[2] == 3:
			imtorsionforce = ET.SubElement(torsionforces, "Improper", attrib={'class1': str(atuple[2]),'class2': '', 'class3': '', 'class4': str(atuple[3]), 'periodicity1': '2', 'phase1': str(newFF___collect_classquadruple__impV2[atuple][0]), 'k1': str(newFF___collect_classquadruple__impV2[atuple][1])})
		elif atuple[2] == 24 or atuple[2] == 47 or atuple[2] == 48:
			imtorsionforce = ET.SubElement(torsionforces, "Improper", attrib={'class1': str(atuple[2]),'class2':'', 'class3': '', 'class4': '', 'periodicity1': '2', 'phase1': str(newFF___collect_classquadruple__impV2[atuple][0]), 'k1': str(newFF___collect_classquadruple__impV2[atuple][1])})
		#elif dict_keywords['fine_tune_imptorsionalV'] == True or dict_keywords['readparamsfromffaffurr'] == True:
		#	imtorsionforce = ET.SubElement(torsionforces, "Improper", attrib={'class1': str(atuple[2]),'class2': str(atuple[0]), 'class3': str(atuple[1]), 'class4': str(atuple[3]), 'periodicity1': '2', 'phase1': str(newFF___classquadruple__impV2[atuple][0]), 'k1': str(newFF___classquadruple__impV2[atuple][1])})
			

			
	# son6 NonbondedForce
	if ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == True ):
		if dict_keywords['fine_tune_only_f14'] == True:
			coulomb14scale = f14
		elif dict_keywords['fine_tune_only_f14'] == False:
			coulomb14scale = f14 / f15
	elif ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == False ):
		coulomb14scale = 0.5
	
	nonbondedforces = ET.SubElement(root, "NonbondedForce", attrib={'coulomb14scale': str(coulomb14scale), 'lj14scale': '0.5'})
	
	# grandson(Atom) in NonbondedForce	
	if ( dict_keywords['fine_tune_Coulomb_fudge_factors'] == True ) and ( dict_keywords['fine_tune_only_f14'] == False ):
		for i in sortedlist_types:
			nonbondedforce = ET.SubElement(nonbondedforces, "Atom", attrib={'type': str(i), 'charge': str(newFF___type__charge[str(i)] * (math.sqrt(f15))), 'sigma': '0', 'epsilon': '0'})
	else:
		for i in sortedlist_types:
			nonbondedforce = ET.SubElement(nonbondedforces, "Atom", attrib={'type': str(i), 'charge': str(newFF___type__charge[str(i)]), 'sigma': '0', 'epsilon': '0'})	
	
	indent(root)
	et = ET.ElementTree(root)
	et.write("ffaffurr-oplsaa.xml", encoding="utf-8", xml_declaration=True,  method="xml") 	
	return()

############################################################
############################################################

def write_custombondforce_para(newFF___pairs__sigma, \
                                newFF___pairs__epsilon, \
                                CN_factor, \
                                f15, \
                                type__zero_transfer_distance, \
		                        type__ChargeTransfer_parameters, \
		                        newFF___type__averageAlpha0eff, \
		                        newFF___gamma):
	
	# creat root
	root = ET.Element("CustomForce")
	
	# son1 CustomBondForce vdw
	bondforce = ET.SubElement(root, "CustomBondForce")
	
	# grandson Bond
	for pair,sigma in newFF___pairs__sigma.items():
		if pair in list_14_interacts:
			bond = ET.SubElement(bondforce, "Bond14", attrib={'atom1': str(pair[0]), 'atom2': str(pair[1]), 'sigma': str(sigma), 'epsilon': str(newFF___pairs__epsilon[pair])})
	for pair,sigma in newFF___pairs__sigma.items():
		if pair not in list_14_interacts:
			bond = ET.SubElement(bondforce, "Bond15", attrib={'atom1': str(pair[0]), 'atom2': str(pair[1]), 'sigma': str(sigma), 'epsilon': str(newFF___pairs__epsilon[pair])})
	
	# son2 CustomChargetransfer
	if dict_keywords['fine_tune_charge_transfer'] == True:
		chargetransfer = ET.SubElement(root, "CustomChargeTransfer", attrib={'CN_factor': str(CN_factor), 'f15': str(f15)})
	elif dict_keywords['fine_tune_charge_transfer'] == False:
		if dict_keywords['readparamsfromffaffurr'] == False:
			chargetransfer = ET.SubElement(root, "CustomChargeTransfer", attrib={'CN_factor': 'No_Charge_transfer', 'f15': str(f15)})
		elif dict_keywords['readparamsfromffaffurr'] == True:
			chargetransfer = ET.SubElement(root, "CustomChargeTransfer", attrib={'CN_factor': str(CN_factor), 'f15': str(f15)}) 
	
	# grandson chargetransfer (atom_type)
	for atomType, params in type__ChargeTransfer_parameters.items():
		chargtran = ET.SubElement(chargetransfer, "Atom", attrib={'type': str(atomType), 'a' : str(params[0]), 'b' : str(params[1]), 'r' : str(type__zero_transfer_distance[atomType])})
	
	# son3 polarization energy
	if dict_keywords['fine_tune_polarization_energy'] == True:
		polarization = ET.SubElement(root, "CustomPoleForce", attrib={'gamma': str(newFF___gamma)})
		
		# grandson polarizability
		for atomType, polar in newFF___type__averageAlpha0eff.items():
			polaParams = ET.SubElement(polarization, "Polarize", attrib={'type': str(atomType), 'polarizability': str(polar)})
	
	indent(root)
	et = ET.ElementTree(root)
	et.write("CustomForce.xml", encoding="utf-8", xml_declaration=True,  method="xml") 
	
	return()
	
############################################################
############################################################	

if __name__ == "__main__":
	main()
