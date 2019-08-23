from simtk.openmm.app import *
from simtk.openmm.app.internal import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os, sys, re

from xml.etree import ElementTree as ET


################amber vdW > oplsaa vdW#########################################################################
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
    return system

################### read paramters from CustomForce.xml ###########################################################
def get_Coulomb_factor(force_file):
	tree = ET.ElementTree(file = force_file)
	
	if tree.getroot().find('NonbondedForce') is not None:
		f14 = float(tree.getroot().find('NonbondedForce').attrib['coulomb14scale'])

	return(f14)


def get_custombondforce_para():
	
	pairs_14__para = {}
	pairs_15__para = {}
	type__ChargeTransfer_parameters = {}
	type__zero_transfer_distance = {}
	type__averageAlpha0eff = {}
	gamma=0.92
	type__averageR0eff = {}
	
	tree = ET.ElementTree(file='CustomForce.xml')
	if tree.getroot().find('CustomBondForce') is not None:
		for bond14 in tree.getroot().find('CustomBondForce').findall('Bond14'):
			if int(bond14.attrib['atom1']) <= int(bond14.attrib['atom2']):
				pair = ( int(bond14.attrib['atom1']), int(bond14.attrib['atom2']) )
			else:
				pair = ( int(bond14.attrib['atom2']), int(bond14.attrib['atom1']) )
			pairs_14__para[pair] = [float(bond14.attrib['sigma']), float(bond14.attrib['epsilon'])]
			
		for bond15 in tree.getroot().find('CustomBondForce').findall('Bond15'):
			if int(bond15.attrib['atom1']) <= int(bond15.attrib['atom2']):
				pair = ( int(bond15.attrib['atom1']), int(bond15.attrib['atom2']) )
			else:
				pair = ( int(bond15.attrib['atom2']), int(bond15.attrib['atom1']) )
			pairs_15__para[pair] = [float(bond15.attrib['sigma']), float(bond15.attrib['epsilon'])]
			
	if tree.getroot().find('CustomChargeTransfer') is not None:
		CN_factor = str(tree.getroot().find('CustomChargeTransfer').attrib['CN_factor'])
		f15 = float(tree.getroot().find('CustomChargeTransfer').attrib['f15'])
		for atom in tree.getroot().find('CustomChargeTransfer').findall('Atom'):
			type__ChargeTransfer_parameters[atom.attrib['type']] = [float(atom.attrib['a']), float(atom.attrib['b'])]
			type__zero_transfer_distance[atom.attrib['type']] = float(atom.attrib['r'])
			
	if tree.getroot().find('CustomPoleForce') is not None:
		gamma= float(tree.getroot().find('CustomPoleForce').attrib['gamma'])
		for atom in tree.getroot().find('CustomPoleForce').findall('Polarize'):
			type__averageAlpha0eff[ atom.attrib['type'] ] = float( atom.attrib['polarizability'] )
			type__averageR0eff[ atom.attrib['type'] ] = float( atom.attrib['rvdW'] )
	
	return(pairs_14__para, \
			pairs_15__para, \
			CN_factor, \
			f15, \
			type__ChargeTransfer_parameters, \
			type__zero_transfer_distance, \
			type__averageAlpha0eff, \
			gamma, \
			type__averageR0eff)

def get_CN(n_atom__xyz, cation_pairs, type__zero_transfer_distance):
	
	CN = 0
	for pair in cation_pairs:
		distance = get_distance(pair[0], pair[1], n_atom__xyz)
		if n_atom__type[pair[0]] in type__zero_transfer_distance.keys():
			if distance <= type__zero_transfer_distance[n_atom__type[pair[0]]]:
				CN += 1
		elif n_atom__type[pair[1]] in type__zero_transfer_distance.keys():
			if distance <= type__zero_transfer_distance[n_atom__type[pair[1]]]:
				CN += 1
	
	return(CN)

def get_atom__type():
	# build system with OpenMM
	pdb = PDBFile('input.pdb')
	forcefield = ForceField('OPLS-AA.xml')
	system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
	
	# get some atom information
	 ## data.atoms, data.atomType, data.atomType[atom]
	data = forcefield._SystemData()
	
	 ## Make a list of all atoms ('name', 'element', 'index', 'residue', 'id')
	data.atoms = list(pdb.topology.atoms())
	
	 ## Make a list of all bonds
	bondedToAtom = forcefield._buildBondedToAtomList(pdb.topology)
	
	 ## data.atomType, residues__atoms
	for chain in pdb.topology.chains():
		for res in chain.residues():      
			[template, matches] = forcefield._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=False)
			
			if matches is None:
				raise Exception('User-supplied template does not match the residue %d (%s)' % (res.index+1, res.name)) 
			else:
				data.recordMatchedAtomParameters(res, template, matches)			
	
	 ## n_atom__type		
	n_atom__type = {}
	for atom in data.atoms:
		n_atom__type[atom.__dict__['index']] = data.atomType[atom]
		
	return(n_atom__type)
	
global n_atom__type
n_atom__type = get_atom__type()


##################functions for each conformer###############################################################################################

# get n_atom__xyz
def get_pdb__info(file_path):
	with open(file_path, 'r') as structure:
		file_lines = structure.readlines()
		lines = list(file_lines)
		
		n_atom__xyz = {}
		for line in lines:
			coords_found = re.match(r'(\s*(\w+)\s*(\d+)\s*(\w+\d*)\s*(\w+)\s*(\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*?)', line)
			if coords_found:
				stringa, n_atom, element, residue, n_residue, x, y, z, int1, int2, symbol = line.split(None)
				n_atom__xyz[int(n_atom)-1] = [float(x), float(y), float(z)]

	return(n_atom__xyz)

# get distance(nm) between two atoms
def get_distance(atom1, atom2, n_atom__xyz):

	distance = math.sqrt(   ( n_atom__xyz[atom1][0] - n_atom__xyz[atom2][0] ) ** 2. \
						+ ( n_atom__xyz[atom1][1] - n_atom__xyz[atom2][1] ) ** 2. \
						+ ( n_atom__xyz[atom1][2] - n_atom__xyz[atom2][2] ) ** 2. )
	distance = distance * 0.1                     # angstrom to nm
	return(distance)

# get E0
def get_E0(n_atom__xyz, n_atom__charge, n_atom_nohyd, ion_index, gamma, type__averageR0eff):
	
	n_atom__E0 = {}
	n_atom__E0[ion_index] = 0
	
	for i in n_atom_nohyd:
		if i != ion_index:
			vec_r_iMe = numpy.array( n_atom__xyz[ ion_index ] ) * 0.1 - numpy.array( n_atom__xyz[ i ] ) * 0.1         # angstrom > nm
			vec_r_Mei = numpy.array( n_atom__xyz[ i ] ) * 0.1 - numpy.array( n_atom__xyz[ ion_index ] ) * 0.1
				
			r_iMe = get_distance(i, ion_index, n_atom__xyz)
			
			# add r_cutoff
			r_vdw_radius = type__averageR0eff[ n_atom__type[i] ] * 0.1 + type__averageR0eff[ n_atom__type[ion_index] ] * 0.1   # angstrom > nm
			r_cutoff = gamma * r_vdw_radius
			
			if r_iMe <= r_cutoff:
				r_iMe = r_cutoff
			
			factor_i = n_atom__charge[ ion_index ] / ( r_iMe ** 3 )
			n_atom__E0[i] = factor_i * vec_r_iMe
			
			factor_Me = n_atom__charge[ i ] / ( r_iMe ** 3 )
			n_atom__E0[ion_index] += factor_Me * vec_r_Mei
			
	return(n_atom__E0)
	
# get Tij
def get_Tij(n_atom__xyz, i, j, gamma, type__averageR0eff):
	
	vec_r_ij = numpy.array( n_atom__xyz[ j ] ) * 0.1 - numpy.array( n_atom__xyz[ i ] ) * 0.1         # angstrom > nm
		
	r_ij = get_distance(i, j, n_atom__xyz)
	
	# add r_cutoff
	r_vdw_radius = type__averageR0eff[ n_atom__type[i] ] * 0.1 + type__averageR0eff[ n_atom__type[j] ] * 0.1   # angstrom > nm
	r_cutoff = gamma * r_vdw_radius
	
	if r_ij <= r_cutoff:
		r_ij = r_cutoff
		
	I = numpy.eye(3)
	
	rij_rij = numpy.dot(vec_r_ij.reshape(-1,1), vec_r_ij.reshape(1,-1))
	Tij = ( 3  / r_ij ** 2 * rij_rij  - I ) / r_ij ** 3
	return(Tij)

# get induced dipole
def get_induced_dipole(n_atom__xyz, n_atom__alpha0eff, n_atom__charge, n_atom_nohyd, ion_index, gamma, type__averageR0eff):
	
	# get E0
	n_atom_nohyd__E0 = get_E0(n_atom__xyz, n_atom__charge, n_atom_nohyd, ion_index, gamma, type__averageR0eff)
	
	# initial guess of the induced dipole of cation
	dipole_Me_initial = numpy.array( [0., 0., 0.] ) 
	
	dipole_Me_1 = dipole_Me_initial
	
	# induced dipole
	n_atom_nohyd__dipole ={}
	dipole_Me_2 = n_atom__alpha0eff[ion_index] *  n_atom_nohyd__E0[ion_index]
	for i in n_atom_nohyd:
		if i != ion_index:
			TiMe = get_Tij(n_atom__xyz, i, ion_index, gamma, type__averageR0eff)
			TMei = get_Tij(n_atom__xyz, ion_index, i, gamma, type__averageR0eff)
			n_atom_nohyd__dipole[i] = n_atom__alpha0eff[i] *  n_atom_nohyd__E0[i] + n_atom__alpha0eff[i] * numpy.dot(dipole_Me_1 , TiMe)
			dipole_Me_2 += n_atom__alpha0eff[ion_index] * numpy.dot(n_atom_nohyd__dipole[i] , TMei)
			
	N = 0
	while numpy.linalg.norm( dipole_Me_2 - dipole_Me_1 ) > 2.0819434e-8:           # 1D = 0.020819434 e*nm   10**(-6) D > e*nm 
		
		dipole_Me_1 = dipole_Me_2
		
		n_atom_nohyd__dipole ={}
		dipole_Me_2 = n_atom__alpha0eff[ion_index] *  n_atom_nohyd__E0[ion_index]
		for i in n_atom_nohyd:
			if i != ion_index:
				TiMe = get_Tij(n_atom__xyz, i, ion_index, gamma, type__averageR0eff)
				TMei = get_Tij(n_atom__xyz, ion_index, i, gamma, type__averageR0eff)
				
				n_atom_nohyd__dipole[i] = n_atom__alpha0eff[i] *  n_atom_nohyd__E0[i] + n_atom__alpha0eff[i] * numpy.dot(dipole_Me_1 , TiMe)
				dipole_Me_2 += n_atom__alpha0eff[ion_index] * numpy.dot(n_atom_nohyd__dipole[i] , TMei)
		N += 1
	
	n_atom_nohyd__dipole[ion_index] = dipole_Me_2
	
	return(n_atom_nohyd__dipole, n_atom_nohyd__E0)
	
##########################################################################################################################
# parameter file we choose
force_file = str(sys.argv[1])
pdb_folder = str(sys.argv[2])

f = open(os.path.join(os.getcwd(),'energies_openmm_pre.kcal'), 'w+')
for file in os.listdir(pdb_folder):
	pdb = PDBFile(os.path.join(pdb_folder, file))
	forcefield = ForceField(force_file)
	system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)
	
	if force_file == 'OPLS-AA.xml':
		system = OPLS_LJ(system)
		
	# if we use newFF parameters
	elif force_file == 'ffaffurr-oplsaa.xml':
		
		# calculate vdw energies through pairs parameters if we use newFF
		pairs_vdw = openmm.CustomBondForce('4*factor*epsilon*((sigma/r)^12 - (sigma/r)^6)')
		
		pairs_vdw.addPerBondParameter("factor")
		pairs_vdw.addPerBondParameter("sigma")
		pairs_vdw.addPerBondParameter("epsilon")
		
		pairs_14__para, \
		 pairs_15__para, \
		 CN_factor, \
		 f15, \
		 type__ChargeTransfer_parameters, \
		 type__zero_transfer_distance, \
		 type__averageAlpha0eff, \
		 gamma, \
		 type__averageR0eff                 = get_custombondforce_para()
		
		for pair in pairs_14__para.keys():
			pairs_vdw.addBond(pair[0], pair[1], [0.5, pairs_14__para[pair][0], pairs_14__para[pair][1]] )
			
		for pair in pairs_15__para.keys():
			pairs_vdw.addBond(pair[0], pair[1], [1, pairs_15__para[pair][0], pairs_15__para[pair][1]] )
			
		system.addForce(pairs_vdw)
		
		if type__ChargeTransfer_parameters:
			
			# get n_atom__xyz
			n_atom__xyz = get_pdb__info(os.path.join(pdb_folder, file))
			
			f14 = get_Coulomb_factor(force_file)
			
			# ion_index, polar_index
			polar_index = []
			for (atom_index, atom) in enumerate(pdb.topology.atoms()):
				if atom.element.symbol == 'Zn' or atom.element.symbol == 'Na':
					ion_index = atom_index
				elif atom.element.symbol == 'S' or atom.element.symbol == 'O' or atom.element.symbol == 'N':
					polar_index.append(atom_index)
			
			# get CN
			cation_pairs = []		
			for i in polar_index:
				if i < ion_index:
					cation_pairs.append((i, ion_index))
				else:
					cation_pairs.append((ion_index, i))
					
			CN = get_CN(n_atom__xyz, cation_pairs, type__zero_transfer_distance)
			
			# load nonbondedforce
			forces = { force.__class__.__name__ : force for force in system.getForces() }
			nbforce = forces['NonbondedForce']
			
			# charge on L
			total_transq = 0
			
			for index in range(nbforce.getNumParticles()):
				
				# charge
				[charge, sigma, epsilon] = nbforce.getParticleParameters(index)

				if n_atom__type[index] in type__ChargeTransfer_parameters.keys():
					r = get_distance(index, ion_index, n_atom__xyz)
					
					# only consider r <= zero_transfer_distance 
					if r < type__zero_transfer_distance[n_atom__type[index]] :
						if (CN == 1) or (CN_factor == 'No_Charge_transfer'):
							transq = (type__ChargeTransfer_parameters[n_atom__type[index]][0] * r + type__ChargeTransfer_parameters[n_atom__type[index]][1]) * (math.sqrt(f15))
						else:
							transq = ( type__ChargeTransfer_parameters[n_atom__type[index]][0] * r + type__ChargeTransfer_parameters[n_atom__type[index]][1] ) * (math.sqrt(f15)) / math.pow( CN, 1.0/float(CN_factor) ) 
						total_transq += transq
						charge_new = ( charge/charge.unit + transq)* charge.unit
					else:
						charge_new = charge
						
					nbforce.setParticleParameters(index, charge_new, sigma, epsilon)
			
			[charge, sigma, epsilon] = nbforce.getParticleParameters(ion_index)
			charge_ion = ( charge/charge.unit - total_transq ) * charge.unit
			nbforce.setParticleParameters(ion_index, charge_ion, sigma, epsilon )
	
			nonbonded_force = forces['NonbondedForce']
			index__charge = {}
			for index in range(nonbonded_force.getNumParticles()):
				[charge, sigma, epsilon] = nonbonded_force.getParticleParameters(index)
				index__charge[index] = charge
				
			for i in range(nonbonded_force.getNumExceptions()):
				(p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
				if q._value != 0.0:
					q_new = index__charge[p1] * index__charge[p2] * f14
					nonbonded_force.setExceptionParameters(i, p1, p2, q_new, sig, eps)
		
		# implement polarization energy
		if type__averageAlpha0eff:
			
			# implement polarization energy, ev > kJ/mol
			pol_force = openmm.CustomExternalForce(' -0.5 * 96.485309 * ( dipole0 * E0 + dipole1 * E1 + dipole2 * E2) ')      
			
			pol_force.addPerParticleParameter("dipole0")
			pol_force.addPerParticleParameter("dipole1")
			pol_force.addPerParticleParameter("dipole2")
			pol_force.addPerParticleParameter("E0")
			pol_force.addPerParticleParameter("E1")
			pol_force.addPerParticleParameter("E2")
			
			# get n_atom_nohyd list, ion_index
			n_atom_nohyd = []
			for (atom_index, atom) in enumerate(pdb.topology.atoms()):
				if atom.element.symbol == 'Zn' or atom.element.symbol == 'Na':
					ion_index = atom_index
					n_atom_nohyd.append(ion_index)
				elif atom.element.symbol != 'H':
					n_atom_nohyd.append(atom_index)
			
			# n_atom_nohyd__alpha0eff
			n_atom_nohyd__alpha0eff = {}
			for index in n_atom_nohyd:
				n_atom_nohyd__alpha0eff[index] = type__averageAlpha0eff[ n_atom__type[index] ]
			
			# get E0
			n_atom__charge = {}
			# load nonbondedforce
			forces = { force.__class__.__name__ : force for force in system.getForces() }
			nonbonded_force = forces['NonbondedForce']
			for index in range(nonbonded_force.getNumParticles()):
				[charge, sigma, epsilon] = nonbonded_force.getParticleParameters(index)
				n_atom__charge[index] = charge/charge.unit
			
			n_atom__xyz = get_pdb__info(os.path.join(pdb_folder, file))
			
			#get induced dipole
			n_atom_nohyd__dipole, n_atom_nohyd__E0 = get_induced_dipole(n_atom__xyz, n_atom_nohyd__alpha0eff, n_atom__charge, n_atom_nohyd, ion_index, gamma, type__averageR0eff)
			
			for index in n_atom_nohyd:
				pol_force.addParticle( index, [ n_atom_nohyd__dipole[index][0], n_atom_nohyd__dipole[index][1], n_atom_nohyd__dipole[index][2], n_atom_nohyd__E0[index][0], n_atom_nohyd__E0[index][1], n_atom_nohyd__E0[index][2] ] )
				
			system.addForce(pol_force)	
		
	integrator = VerletIntegrator(0.002*picoseconds)
	simulation = Simulation(pdb.topology, system, integrator)
	
	simulation.context.setPositions(pdb.positions)
	state = simulation.context.getState(getEnergy=True) # getForces =True)
	
	savedStdout = sys.stdout
	sys.stdout = f
	print(file.split('.')[0], state.getPotentialEnergy()/4.184/kilojoules_per_mole) 
