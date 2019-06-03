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
	
	return(pairs_14__para, \
			pairs_15__para, \
			CN_factor, \
			f15, \
			type__ChargeTransfer_parameters, \
			type__zero_transfer_distance)
			
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
		 type__zero_transfer_distance        = get_custombondforce_para()
		
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
			if CN_factor != 'No_CN_factor':
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
						if (CN_factor == 'No_CN_factor') or (CN == 1) or (CN_factor == 'No_Charge_transfer'):
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
	
	integrator = VerletIntegrator(0.002*picoseconds)
	simulation = Simulation(pdb.topology, system, integrator)
	
	simulation.context.setPositions(pdb.positions)
	state = simulation.context.getState(getEnergy=True)
	
		
	savedStdout = sys.stdout
	sys.stdout = f
	print(file.split('.')[0], state.getPotentialEnergy()/4.184/kilojoules_per_mole) 

