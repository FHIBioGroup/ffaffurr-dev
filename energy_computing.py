from simtk.openmm.app import *
from simtk.openmm.app.internal import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os, sys, re

from xml.etree import ElementTree as ET


################ amber vdW combination rule > oplsaa vdW combination rule #########################################################################
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

################### read paramters from CustomForce.xml ###########################################################

def get_custombondforce_para():
	
	pairs_14__para = {}
	pairs_15__para = {}
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
	
	return(pairs_14__para, \
			pairs_15__para)
			

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
		 pairs_15__para    = get_custombondforce_para()
		
		for pair in pairs_14__para.keys():
			pairs_vdw.addBond(pair[0], pair[1], [0.5, pairs_14__para[pair][0], pairs_14__para[pair][1]] )
			
		for pair in pairs_15__para.keys():
			pairs_vdw.addBond(pair[0], pair[1], [1, pairs_15__para[pair][0], pairs_15__para[pair][1]] )
			
		system.addForce(pairs_vdw)
		
	for i in range(system.getNumForces()):
		force = system.getForce(i)
		force.setForceGroup(i)	
		
	integrator = VerletIntegrator(0.002*picoseconds)
	simulation = Simulation(pdb.topology, system, integrator)
	simulation.context.setPositions(pdb.positions)
	state = simulation.context.getState(getEnergy=True)

	
	savedStdout = sys.stdout
	sys.stdout = f
	print(file.split('.')[0], state.getPotentialEnergy()/4.184/kilojoules_per_mole) 

#os.system('sort energies_openmm_pre.kj > energies_openmm.kj')
