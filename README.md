# ffaffurr-dev

## Introduction
FFAFFURR is a Python tool, which enables the parametrization of OPLS-AA and CTPOL model. It is shorten for Framework For Adjusting Force Fields Using Regularized Regression.

## Requirements
- python 3 
- OpenMM 7.4.2
- [pyswarm](https://github.com/XiaojuanHu/pyswarm)

## Inputs
- **OPLS-AA.xml**
Because the goal is to "adjust" existing parameters of the OPLS-AA FF, it is reasonable to provide a standardized listing of parameters as input.  Thus, parameter file "OPLS-AA.xml" is needed. For any system already set-up with OpenMM, e.g. when using the standard OPLS-AA FF parameters that are distributed with the OpenMM package, the standardized parameters and connectivity list are taken from the inner code of OpenMM by ffaffurr.py.
- **input.pdb**
The "input.pdb" file is the basic coordinate file that OpenMM-readable.
- **ffaffurr.input**
The input file "ffaffurr.input" contains the "switches" that control the behavior of the framework, e.g. what kind of parameters are to be "adjusted" or what regression model to use.
- **ffaffurr.input.FHI-aims-logfiles**
the input file "ffaffurr.input.FHI-aims-logfiles" contains a list of FHI-aims-specific (https://fhi-aims.org/) output files produced when calculating single-point DFT energies. Obviously, these files must be produced for a set of conformers that serves as training data.
	- **resp.chrg**
	If RESP charges are specified in "ffaffurr.input", a seperate file 'resp.chrg' containing atomic RESP charges for each conformer should be included in the folders listed in ffaffurr.input.FHI-aims-logfiles.

## List of input keywords
All the available keywords and their descriptions in **ffaffurr.input** are listed below. In **ffaffurr.input**, lines starting with “#” is a comment.

### Choose original parameters
-  **Tag: readparamsfromffaffurr** 
	- Usage: readparamsfromffaffurr *True/False*
	- Purpose: 
> - choose original parameters from
> -     OPLS-AA.xml
> -    (use: readparamsfromffaffurr False)
> -   OR ffaffurr-oplsaa.xml.This is used to optimize parameters step by  step.
> -   (use: readparamsfromffaffurr True)

### Choose the fitting type (Weighted or unweighted)

- **Tag: boltzmann_weighted_fitting**
	- Usage: boltzmann_weighted_fitting  *True/False*
	- Purpose: 
> -  Boltzmann factors are of form: A * exp( - Ei_QM / RT)
> -  A = normalization constant
> -  Ei_QM = QM relative energy
> -  RT = temperature factor
> -  Boltzmann weighted fitting can be used
> -      (use: readparamsfromffaffurr False)
> -      OR unweighted fitting
> -     (use: readparamsfromffaffurr False)

- **Tag: temperature_factor**
	- Usage: temperature_factor *value*
	- Purpose:
> -  only if boltzmann_weighted_fitting = True:
> -  set temperature factor (e.g. 32)

### Keywords for bond parametrization

- **Tag: fine_tune_r0**
	- Usage: fine_tune_r0 *True/False*
	- Purpose:
> -  bonding terms are of form: 0.5 * kB * (r-r0)^2^
> -  kB = spring constant parameter
> -      -> original FF parameters are used
> -         (parameters not altered)
> - r0 = equlibrium bond length parameter between pair of atoms
> -      -> original FF parameters can be used (parameters not altered)
> -         (use: fine_tune_r0 False)
> -         OR an average distance over all FHI-aims input logfiles is computed to estimate (fine-tune) r0
> -            (use: fine_tune_r0 True)
> -            -> this makes ONLY sense if geometrically optimized structures have been used in high-level input logfiles

### Keywords for angle parametrization

- **Tag: fine_tune_theta0**
	- Usage: fine_tune_theta0 *True/False*
	- Purpose:
> - angles bending terms are of form: 0.5 * Ktheta * (theta-theta0)^2^
> - ktheta = spring constant parameter
> -          -> original FF parameters are used
> -             (parameters not altered)
> - theta0 = equlibrium angle parameter between triplet of atoms
> -          -> original FF parameters can be used (parameters not altered)
> -             (use: fine_tune_theta0 False)
> -             OR an average angle over all FHI-aims input logfiles is computed to estimate (fine-tune) theta0
> -                (use: fine_tune_theta0 True)
> -                -> this makes ONLY sense if geometrically optimized
> -                   structures have been used in FHI-aims input logfiles

### Keywords for torsion parametrization

- **Tag: fine_tune_torsionalV**
	- Usage: fine_tune_torsionalV *False/Regression*
	- Purpose:
> - torsion (dihedral angles) terms are of form: V * ( 1+cos( nθ − θ0) )
> -       V = torsional parameter
> -      -> original FF parameters can be used (parameters not altered)
> -         (use: fine_tune_torsionalV False)
> -         OR parameters can be assigned using regression from energies of FHI-aims input logfiles
> -            (use: fine_tune_torsionalV Regression)
> -       (n = periodicity, θ0 = phase offset, θ = dihedral angle)

- **Tag: RegressionTorsionalVall**
	- Usage: RegressionTorsionalVall *True/False*
	- Purpose:
> - only if fine_tune_torsionalV = Regression:
> - either ALL torsional parameters (V1, V2, V3) are considered in regrssion
> - (use: RegressionTorsionalVall True)
> -     OR only torsional parameters are considered that are non-zero in original FF
> -        (use: RegressionTorsionalVall False)
> - Note: cases for which V1 = V2 = V3 = 0 are never taken into account (those cases are usually quadruples that contain a hydrogen atom)

- **Tag: Regression_torsionalV_Method**
	- Usage: Regression_torsionalV_Method *LinearRegression/Ridge/Lasso*
	- Purpose:
> - only if fine_tune_torsionalV = Regression:
> -   use linear regression
> -   OR Ridge regression
> -   OR Lasso regression

- **Tag: regularization_parameter_torsionalV**
	- Usage: regularization_parameter_torsionalV *value*
	- Purpose:
> - only if fine_tune_torsionalV = Regression AND Regression_torsionalV_Method = {Ridge/Lasso}:
> - set regularization parameter (e.g. 0.1)

### Keywords for improper torsion parametrization

- **Tag: fine_tune_imptorsionalV**
	- Usage: fine_tune_imptorsionalV *False/Regression*
	- Purpose:
> - improper torsion terms are of form: V2imp  * ( 1+cos(2*θ − θ0) )
> - V2imp = improper torsional parameter
> -      -> original FF parameters are used (parameters not altered)
> -         (use: fine_tune_imptorsionalV False)
> -         OR parameters can be assigned using regression from energies of FHI-aims input logfiles
> -         (use: fine_tune_imptorsionalV Regression)

- **Tag: Regression_imptorsionalV_Method**
	- Usage: Regression_imptorsionalV_Method *LinearRegression/Ridge/Lasso*
	- Purpose:
> - only if fine_tune_imptorsionalV = Regression:
> -   use linear regression
> -   OR Ridge regression
> -   OR Lasso regression

- **Tag: regularization_parameter_imptorsionalV**
	- Usage: regularization_parameter_imptorsionalV *value*
	- Purpose:
> - only if fine_tune_imptorsionalV = Regression AND Regression_imptorsionalV_Method = {Ridge/Lasso}:
> - set regularization parameter (e.g. 0.1)

### Keywords for LJ parametrization

- **Tag: fine_tune_sigma**
	- Usage: fine_tune_sigma *False/TS*
	- Purpose:
> - van der Waals (vdW; Lennard-Jones) terms are of form: 4*epsilon * f * [ (sigma/r)^12^ - (sigma/r)^6^ ]
> -  f =  0 for 1-2-interactions and 1-3-interactions
> - f = 1/2 for 1-4-interactions
> -  f =    1 for 1-5-interactions and higher
> - sigma = distance parameter at which the inter-particle potential is zero
> -         -> original FF parameters can be used (parameters not altered)
> -            (use: fine_tune_sigma False)
> -            OR sigma is calculated from FHI-aims input logfiles using the Tkatchenko-Scheffler (TS) method
> -               (use: fine_tune_sigma TS)

- **Tag: fine_tune_epsilon**
	- Usage: fine_tune_epsilon *False/TS/RegressionTS/RegressionMBD/RegressionTot*
	- Purpose:
> - epsilon = potential well depth parameter
> -         -> original FF parameters can be used (parameters not altered)
> -            (use: fine_tune_epsilon False)
> -            OR epsilon is calculated from FHI-aims input logfiles using the Tkatchenko-Scheffler (TS) method
> -               (use: fine_tune_epsilon TS)
> -            OR epsilon parameters can be assigned using regression from TS energies of FHI-aims input logfiles
> -               (use: fine_tune_epsilon RegressionTS)
> -            OR epsilon parameters can be assigned using regression from MBD@rsSCS energies of FHI-aims input logfiles
> -               (use: fine_tune_epsilon RegressionMBD)
> -            (Note: I don't know if the latter two options are even useful since both those DFT energies are calculated using damping functions that in turn are parameterized against individual exchange-correlation functionals)
> -           OR epsilon parameters can be assigned using regression from total energies of FHI-aims input logfiles
> -               (use: fine_tune_epsilon RegressionTot)

- **Tag: RegressionEpsilonMethod**
	- Usage: RegressionEpsilonMethod *LinearRegression/Ridge/Lasso*
	- Purpose:
> - only if fine_tune_epsilon = Regression{TS/MBD/Tot}:
> -   use linear regression
> -   OR Ridge regression
> -   OR Lasso regression

- **Tag:regularization_parameter_epsilon**
	- Usage: regularization_parameter_epsilon *value*
	- Purpose:
> - only if RegressionEpsilonMethod = {Ridge/Lasso}:
> - set regularization parameter (e.g. 0.1)

- **Tag: RestrictRegressionEpsilonPositive**
	- Usage: RestrictRegressionEpsilonPositive *True/False*
	- Purpose:
> - only if fine_tune_epsilon = Regression{TS/MBD/Tot} AND RegressionEpsilonMethod = Lasso:
> -   restrict regression-fitted epsilon parameters to be positive
> -   (use: RestrictRegressionEpsilonPositive True)
> -   OR not (use: RestrictRegressionEpsilonPositive False)

- **Tag: SetExplicitlyZero_vdW_SigmaEpsilon**
	- Usage: SetExplicitlyZero_vdW_SigmaEpsilon *True/False*
	- Purpose:
> - only if fine_tune_{sigma/epsilon} = TS:
> -   some epsilon & sigma parameters might be zero in original FF
> -     -> Do you want to also set them explicitly to zero even if you (re-)calculate/assign them using TS?
> -        (use SetExplicitlyZero_vdW_SigmaEpsilon True)
> -        OR not (use: SetExplicitlyZero_vdW_SigmaEpsilon False)

### Keywords for electrostatic parametrization

- **Tag: fine_tune_charge**
	- Usage: fine_tune_charge *False/Hirshfeld/ESP/RESP*
	- Purpose:
> - Coulomb terms are of form: f * q1*q2 / r12
> -  f =   0 for 1-2-interactions and 1-3-interactions
> - f = 1/2 for 1-4-interactions
> - f =   1 for 1-5-interactions and higher
> - q1,q2 = partial charge parameters of atoms
> -         -> original FF parameters can be used (parameters not altered)
> -            (use: fine_tune_charge False)
> -         OR partial charge parameters can be assigned using Hirshfeld charges from FHI-aims input logfiles
> -            (use: fine_tune_charge Hirshfeld)
> -         OR partial charge parameters can be assigned using ESP charges from FHI-aims input logfiles
> -            (use: fine_tune_charge ESP)
> -         OR partial charge parameters can be assigned using RESP charges from external RESP charge files
> -            (use: fine_tune_charge RESP)

- **Tag: fine_tune_Coulomb_fudge_factors**
	- Usage: fine_tune_Coulomb_fudge_factors *True/False*
	- Purpose:
> - Coulomb fudge factors are of form (in original OPLS-AA FF):
> -  f =      0 for 1-2-interactions and 1-3-interactions
> - f = 1/2 for 1-4-interactions
> - f =   1 for 1-5-interactions and higher
> -   -> original fudge factors can be used (fudge factors not altered)
> -      (use: fine_tune_Coulomb_fudge_factors False)
> -   OR fudge factors can be assigned using linear regression from total energies of FHI-aims input logfiles
> -      (use: fine_tune_Coulomb_fudge_factors True)

- **Tag: fine_tune_only_f14**
	- Usage: fine_tune_only_f14 *True/False*
	- Purpose:
> - only if fine_tune_Coulomb_fudge_factors = True:
> - general OPLS-AA form: Ecoul =  f_14 * Ecoul(1-4) + f_15 * Ecoul(1-5)
> -       --> f_14 = fudge factor for 1-4-interactions (default: 0.5)
> -       --> f_15 = fudge factor for 1-5-interactions (default: 1.0)
> - -> estimate f_14 and f_15
> -    (use: fine_tune_only_f14 False)
> - OR estimate ONLY f_14
> -    (use: fine_tune_only_f14 True)

### Keywords for charge transfer parametrization

- **Tag: fine_tune_charge_transfer**
	- Usage: fine_tune_charge_transfer *False/Tot/ChargeDistr/ESP*
	- Purpose:
> - Charge transfers are of form CN ** (1/alpha) * (a * r + b):
> - CN = coordination number
> - a = slope of charge transfer
> - b = offset of charge transfer
> -     -> don't consider charge transfer
> -         (use: fine_tune_charge_transfer False)
> -      OR fitting charge transfer params to total energies of FHI-aims input logfiles
> -         (use: fine_tune_charge_transfer Tot) 
> -      OR fitting charge transfer params to charge distribution of FHI-aims input logfiles
> -         (use: fine_tune_charge_transfer ChargDistr)
> - alpha, a, b are obtained by partical swarm optimizaiton(PSO)
> -      OR fitting charge transfer params to electrostatic potentials of FHI-aims outputs
> -         (use: fine_tune_charge_transfer ESP)

- **Tag: ChargeTransferESPMethod**
	- Usage: ChargeTransferESPMethod *PSO/Ridge/Lasso*
	- Purpose:
> - only if fine_tune_charge_transfer = ESP:
> -   use PSO
> -   OR Ridge regression
> -   OR Lasso regression

- **Tag: regularization_parameter_CharTran**
	- Usage: regularization_parameter_CharTran *value*
	- Purpose:
> - only if ChargeTransferESPMethod = {Ridge/Lasso}:
> - set regularization parameter (e.g. 0.1)

- **Tag: number_of_pso_CT**
	- Usage: number_of_pso_CT *value*
	- Purpose:
> - only if fine_tune_charge_transfer = Tot/ChargDistr OR ChargeTransferESPMethod = PSO:
> - set the number of independent PSO running (e.g. 1).

- **Tag: number_of_processes_CT**
	- Usage: number_of_processes_CT *value*
	- Purpose:
> - only if fine_tune_charge_transfer = Tot/ChargDistr OR ChargeTransferESPMethod = PSO:
> - set the number of processes( e.g. 2. if processes > 1, then pso is paralleled)

- **Tag: fine_tune_isolated_charges**
	- Usage: fine_tune_isolated_charges *False/Hirshfeld/ESP/RESP*
	- Purpose:
> - only if fine_tune_charge_transfer = Tot/ChargDistr OR ChargeTransferESPMethod = PSO:
> - charges of atoms before charge transfer:
> -   -> original FF parameters can be used
> -      (use: fine_tune_isolated_charges False)
> -   OR Hirshfeld charges
> -      (use: fine_tune_isolated_charges Hirshfeld)
> -   OR ESP charges
> -      (use: fine_tune_isolated_charges ESP) 
> -   OR RESP charges
> -      (use: fine_tune_isolated_charges RESP)

- **Tag: fine_tune_charges_distribution**
	- Usage: fine_tune_charges_distribution *Hirshfeld/ESP/RESP*
	- Purpose:
> - only if fine_tune_charge_transfer = ChargDistr:
> - charges distribution of fhiaims:
> -   -> Hirshfeld charges
> -      (use: fine_tune_isolated_charges Hirshfeld)
> -   OR ESP charges
> -      (use: fine_tune_isolated_charges ESP) 
> -   OR RESP charges
> -      (use: fine_tune_isolated_charges RESP)

- **Tag: fine_tune_starting_points**
	- Usage: fine_tune_starting_points *False/file_name*
	- Purpose:
> - only if fine_tune_charge_transfer = Tot/ChargDistr OR ChargeTransferESPMethod = PSO:
> - choose starting points of PSO:
> -   -> generate random points as starting points
> -      (use: fine_tune_starting_points False)
> -   OR define some of the starting points
> -      (use: fine_tune_starting_points FILE_OF_THE_STARTING_POINTS. e.g. CT_initial)

### Keywords for polarization parametrization

- **Tag: fine_tune_polarization_energy**
	- Usage: fine_tune_polarization_energy *True/False*
	- Purpose:
> - Polarization terms are of form -0.5 * Alpha * E * E0 :
> - Alpha = polarizability of atoms
> -     E = total electrostatic field produced by atomic charges and induced dipole
> -    E0 = electrostatic field produced by charges in the system
> -    -> polarization energy can be included in force field
> -       (use: fine_tune_polarization_energy True)
> -    OR not (use: fine_tune_polarization_energy False)

- **Tag: fine_tune_polarizabilities**
	- Usage: fine_tune_polarizabilities *True/False*
	- Purpose:
> - only if fine_tune_polarization_energy = True:
> - atomic polarizabilities:
> -   -> polarizabilities can be calculated from FHI-aims
> -      (use: fine_tune_polarizabilities False)
> -   OR can be assigned using particle swarm optimization
> -      (use: fine_tune_polarizabilities True)

- **Tag: number_of_pso_POL**
	- Usage: number_of_pso_POL *value*
	- Purpose:
> - only if fine_tune_polarizabilities = True:
> - set the number of independent PSO running (e.g. 1)

- **Tag: number_of_processes_POL**
	- Usage: number_of_processes_POL *value*
	- Purpose:
> - only if fine_tune_polarizabilities = True:
> - set the number of processes( e.g. 2. if processes > 1, then pso is paralleled)

## Outputs

- **ffaffurr-oplsaa.xml**
"ffaffurr-oplsaa.xml" includes all the parameters in standard OPLS-AA FF.
- **CustomForce.xml**
"CustomForce.xml" includes charge transfer and polarization parameters.

## Example Usage

An example of the usage of FFAFFURR is in the **workingExample-AcCysNMe** folder. The example should be run with the following command:
~~~
python ../ffaffurr.py
~~~

## Contact

Any feedback, questions, bug reports should be report through the [Issue Tracker](https://github.com/XiaojuanHu/ffaffurr-dev/issues).

## License

This package is provided under license:

## Citation
When using FFAFFURR in published work, please cite the following paper: