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

## Example Usage

FFAFFURR should be run with the following command:
~~~
python ffaffurr.py
~~~

## Outputs

- **ffaffurr-oplsaa.xml**
"ffaffurr-oplsaa.xml" includes all the parameters in standard OPLS-AA FF.
- **CustomForce.xml**
"CustomForce.xml" includes charge transfer and polarization parameters.

## Contact

Any feedback, questions, bug reports should be report through the [Issue Tracker](https://github.com/XiaojuanHu/ffaffurr-dev/issues).

## License

This package is provided under license:

## Citation
When using FFAFFURR in published work, please cite the following paper: