[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: [la_bombas_del_diablo]

## Overview

[This project implements a Blade Element Momentum (BEM) theory-based optimization model to predict the aerodynamic performance of a wind turbine (specifically, the IEA-15-240-RWT reference turbine).
It calculates the blade thrust, torque, and aerodynamic power at different wind speeds using detailed airfoil data, blade geometry, and operational parameters.

The objective is to iteratively solve for optimal axial and tangential induction factors for each blade element, ensuring convergence, and outputting an optimized power curve that can be compared against reference data.]

## Quick-start guide

### [step 1: Clone the repository

* git clone https://github.com/github-username/final-project-la_bombas_del_diablo.git
* cd <final-project-la_bombas_del_diablo>

### Step 2: Set up environment Named `new_env_for_BEM`

### Mac/Linux:
* ```bash
* python3 -m venv new_env_for_BEM
* source new_env_for_BEM/bin/activate
 
### Windows 
* python -m venv new_env_for_BEM
* new_env_for_BEM\Scripts\activate

### Step 3: Install dependencies
If you have earlier verions of python
* pip install "package" 

If you have python3 use: 
* pip3 install "package"

(replace "package" with each of the following)
* numpy
* pandas
* matplotlib
* scipy

e.g. "pip install numpy pandas matplotlib scipy"



### Step 4: Check Folder Structure

Make sure the following files are in place (especially under inputs/IEA-15-240-RWT/):



```text
project-root/
├── inputs/
│   └── IEA-15-240-RWT/
│       ├── Airfoils/
│       ├── IEA_15MW_RWT_Onshore.opt
│       └── IEA-15-240-RWT_AeroDyn15_blade.dat
```



Note: Ensure that the inputs/IEA-15-240-RWT/ directory contains:
* IEA_15MW_RWT_Onshore.opt
* Airfoils 
* IEA-15-240-RWT_AeroDyn15_blade.dat


### Step 5: Run main.py

Enter the following in the terminal
depending on your version of python. if you have python 3 use the following 
* python3 examples/main.py 

if you dont have the updated version of python use 
* python examples/main.py
]

## Architecture

### File structure

```text
project/
├── examples/                          # Example scripts and usage
│   ├── __init__.py
│   └── main.py                        # Main script to run BEM optimization
│
├── inputs/                            # Input data for turbine model
│   └── IEA-15-240-RWT/
│       ├── Airfoils/                  # Airfoil geometry and polar data
│       ├── IEA_15MW_RWT_Onshore.opt   # OpenFAST .opt file
│       └── IEA-15-240-RWT_AeroDyn15_blade.dat  # Blade definition
│   └── rotor_diagram.jpeg             # Visual reference image
│
├── outputs/                           # All generated outputs
│   ├── pictures/                      # Visualizations vand result plots
│   └── results/                       # Numerical results (e.g. CSVs)
│
├── src/                               # Source code of the project
│   └── bombas_package/                # Main Python package for BEM
│       ├── utils/                     # Utility/helper scripts (if applicable)
│       ├── __init__.py
│       └── BemOptimization.py         # Core class for BEM computation
│
├── tests/                             # Unit and integration tests
│   ├── conftest.py                    # Pytest configuration
│   ├── test_functions.py              # Tests for functions in functions.py
│   └── __pycache__/                   # Compiled bytecode (ignored by Git)
│
├── .coverage                          # Coverage report from pytest-cov
├── .gitignore                         # Git ignore rules
├── Aerodynamics of Wind Turbines.pdf # Reference textbook/resource
├── LICENSE                            # License for open-source usage
├── pyproject.toml                     # Project metadata and dependencies
├── README.md                          # Project description and usage instructions
├── setup.py                           # Script for installing the package
└── TO_DO.md                           # Ongoing task and feature list
```

### Code Architecture 

![Code Diagram](./outputs/pictures/Codediagram.png)


### Class description
The BemOptimization class implements the core logic for Blade Element Momentum (BEM) analysis of a wind turbine rotor. It simulates aerodynamic behavior at a specific operating point (a wind speed from a power curve) and computes performance metrics like aerodynamic power, torque, and thrust based on blade geometry and airfoil characteristics.

initialize_elements_df(blade_data_df)
* Initializes and returns a DataFrame for blade elements, setting initial induction factors and computing initial flow angles.


optimize_induction_factors(...)
* Performs iterative BEM calculations:
* Updates flow angles, angle of attack, and aerodynamic coefficients (Cl, Cd, Cn, Ct).
* Applies tip loss corrections and computes new induction factors.
* Stops when convergence is reached or iteration limit is hit.
* Updates self.elements_df.

calculate_thrust_and_power(...)
* Given converged induction values:
* Calculates differential thrust and torque along the blade.
* Integrates to find total rotor thrust, torque, and aerodynamic power.
* Computes nondimensional coefficients CT and CP
* Results are stored in the instance for later use or plotting.


## Peer review

[]
