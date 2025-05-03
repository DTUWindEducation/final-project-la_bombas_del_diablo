[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: [la_bombas_del_diablo]

## Overview

[This project implements a Blade Element Momentum (BEM) theory-based optimization model to predict the aerodynamic performance of a wind turbine (specifically, the IEA-15-240-RWT reference turbine).
It calculates the blade thrust, torque, and aerodynamic power at different wind speeds using detailed airfoil data, blade geometry, and operational parameters.

The objective is to iteratively solve for optimal axial and tangential induction factors for each blade element, ensuring convergence, and outputting an optimized power curve that can be compared against reference data.]

## Quick-start guide

[step 1: Clone the repository

git clone https://github.com/github-username/final-project-la_bombas_del_diablo.git

cd <your-repo-folder>

step 2: Navigate to the root of the cloned repository (For example use Anaconda prompt)

Example of a path: (cd is the command to change directory)

cd C:\Users\User_example\folder_example\final-project-la_bombas_del_diablo

Step 3: Install the package:

pip install .

the package contains the following dependicies which will be installed automatically (See setup.py file for details)

numpy
pandas
matplotlib
scipy


Step 4: Run the main script

python .\examples\main.py

## Architecture

[project/
├── inputs/                 # Contains airfoil files, blade data, power curve
│   └── IEA-15-240-RWT/
│       ├── Airfoils/
│       ├── IEA_15MW_RWT_Onshore.opt
│       └── IEA-15-240-RWT_AeroDyn15_blade.dat
├── src/
│   ├── main.py              # Main BEM optimization 
|   ├── __init__.py          # Functions
script
│   ├── functions.py         # Helper functions for BEM calculations
│
├── outputs/                 # Contains generated plots and results
│   └── pictures/
├── tests/                   # Unit tests for core functions
├── README.md                # This file
└── Collaboration.md         # Dependencies list
]


Package Structure and Classes:

src/main.py
Class: Bem_optimization

Handles the entire BEM optimization and results extraction process.

__init__(self, index): Initializes optimization for a wind speed entry.

initialize_elements_df(self): Prepares blade elements and inductions.

optimize_induction_factors(self, ...): Iteratively solves for optimal axial and tangential induction factors.

calculate_thrust_and_power(self, elements_df): Computes total thrust, torque, and aerodynamic power for a given condition.


src/functions.py
Utility and helper functions, including:

Data reading functions:

read_airfoil_file(), read_airfoil_polar_file(), read_blade_data_file(), read_power_curve_file()

Core BEM computations:

compute_flow_angle(), compute_local_angle_of_attack(), interpolate_Cl_Cd_coeff(), etc.

Aerodynamic calculations:

compute_thrust_coeff(), compute_power_coeff(), compute_dT(), compute_dM()

Convergence check and update rules:

check_convergence(), update_axial_joe(), update_tangential_joe()

Plotting utilities:

plot_airfoils(), plot_airfoils_3d(), plot_results_vs_ws(), etc.


Logical Flow Diagram:

1. Load inputs: airfoil data, blade geometry, power curve
2. For each wind speed:
    - Initialize a BEM_optimization instance
    - Optimize induction factors iteratively
    - Compute thrust, torque, aerodynamic power
    - Store results
3. Plot results:
    - Compare optimized BEM outputs with reference power and thrust curves


Inputs (Airfoils, Blade, Power Curve)
        ↓
 BEM_optimization (per wind speed)
        ↓
 Optimize induction factors
        ↓
Compute Loads → Results → Plots


## Peer review

[Plot verification (comparison of BEM vs reference power curve and thrust curve).]
