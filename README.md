[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Package

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

C:.
│   Aerodynamics of Wind Turbines_24_09_05_13_53_36.pdf
│   Collaboration.md
│   LICENSE
│   pyproject.toml
│   README.md
│   setup.py
│   __init__.py
│
├───examples
│   │   main.py
│   │   __init__.py
│
├───inputs
│   │   
│   │   rotor_diagram.jpeg
│   │
│   └───IEA-15-240-RWT
│       │   IEA-15-240-RWT_AeroDyn15_blade.dat
│       │   IEA_15MW_RWT_Onshore.opt
│       │
│       └───Airfoils
│
├───outputs
│   │
│   ├───pictures
│   │       3D_Airfoil_Geometry.png
│   │       Converged_and_non_converged_Power_vs_wind_speed.png
│   │       Converged_and_non_converged_Thrust_vs_wind_speed.png
│   │
│   └───results
│           converged_results_df.csv
│           results.csv
│
├───src
│   │
│   ├───bombas_package
│   │   │   BemOptimization.py
│   │   │   __init__.py
│   │   │
│   │   ├───utils
│   │   │   │   functions.py
│   │   │   │   __init__.py
│
└───tests
    │   .gitkeep
    │   conftest.py
    │   test_bem.py

Package Structure and Classes:
Main Script: \examples\main
Class: \src\bombas_package\Bem_optimization

Handles the entire BEM optimization and results extraction process.

__init__(self, index): Initializes optimization for a wind speed entry.

initialize_elements_df(self, blade_data_df): 
    Prepares the dataframe all the parameters will be stored in, and loads the relevant blade data (span position, blade twist angle, chord length) into it, and initializes the induction factors to 0.

optimize_induction_factors(self, ...): 
    Iteratively solves for optimal axial and tangential induction factors.

calculate_thrust_and_power(self, elements_df): 
    Computes total thrust, torque, and aerodynamic power.

# Test Classes Description




# Code Description
Flow diagram of the code can be seen in the file "flow_diagram_bem_optimization.png"
![Flow Diagram](./flow_diagram_bem_optimization.png)

# Methodology and Git Workflow
Before the beginning of the project, the group decided to follow a workflow as described underneath:

Optimally we will schedule meetings for pair-programming. This means we code together using screensharing with one main coder while the others watch and instructs the "driver".

When this is not possible, remote work by ourselves should follow this workflow and these rules:
 
 1. Pull
 2. Write in chat: what am i gonna work on, whats the status since last time?
 3. Create feature branch, not name-branch, specific to what youre gonna work on
 4. 
    a. When something works and there is like a natural pause, commit and push. 
    b. Commit: write clearly what you did

 5. Pull request for that specific thing you did
 6. Write in chat: please pull it
 7. Write in chat: status so far
 8. Optional: keep working on something else/new, and then the workflow reiterates.