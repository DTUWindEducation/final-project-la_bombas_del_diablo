[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Package

Team: [la_bombas_del_diablo]

## Overview

This project implements a Blade Element Momentum (BEM) theory-based optimization model to predict the aerodynamic performance of a wind turbine — specifically, the IEA-15-240-RWT reference turbine.

It calculates blade thrust, torque, and aerodynamic power at various wind speeds using detailed airfoil data, blade geometry, and operational parameters.

The objective is to iteratively solve for optimal axial and tangential induction factors for each blade element, ensure convergence, and output an optimized power curve that can be compared against reference data.


## Quick-start Guide

### Step 1: Clone the repository

(In git bash)
git clone https://github.com/github-username/final-project-la_bombas_del_diablo.git
cd final-project-la_bombas_del_diablo

### Step 2: Navigate to the root of the cloned repository
You can do this using a terminal like Anaconda Prompt.
Example path on Windows:

cd C:\Users\User_example\folder_example\final-project-la_bombas_del_diablo

### Step 3: Install the package
Run the following command:

pip install .

This will automatically install the required dependencies listed in setup.py, including:

numpy, pandas, matplotlib, scipy

### Step 4: Run the main script

python .\examples\main.py



## Architecture
This section describes the folder structure, the main class `BemOptimization` and its key functions, as well as the overall workflow of the package — visually presented in a flow diagram and complemented by a more technical explanation.


The folder structure of the package 

![Folder structure](./folder_structure.png)


## Package Structure and Classes

### Main Script

### BemOptimization

This class handles the entire BEM optimization and results extraction process.

#### `__init__(self, index)`
Initializes the optimization for a given wind speed entry.

#### `initialize_elements_df(self, blade_data_df)`
Prepares the DataFrame where all parameters will be stored.  
Loads the relevant blade data — span position, blade twist angle, chord length — into it, and initializes the induction factors to zero.

#### `optimize_induction_factors(self, ...)`
Iteratively solves for optimal axial and tangential induction factors until convergence.

#### `calculate_thrust_and_power(self, elements_df)`
Computes total thrust, torque, and aerodynamic power from the blade element data.


# Test Classes Description




## Code Description
The code works bla bla bla.

A flow diagram of the code can be found in the file `flow_diagram_bem_optimization.png`:

![Flow Diagram](./flow_diagram_bem_optimization.png)


## Methodology and Git Workflow

Before the beginning of the project, the group agreed to follow the workflow described below:

### Pair Programming (Preferred)
- Schedule meetings for pair programming.
- Code together using screensharing.
- One person acts as the "driver" (writes the code), while others instruct and assist.

### Solo Remote Work
When working individually, follow this Git workflow and communication protocol:

1. **Pull** the latest changes from the main branch.
2. **Post in chat**:
   - What you're going to work on.
   - What the current status is since last time.
3. **Create a feature branch** (not a name-branch); use a descriptive name based on what you're working on.
4. **Code and commit regularly**:
   - a. When something works or there's a natural pause, **commit and push**.
   - b. **Commit message** should clearly state what was done.
5. **Create a pull request** for the specific change.
6. **Ask in chat**: "Please pull it."
7. **Post in chat**: Current status and progress.
8. *(Optional)* Start working on a new task, and repeat the workflow from step 1.
