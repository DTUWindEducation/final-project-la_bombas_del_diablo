# %% imports
print('initializing main.py')
"""Script for the final project"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, sqrt, tan, arctan, arcsin, arccos
import pandas as pd
import scipy as sp
import re
import functions as fn
print('imported libraries')

# %% Load airfoil data
# Define data directory - will work on any computer
DATA_DIR = Path(__file__).resolve().parent.parent
#print(f'Data directory: {DATA_DIR}')

# Read the coordinates and polar data for the airfoils
airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils')
print('airfoil data loaded')
# %% read powercurve
power_curve_df = fn.read_power_curve_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')
print('power curve data loaded')

# %% PLOT AIRFOILS
print('plotting airfoils')
#fn.plot_airfoils(airfoil_coords)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area
#V_inflow_initial = 8  # m/s, initial inflow velocity

span_positions = np.linspace(0, ROTOR_RADIUS, 50)  # m, span positions from root (0) to tip
print('constants defined')
# %% Compute induction factors

# %% Step 1: Initialize a and a', typically a=a'=0
# power_curve_df['a'] = 0.0  # axial induction factor
# power_curve_df['a_prime'] = 0.0  # tangential induction factor

# %% Step 2: Step 2: Compute the flow angle ϕ.
flow_angles_df = fn.flow_angle_loop(span_positions, power_curve_df)
print(flow_angles_df.head())
print('flow angles computed')
# %% Step 3: Compute the local angle of attack α.
# %% Step 4: Compute Cl(α) and Cd(α) by interpolation based on the airfoil polars.
# %% Step 5: Compute Cn and Ct.
# %% Step 6: Update a and a′.
# %% Step 7: If a and a′ change beyond a set tolerance, return to Step 2; otherwise, continue.
# %% Step 8: Compute the local contribution to thrust and torque.
# %% Loop over all blade elements, integrate to get thrust (T) and torque (M), then compute power output.
# %% Compute thrust coefficient CT and power coefficient CP.

# %% Compute lift and drag coefficients
# Define span positions and angles of attack
n_r = 50  # number of radial positions
n_alpha = 36  # number of angle of attack points
r_positions = np.linspace(0, ROTOR_RADIUS, n_r)  # m, radial positions from root (0) to tip
alpha_range = np.linspace(-10, 25, n_alpha)  # degrees

# Initialize matrices to store results
Cl_matrix = np.zeros((len(r_positions), len(alpha_range)))
Cd_matrix = np.zeros((len(r_positions), len(alpha_range)))

# Get sorted list of all available airfoil sections
available_sections = sorted(list(airfoil_polar_data.keys()))
n_sections = len(available_sections)
print(f"Using {n_sections} airfoil sections")

# Compute Cl and Cd for each position and angle
for i, r in enumerate(r_positions):
    # Normalize radius for airfoil selection
    r_norm = r / ROTOR_RADIUS
    
    # Map normalized radius to available airfoil section
    section_index = int(round(r_norm * (n_sections - 1)))
    section = available_sections[section_index]
    
    # Get polar data for this section
    polar = airfoil_polar_data[section]
    
    # Interpolate Cl and Cd for each angle of attack
    Cl_matrix[i, :] = np.interp(alpha_range, polar['Alpha'], polar['Cl'])
    Cd_matrix[i, :] = np.interp(alpha_range, polar['Alpha'], polar['Cd'])

# Plot results
plt.figure(figsize=(15, 6))

# Plot Cl
plt.subplot(121)
contour_cl = plt.contourf(alpha_range, r_positions/ROTOR_RADIUS, Cl_matrix, levels=20, cmap='viridis')
plt.colorbar(contour_cl, label='Cl')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('r/R')
plt.title('Lift Coefficient Distribution')
plt.grid(True)

# Plot Cd
plt.subplot(122)
contour_cd = plt.contourf(alpha_range, r_positions/ROTOR_RADIUS, Cd_matrix, levels=20, cmap='viridis')
plt.colorbar(contour_cd, label='Cd')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('r/R')
plt.title('Drag Coefficient Distribution')
plt.grid(True)

plt.tight_layout()

# Save the plot
save_path = os.path.join(pictures_dir, 'Cl_Cd_Distribution.png')
plt.savefig(save_path)
if show_plot:
    plt.show()
plt.close()