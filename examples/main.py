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
print(f'Data directory: {DATA_DIR}')


# Path to airfoils directory - properly using Path objects for cross-platform compatibility
airfoils_dir = DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils'
print(f'Airfoils directory: {airfoils_dir}')

# Initialize dictionaries to store data - ADD THESE LINES
airfoil_data = {}
airfoil_polar_data = {}

# For airfoil coordinate files (.txt)
for file_path in airfoils_dir.glob('*.txt'):
    try:
        # Get both the airfoil number and data
        airfoil_num, df = fn.read_airfoil_file(file_path)
        
        # Store DataFrame in dictionary using simplified key
        airfoil_data[airfoil_num] = df
        
        #print(f"Successfully read airfoil {airfoil_num} with {len(df)} points")
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

print(f"Successfully read airfoil coordinates data. Amount of airfoil data: {len(airfoil_data)}")
print(airfoil_num)
print(f' head of airfoil coords data {airfoil_data["00"].head()}')



# For polar files (.dat)
for file_path in airfoils_dir.glob('*.dat'):
    try:
        # Get both the airfoil number and data
        airfoil_num, df = fn.read_airfoil_polar_file(file_path)
        
        # Store DataFrame in dictionary using simplified key
        airfoil_polar_data[airfoil_num] = df
        
        #print(f"Successfully read polar data for airfoil {airfoil_num} with {len(df)} points")
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

print(f"Successfully read polar data. Amount of airfoil data: {len(airfoil_polar_data)}")
print(f' head of airfoil polar data {airfoil_polar_data["00"].head()}')  # Now works directly


# %% PLOT AIRFOILS

# %% PLOT AIRFOILS
print('plotting airfoils')
# Define this variable before the loop - set to True if you want to display plots
show_plot = False  # Change to True to show plots interactively

for airfoil_num in airfoil_data:
    plt.figure(figsize=(10, 6))
    
    # Plot using the simplified airfoil numbers
    plt.scatter(airfoil_data[airfoil_num]['x/c'], 
              airfoil_data[airfoil_num]['y/c'], 
              s=10, 
              label=f'Airfoil {airfoil_num}')
    
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(f'Airfoil Geometry {airfoil_num}')
    plt.legend()
    plt.grid(True)
    
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists
    
    save_path = os.path.join(pictures_dir, f'Airfoil_Geometry_{airfoil_num}.png')
    plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    plt.close()
print(f"Saved {len(airfoil_data)} airfoil plots to {pictures_dir}")



# %% Compute lift coefficient (Cl) and drag coefficient (Cd)
#  as function of span position (r) and angle of attack (α)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area

# ...existing code...
A = pi*ROTOR_RADIUS**2  # m^2, rotor area

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


# %% Compute induction factors
