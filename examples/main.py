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
