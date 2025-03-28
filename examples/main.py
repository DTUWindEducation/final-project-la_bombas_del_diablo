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

# Dictionary to store DataFrames
airfoil_data = {}

# Check if directory exists
if not airfoils_dir.exists():
    print(f"Directory not found: {airfoils_dir}")
    print(f"Current working directory: {Path.cwd()}")
else:
    # Loop through all .txt files in the directory
    for file_path in airfoils_dir.glob('*.txt'):
        try:
            # Extract airfoil name from filename
            airfoil_name = file_path.stem
            
            # Read file into DataFrame
            df = fn.read_airfoil_file(file_path)
            
            # Store DataFrame in dictionary
            airfoil_data[airfoil_name] = df
            
            print(f"Successfully read {airfoil_name} with {len(df)} points")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Example: Access a specific airfoil DataFrame
    if airfoil_data:
        # Print names of all airfoils loaded
        print("\nLoaded airfoils:")
        for name in airfoil_data:
            print(f"- {name}")
        
        # Example of using the first airfoil
        first_airfoil = list(airfoil_data.keys())[0]
        print(f"\nFirst 5 rows of {first_airfoil}:")
        print(airfoil_data[first_airfoil].head())
    else:
        print("No airfoil data was loaded.")


# Add this to your main code to load both file types
# Dictionary to store polar DataFrames
airfoil_polar_data = {}

# Load the polar (.dat) files
for file_path in airfoils_dir.glob('*.dat'):
    try:
        # Extract airfoil name from filename
        airfoil_name = file_path.stem
        
        # Read file into DataFrame
        result = fn.read_airfoil_polar_file(file_path)
        
        # Store DataFrame in dictionary
        airfoil_polar_data[airfoil_name] = result
        
        print(f"Successfully read polar data for {airfoil_name} with {len(result['data'])} points")
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

# Example: Access a specific airfoil polar DataFrame
#show the first polar airforl data
print("\nLoaded airfoil polar data:")
airfoil_polar_data_names = list(airfoil_polar_data.keys())
print(airfoil_polar_data_names[0])


# %% PLOT AIRFOILS
#first_airfoil = list(airfoil_data.keys())[0]
#print(f"\nFirst 5 rows of {first_airfoil}:")
#print(airfoil_data[first_airfoil].head())
print('plotting airfoils')
# Define this variable before the loop - set to True if you want to display plots
show_plot = False  # Change to True to show plots interactively

print('plotting airfoils')
for airfoil_name in airfoil_data:
    plt.figure(figsize=(10, 6))
    
    # Use the current airfoil from the loop instead of hardcoding
    plt.scatter(airfoil_data[airfoil_name]['x/c'], 
              airfoil_data[airfoil_name]['y/c'], 
              s=10, 
              label=f'Airfoil {airfoil_name.split("_")[1].replace("AF", "")}')
    
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(f'Airfoil Geometry for {airfoil_name}')
    plt.legend()
    plt.grid(True)
    
    # Define the path to save the figure in the results/Pictures folder
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists
    
    # Extract airfoil number for the filename
    airfoil_num = airfoil_name.split("_")[1].replace("AF", "")
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
