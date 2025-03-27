# %% imports
print('initializing main.py')
"""Script for the final project"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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


# %% PLOT AIRFOILS
#first_airfoil = list(airfoil_data.keys())[0]
#print(f"\nFirst 5 rows of {first_airfoil}:")
#print(airfoil_data[first_airfoil].head())
print('plotting airfoils')
plt.figure(figsize=(10, 6))
plt.scatter(airfoil_data['IEA-15-240-RWT_AF49_Coords']['x/c'], airfoil_data['IEA-15-240-RWT_AF49_Coords']['y/c'], s=10, label='Airfoil 49')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.title('Airfoil Geometry for Airfoil 0')
plt.legend()
plt.grid(True)
plt.show()

