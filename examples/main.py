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

# Read the coordinates and polar data for the airfoils
airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(airfoils_dir)

# %% PLOT AIRFOILS
print('plotting airfoils')
fn.plot_airfoils(airfoil_coords)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area

# %% Step 1: Initialize a and a', typically a=a'=0
a = 0.0  # axial induction factor
a_prime = 0.0  # tangential induction factor