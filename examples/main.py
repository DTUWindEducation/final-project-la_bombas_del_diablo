# %% imports
print('\n initializing main.py')
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
print(' \n imported libraries')

# %% Load airfoil data
# Define data directory - will work on any computer
DATA_DIR = Path(__file__).resolve().parent.parent
#print(f'Data directory: {DATA_DIR}')

# Read the coordinates and polar data for the airfoils
airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils')
print(' \n airfoil data loaded')
# %% read powercurve
power_curve_df = fn.read_power_curve_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')
# print(' \n power curve dataframe:')
# print(' \n power curve shape:', power_curve_df.shape)
print(' \n power curve data loaded')
print("Power curve columns:", power_curve_df.columns.tolist())
# %% read blade data
blade_data_df = fn.read_blade_data_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA-15-240-RWT_AeroDyn15_blade.dat')
# print(' \n blade data dataframe:')
print(' \n blade data loaded')
# print(' \n blade data shape:', blade_data_df.shape)
# print(blade_data_df.head())

# %% PLOT AIRFOILS
#print(' \n plotting airfoils')
#incorporate blade span and twist angle in 3d plot
#fn.plot_airfoils(airfoil_coords)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area
V_INFLOW = power_curve_df['wind_speed'].iloc[5]  # m/s, initial inflow velocity
ROTATIONAL_SPEED = power_curve_df['rot_speed'].iloc[5]*(60/(2*pi))  # RAD/S, initial rotational speed
print('V_INFLOW:', V_INFLOW)
print('ROTATIONAL_SPEED:', ROTATIONAL_SPEED)

print(' \n constants defined')

# %% Step 1 and 2: Step 2: Compute the flow angle ϕ.
span_positions = blade_data_df['BlSpn'].values  # m, span positions from root (0) to tip
flow_angles_df = fn.flow_angle_loop(span_positions, V_INFLOW, ROTATIONAL_SPEED)
print(' \n flow angles df:')
print(flow_angles_df.head())
# print(shape(flow_angles_df))
print(' \n flow angles shape:', flow_angles_df.shape)
print(' \n flow angles computed')
# %% Step 3: Compute the local angle of attack α.
# Get pitch angle for V_INFLOW
pitch_angle = power_curve_df['pitch'].iloc[5]  # Get single pitch angle value
# Compute local angle of attack using single wind speed data
df_local_angle_of_attack = fn.compute_local_angle_of_attack(flow_angles_df, pitch_angle, blade_data_df)

print(' \n local angle of attack dataframe:')
print(df_local_angle_of_attack.head())
print(' \n local angle of attack shape:', df_local_angle_of_attack.shape)
print(' \n local angle of attack computed')

# %% Step 4: Compute Cl(α) and Cd(α) by interpolation based on the airfoil polars.
# we already hàve them??

# %% Step 5: Compute Cn and Ct.
# Cl = 200x50 values, (50 airfoils and 200 values for each ) 
# Cd = 200x50 values,
fn.compute_Cn

# %% Step 6: Update a and a′.
# %% Step 7: If a and a′ change beyond a set tolerance, return to Step 2; otherwise, continue.
# %% Step 8: Compute the local contribution to thrust and torque.
# %% Loop over all blade elements, integrate to get thrust (T) and torque (M), then compute power output.
# %% Compute thrust coefficient CT and power coefficient CP.

