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
# print(' \n airfoil data loaded')
# %% read powercurve
power_curve_df = fn.read_power_curve_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')
# print(' \n power curve dataframe:')
# print(' \n power curve shape:', power_curve_df.shape)
print(' \n power curve data loaded')
# print("Power curve columns:", power_curve_df.columns.tolist())
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
V_INFLOW = power_curve_df['wind_speed'].iloc[4]  # m/s, initial inflow velocity
ROTATIONAL_SPEED = power_curve_df['rot_speed'].iloc[4]*(60/(2*pi))  # RAD/S, initial rotational speed
PITCH_ANGLE = power_curve_df['pitch'].iloc[4]  # degrees, initial pitch angle
# print('V_INFLOW:', V_INFLOW)
# print('ROTATIONAL_SPEED:', ROTATIONAL_SPEED)
# print('PITCH_ANGLE:', PITCH_ANGLE)

print(' \n constants defined')

# %% Step 1 and 2: Step 2: Compute the flow angle ϕ.
# span_positions = blade_data_df['BlSpn'].values  # m, span positions from root (0) to tip
axial_induction = 0.0  # axial induction factor (assumed constant for simplicity)
tangential_induction = 0.0  # tangential induction factor (assumed constant for simplicity)

flow_angles = fn.compute_flow_angle(blade_data_df, 'BlSpn', axial_induction, tangential_induction,
                                    V_INFLOW, ROTATIONAL_SPEED)
# flow_angles_df = fn.flow_angle_loop(span_positions, V_INFLOW, ROTATIONAL_SPEED)
# print(' \n flow angles df:')
# print(flow_angles)
# print(shape(flow_angles_df))
# print(' \n flow angles shape:', flow_angles.shape)
print(' \n flow angles computed')
# %% Step 3: Compute the local angle of attack α.
# Get pitch angle for V_INFLOW
# Compute local angle of attack using single wind speed data
local_angle_of_attack, local_angle_of_attack_deg  = fn.compute_local_angle_of_attack(flow_angles, PITCH_ANGLE, blade_data_df, 'BlTwist')

# print(' \n local angle of attack array (deg):')
# print(local_angle_of_attack_deg)
# print(' \n local angle of attack shape:', local_angle_of_attack_deg.shape)
print(' \n local angle of attack computed')

# Put angles into a dataframe
angles_df = pd.DataFrame({
    'span_position': blade_data_df['BlSpn'],
    'flow_angle_deg': np.degrees(flow_angles),
    'local_angle_of_attack_deg': local_angle_of_attack_deg,
    'flow_angle_rad': flow_angles,
    'local_angle_of_attack_rad': local_angle_of_attack
    
})

# print(angles_df.head())
# %% Step 4: Compute Cl(α) and Cd(α) by interpolation based on the airfoil polars.
#lets use one airfoil, fx the first one
chosen_airfoil = airfoil_polar['00']
airfoil_df = pd.DataFrame(chosen_airfoil)
# print(' \n airfoil df shape:', airfoil_df.shape)
# print(airfoil_df.head())
# print(airfoil_df.columns.tolist())

#Interpolate airfoil data at each blade element's angle of attack:

angles_df['Cl'] = np.interp(local_angle_of_attack_deg, airfoil_df['Alpha'], airfoil_df['Cl'])
angles_df['Cd'] = np.interp(local_angle_of_attack_deg, airfoil_df['Alpha'], airfoil_df['Cd'])



#airfoil_df['Cn'] = fn.compute_Cn(airfoil_df['Cl'], airfoil_df['Cd'],
                                #   local_angle_of_attack, flow_angles, ROTATIONAL_SPEED, V_INFLOW)


# %% Step 5: Compute Cn and Ct.
angles_df['Cn'] = fn.compute_Cn(angles_df['Cl'], angles_df['Cd'], angles_df['flow_angle_rad'])
angles_df['Ct'] = fn.compute_Ct(angles_df['Cl'], angles_df['Cd'], angles_df['flow_angle_rad'])

print('n \Cn and Ct Computed')
print(angles_df['Cn'])
print(angles_df['Ct'])
# Cl = 200x50 values, (50 airfoils and 200 values for each ) 
# Cd = 200x50 values,
# fn.compute_Cn

# %% Step 6: Update a and a′.
angles_df['local_solidity'] = fn.compute_local_solidity(angles_df, blade_data_df, 'BlChord', 'span_position')
print(angles_df['local_solidity'])

angles_df['axial_induction'] = fn.update_axial(angles_df, 'flow_angle_rad', 'local_solidity', 'Cn')

print(angles_df['axial_induction'])

angles_df['tangential_induction'] = fn.update_tangential(angles_df, 'flow_angle_rad', 'local_solidity', 'Ct')
print(angles_df['tangential_induction'])

# %% Step 7: If a and a′ change beyond a set tolerance, return to Step 2; otherwise, continue.

# %% Step 8: Compute the local contribution to thrust and torque.
# %% Loop over all blade elements, integrate to get thrust (T) and torque (M), then compute power output.
# %% Compute thrust coefficient CT and power coefficient CP.

