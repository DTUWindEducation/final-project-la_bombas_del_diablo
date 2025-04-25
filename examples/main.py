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
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
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
# blade_span = blade_data_df['BlSpn'].values  # m, span positions from root (0) to tip
# blade_twist = blade_data_df['BlTwist'].values  # degrees, twist angles at corresponding span positions



fn.plot_airfoils_3d(airfoil_coords, blade_data_df['BlSpn'], blade_data_df['BlTwist'], show_plot=False)
# fn.plot_airfoils(airfoil_coords)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area
POWER_CURVE_INDICE = 4
V_INFLOW = power_curve_df['wind_speed'].iloc[POWER_CURVE_INDICE]  # m/s, initial inflow velocity
ROTATIONAL_SPEED = power_curve_df['rot_speed'].iloc[POWER_CURVE_INDICE]*((2*pi)/60)  # RAD/S, initial rotational speed
PITCH_ANGLE_DEG = power_curve_df['pitch'].iloc[POWER_CURVE_INDICE]  # degrees, initial pitch angle
PITCH_ANGLE_RAD = PITCH_ANGLE_DEG * (pi / 180)  # convert to radians
# print('V_INFLOW:', V_INFLOW)
# print('ROTATIONAL_SPEED:', ROTATIONAL_SPEED)
# print('PITCH_ANGLE:', PITCH_ANGLE)

print(' \n constants defined')

# %% Step 1 and 2: Step 2: Compute the flow angle ϕ.
axial_induction = 0.0  # axial induction factor (assumed constant for simplicity)
tangential_induction = 0.0  # tangential induction factor (assumed constant for simplicity)

#compute flow angle for all blade elements in radians
flow_angles = fn.compute_flow_angle(blade_data_df, 'BlSpn', axial_induction,
                                    tangential_induction, V_INFLOW, ROTATIONAL_SPEED)
# print(flow_angles*180/pi)
print(' \n flow angles computed')
# %% Step 3: Compute the local angle of attack α.
local_angle_of_attack, local_angle_of_attack_deg  = fn.compute_local_angle_of_attack(flow_angles, PITCH_ANGLE_RAD, blade_data_df,
                                                                                     'BlTwist')

print(' \n local angle of attack array (deg):')
print(local_angle_of_attack_deg)
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

# print(angles_df)
# %% Step 4: Compute Cl(α) and Cd(α) by interpolation based on the airfoil polars.
#lets use one airfoil, fx the first one
chosen_airfoil = airfoil_polar['25'] #take the middle airfoil
airfoil_df = pd.DataFrame(chosen_airfoil)
# print(' \n airfoil df shape:', airfoil_df.shape)
# print(airfoil_df.head())
# print(airfoil_df.columns.tolist())

#Interpolate airfoil data at each blade element's angle of attack:
# print(' check interpolation function (debugging)')

# print('local angle of attack:', local_angle_of_attack_deg[-1])
# Cl = np.interp(local_angle_of_attack_deg[-1], airfoil_df['Alpha'], airfoil_df['Cl'])
# Cd = np.interp(local_angle_of_attack_deg[-1], airfoil_df['Alpha'], airfoil_df['Cd'])
# print(f'Cl: {Cl}')
# print(f'Cd: {Cd}')

angles_df['Cl'] = np.interp(local_angle_of_attack_deg, airfoil_df['Alpha'], airfoil_df['Cl'])
angles_df['Cd'] = np.interp(local_angle_of_attack_deg, airfoil_df['Alpha'], airfoil_df['Cd'])


# %% Step 5: Compute Cn and Ct.
angles_df['Cn'] = fn.compute_normal_coeff(angles_df['Cl'], angles_df['Cd'], angles_df['flow_angle_rad']) #50 values
angles_df['Ct'] = fn.compute_tangential_coeff(angles_df['Cl'], angles_df['Cd'], angles_df['flow_angle_rad']) #50 values

print('n \Cn and Ct Computed')
# print(angles_df['Cn'])
# print(angles_df['Ct'])


# %% Step 6: Update a and a′.
angles_df['local_solidity'] = fn.compute_local_solidity(blade_data_df, 'BlChord', 'BlSpn')
# print(angles_df['local_solidity'])

angles_df['axial_induction'] = fn.update_axial(angles_df, 'flow_angle_rad', 'local_solidity', 'Cn', BLADES_NO, ROTOR_RADIUS)

print(angles_df['axial_induction'])

angles_df['tangential_induction'] = fn.update_tangential(angles_df, 'flow_angle_rad', 'local_solidity', 'Ct')
print(angles_df['tangential_induction'])

# print("Flow angle range:", angles_df['flow_angle_deg'])
# print("Local Solidity Range (sigma):", angles_df['local_solidity'])
# print("Tangential Force Coefficient (Ct) Range:", angles_df['Ct'])




# %% Step 7: If a and a′ change beyond a set tolerance, return to Step 2; otherwise, continue.

# %% Step 8: Compute the local contribution to thrust and torque.

# Initialize arrays to store dT and dM values for each element
num_elements = len(angles_df)
dT_values = np.zeros(num_elements)
dM_values = np.zeros(num_elements)

# Calculate dr for all elements
# make array of differential span values
dr_values = np.diff(angles_df['span_position'].values, prepend=angles_df['span_position'].values[0]) #

# Calculate differential thrust and torque for all elements at once
dT_values = fn.compute_dT(angles_df['span_position'].values,
                         dr_values,
                         RHO,
                         V_INFLOW,
                         angles_df['axial_induction'].values)

dM_values = fn.compute_dM(angles_df['span_position'].values,
                         dr_values,
                         RHO,
                         V_INFLOW,
                         angles_df['axial_induction'].values,
                         angles_df['tangential_induction'].values,
                         ROTATIONAL_SPEED)

# Add dT and dM values to the dataframe
angles_df['dT'] = dT_values
angles_df['dM'] = dM_values

print("\nDifferential thrust and torque values along blade:")
# print(angles_df[['span_position', 'dT', 'dM']])
print(' \n dT and dM computed')

# Calculate total thrust and torque for one blade
total_thrust_one_blade = np.sum(dT_values)
total_torque_one_blade = np.sum(dM_values)

# Calculate total thrust and torque for all blades using the new function
total_thrust = total_thrust_one_blade * BLADES_NO
total_torque = total_torque_one_blade * BLADES_NO

# Calculate aerodynamic power
aero_power = fn.compute_aerodynamic_power(total_torque, ROTATIONAL_SPEED)

print(' \n Power computed')

# Calculate thrust and power coefficients
thrust_coeff = fn.compute_thrust_coeff(RHO, A, V_INFLOW, total_thrust)
power_coeff = fn.compute_power_coeff(RHO, A, V_INFLOW, aero_power)

print("\nResults:")
print(f"Total thrust: {total_thrust:.2f} N")
print(f"Total torque: {total_torque:.2f} N·m")
print(f"Aerodynamic power: {aero_power/1e6:.2f} MW")
print(f"Thrust coefficient (CT): {thrust_coeff:.3f}")
print(f"Power coefficient (CP): {power_coeff:.3f}")

print(' \n CT and CP computed')

# %% Loop over all blade elements, integrate to get thrust (T) and torque (M), then compute power output.
# %% Compute thrust coefficient CT and power coefficient CP.

# print range of each column in angles_df
print("\nRange of each column in angles_df:")
for col in angles_df.columns:
    print(f"{col}: {angles_df[col].min():.2f} to {angles_df[col].max():.2f}")

# plot cl/cd vs alpha
fn.plot_clcd_vs_alpha(angles_df)

#plot cd vs cl
fn.plot_cd_cl(angles_df)


print(' \n finished main.py')

