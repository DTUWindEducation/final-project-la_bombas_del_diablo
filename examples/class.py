# imports
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

class Bem_optimization:
    def __init__(self, index):
        self.index = index
        self.wind_speed = power_curve_df['wind_speed'].iloc[index]  # m/s, initial inflow velocity
        self.rotational_speed = power_curve_df['rot_speed'].iloc[index]*((2*pi)/60)  # RAD/S, initial rotational speed
        self.pitch_rad = power_curve_df['pitch'].iloc[index]  # degrees, initial pitch angle
        self.pitch_deg = self.pitch_rad * (pi / 180)  # convert to radians        
        self.elements_df = self.initialize_elements_df()
        self.convergence_reached = False
        self.iteration_counter = 0

    def initialize_elements_df(self):
        elements_df = pd.DataFrame({
        'axial_induction': np.zeros(50),
        'tangential_induction': np.zeros(50),
        'span_position': blade_data_df['BlSpn']
        })

        return elements_df
    
    def optimize_induction_factors(self, elements_df, convergence_reached, 
                                   iteration_counter, MAX_ITERATIONS, TOLERANCE = 1e-3,):
        print('\nInitializing induction factor optimization loop for wind speed:', self.wind_speed)
        
        while convergence_reached == False and iteration_counter < MAX_ITERATIONS:
            #compute flow angle for all blade elements in radians
            elements_df['flow_angle_rad'], elements_df['flow_angle_deg']   = fn.compute_flow_angle(elements_df, self.wind_speed, 
                                                        self.rotational_speed)
            #  Step 3: Compute the local angle of attack α.
            elements_df['local_angle_of_attack_rad'], elements_df['local_angle_of_attack_deg']  = fn.compute_local_angle_of_attack(
                elements_df['flow_angle_rad'], self.pitch_deg, blade_data_df, 'BlTwist')
            #  Step 4: Compute Cl(α) and Cd(α) by interpolation based on the airfoil polars.
            # Use airfoil that matches the span position of the blade element and interpolate Cl and Cd
            elements_df['Cl'], elements_df['Cd']  = fn.interpolate_Cl_Cd_coeff(elements_df, airfoil_polar)
            #  Step 5: Compute Cn and Ct.
            elements_df['Cn'] = fn.compute_normal_coeff(elements_df['Cl'], elements_df['Cd'], elements_df['flow_angle_rad']) 
            elements_df['Ct'] = fn.compute_tangential_coeff(elements_df['Cl'], elements_df['Cd'], elements_df['flow_angle_rad']) 
            #  Step 6: Update a and a′.
            elements_df['local_solidity'] = fn.compute_local_solidity(blade_data_df, 'BlChord', 'BlSpn')
            elements_df['prandtl_factor'] = fn.prandtl_correction(elements_df, BLADES_NO, ROTOR_RADIUS)
            # compute C_T
            elements_df['delta_thrust_coeff'] = fn.update_delta_thrust_coeff(elements_df)
            
            # Correction factor functions to update induction factors
            elements_df['axial_induction_new'] = fn.update_axial_joe(elements_df)
            elements_df['tangential_induction_new'] = fn.update_tangential_joe(elements_df)

            # Given functions to update (no correction factor)
            # elements_df['axial_induction_new'] = fn.update_axial(elements_df)
            # elements_df['tangential_induction_new'] = fn.update_tangential(elements_df)

            # #  Step 7: If a and a′ change beyond a set tolerance, return to Step 2; otherwise, continue.
            # Check if the absolute difference between the new and old values of a and a' is less than the tolerance
            convergence_reached, elements_df, iteration_counter = fn.check_convergence(elements_df, TOLERANCE, iteration_counter,
                                                                                        convergence_reached)
            
            # Check if maximum iterations have been reached
            if iteration_counter == MAX_ITERATIONS:
                print('Maximum iterations reached, stopping loop')

        self.elements_df = elements_df
        self.convergence_reached = convergence_reached
        self.iteration_counter = iteration_counter

        return None
                        
    def calculate_thrust_and_power(self, elements_df):
        # Initialize arrays to store dT and dM values for each element
        num_elements = len(elements_df)
        dT_values = np.zeros(num_elements)
        dM_values = np.zeros(num_elements)

        # Calculate dr for all elements

        # make array of differential span values
        dr_values = np.diff(elements_df['span_position'].values, prepend=elements_df['span_position'].values[0]) # 

        # Calculate differential thrust and torque for all elements at once
        elements_df['dT'] = fn.compute_dT(elements_df['span_position'].values, 
                                dr_values, 
                                RHO, 
                                self.wind_speed, 
                                elements_df['axial_induction'].values) #N

        elements_df['dM'] = fn.compute_dM(elements_df['span_position'].values, 
                                dr_values, 
                                RHO, 
                                self.wind_speed,
                                elements_df['axial_induction'].values,
                                elements_df['tangential_induction'].values,
                                self.rotational_speed) #N*m
        
        # Calculate total thrust and torque for one blade
        total_thrust_per_blade = elements_df['dT'].sum() #N
        total_torque_per_blade = elements_df['dM'].sum()#N*m

        # Calculate total thrust and torque for all blades using the new function
        total_thrust = total_thrust_per_blade * BLADES_NO * 1/1000 #kN
        total_torque = total_torque_per_blade * BLADES_NO * 1/1000 #kN*m

        # Calculate aerodynamic power
        aero_power = fn.compute_aerodynamic_power(total_torque, self.rotational_speed) #kW

        # Calculate thrust and power coefficients
        thrust_coeff = fn.compute_thrust_coeff(RHO, A, self.wind_speed, total_thrust)
        power_coeff = fn.compute_power_coeff(RHO, A, self.wind_speed, aero_power)
        
        print('Finished calculating thrust and power')

        self.total_thrust = total_thrust
        self.total_torque = total_torque
        self.aero_power = aero_power
        self.thrust_coeff = thrust_coeff
        self.power_coeff = power_coeff

        return None

# %% Load data
# Define data directory - will work on any computer
DATA_DIR = Path(__file__).resolve().parent.parent

# Read the coordinates and polar data for the airfoils
airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils')
# read powercurve
power_curve_df = fn.read_power_curve_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')

# read blade data
blade_data_df = fn.read_blade_data_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA-15-240-RWT_AeroDyn15_blade.dat')

# %% PLOT AIRFOILS
fn.plot_airfoils_3d(airfoil_coords, blade_data_df['BlSpn'], blade_data_df['BlTwist'], show_plot=False)

# %% CONSTANTS
# Define constants
RHO = 1.225  # kg/m^3, air density at sea level
ROTOR_RADIUS = 240/2  # m, rotor radius for IEA 15-240 RWT
HUB_HEIGHT = 150 #M, hub height for IEA 15-240 RWT
RATED_POWER = 15e6  # W, rated power for IEA 15-240 RWT
BLADES_NO = 3
A = pi*ROTOR_RADIUS**2  # m^2, rotor area

# Create dictionary to store each optimization instance
optimization_instances = {}

# %% Optimization loop for each wind speed in the power curve
for i in range(len(power_curve_df)):
    # Data frame initialization
    
    instance = Bem_optimization(i)
    instance.optimize_induction_factors(instance.elements_df, instance.convergence_reached,
                                        instance.iteration_counter, MAX_ITERATIONS=100, TOLERANCE=1e-3)
    instance.calculate_thrust_and_power(instance.elements_df)
    optimization_instances[i] = instance
    print(f'Finished optimization for wind speed: {instance.wind_speed}')

# Store results in a DataFrame
results_df = power_curve_df.copy()

# Loop through the optimization instances and store results in the DataFrame
for i in range(len(optimization_instances)):
    results_df.loc[i, 'total_thrust_bem'] = optimization_instances[i].total_thrust
    results_df.loc[i, 'total_torque_bem'] = optimization_instances[i].total_torque
    results_df.loc[i, 'aero_power_bem'] = optimization_instances[i].aero_power
    results_df.loc[i, 'thrust_coeff_bem'] = optimization_instances[i].thrust_coeff
    results_df.loc[i, 'power_coeff_bem'] = optimization_instances[i].power_coeff

print(results_df)

# create a mask for the parameters that have not converged 
convergence_mask = np.array([optimization_instances[i].convergence_reached for i in range(len(optimization_instances))])

# Print summary of convergence
print(f"Converged solutions: {sum(convergence_mask)} out of {len(convergence_mask)}")

# Create filtered dataframe with only converged results
converged_results_df = results_df[convergence_mask].copy()


print(results_df[['wind_speed', 'total_thrust_bem', 'total_torque_bem', 'aero_power_bem', 'thrust_coeff_bem', 'power_coeff_bem']])
# %% PLOT RESULTS
# Plot power reference data vs BEM results (# Plot only converged results)

# Convert power to MW for plotting
converged_results_df['aero_power_bem'] = converged_results_df['aero_power_bem'] / 1000  # kW to MW
converged_results_df['aero_power'] = converged_results_df['aero_power'] / 1000  # kW to MW
# Power Curve Plot
fn.plot_results_vs_ws(converged_results_df, 'aero_power', 'Reference power curve', 'aero_power_bem', 'BEM power curve', 'Power [MW]')
# Thrust Curve Plot
fn.plot_results_vs_ws(converged_results_df, 'aero_thrust', 'Reference thrust curve', 'total_thrust_bem', 'BEM thrust curve', 'Thrust [N]')

# fn.plot_scatter(power_curve_df, 'wind_speed', 'aero_power', 
#                 'Reference_Power_Curve', 'Wind_Speed_[m/s]', 'Power [kW]')