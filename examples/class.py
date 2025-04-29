"""Script for the final project"""

from pathlib import Path
import numpy as np
import pandas as pd
import functions as fn

class BemOptimization:
    """Class for BEM optimization"""

    def __init__(self, index):
        """Initialize the BEM optimization instance."""
        self.index = index
        self.wind_speed = power_curve_df['wind_speed'].iloc[index]
        self.rotational_speed = power_curve_df['rot_speed'].iloc[index] * ((2 * np.pi) / 60)
        self.pitch_rad = power_curve_df['pitch'].iloc[index]
        self.pitch_deg = self.pitch_rad * (np.pi / 180)
        self.elements_df = self.initialize_elements_df()
        self.convergence_reached = False
        self.iteration_counter = 0

    def initialize_elements_df(self):
        """Initialize the elements DataFrame."""
        elements_df = pd.DataFrame({
            'axial_induction': np.zeros(50),
            'tangential_induction': np.zeros(50),
            'span_position': blade_data_df['BlSpn']
        })
        return elements_df

    def optimize_induction_factors(self, elements_df, convergence_reached, iteration_counter,
                                   max_iterations, tolerance=1e-3):
        """Optimize induction factors."""
        print(f'\nInitializing induction factor optimization loop for wind speed: {self.wind_speed}')
        while not convergence_reached and iteration_counter < max_iterations:
            elements_df['flow_angle_rad'], elements_df['flow_angle_deg'] = fn.compute_flow_angle(
                elements_df, self.wind_speed, self.rotational_speed)
            elements_df['local_angle_of_attack_rad'], elements_df['local_angle_of_attack_deg'] = (
                fn.compute_local_angle_of_attack(
                    elements_df['flow_angle_rad'], self.pitch_deg, blade_data_df, 'BlTwist'
                )
            )
            elements_df['Cl'], elements_df['Cd'] = fn.interpolate_Cl_Cd_coeff(elements_df, airfoil_polar)
            elements_df['Cn'] = fn.compute_normal_coeff(elements_df['Cl'], elements_df['Cd'],
                                                        elements_df['flow_angle_rad'])
            elements_df['Ct'] = fn.compute_tangential_coeff(elements_df['Cl'], elements_df['Cd'],
                                                            elements_df['flow_angle_rad'])
            elements_df['local_solidity'] = fn.compute_local_solidity(blade_data_df, 'BlChord', 'BlSpn')
            elements_df['prandtl_factor'] = fn.prandtl_correction(elements_df, BLADES_NO, ROTOR_RADIUS)
            elements_df['delta_thrust_coeff'] = fn.update_delta_thrust_coeff(elements_df)
            elements_df['axial_induction_new'] = fn.update_axial_joe(elements_df)
            elements_df['tangential_induction_new'] = fn.update_tangential_joe(elements_df)
            convergence_reached, elements_df, iteration_counter = fn.check_convergence(
                elements_df, tolerance, iteration_counter, convergence_reached)
            if iteration_counter == max_iterations:
                print('Maximum iterations reached, stopping loop')
        self.elements_df = elements_df
        self.convergence_reached = convergence_reached
        self.iteration_counter = iteration_counter

    def calculate_thrust_and_power(self, elements_df):
        """Calculate thrust and power."""
        dr_values = np.diff(elements_df['span_position'].values,
                            prepend=elements_df['span_position'].values[0])
        elements_df['dT'] = fn.compute_dT(elements_df['span_position'].values, dr_values, RHO,
                                          self.wind_speed, elements_df['axial_induction'].values)
        elements_df['dM'] = fn.compute_dM(elements_df['span_position'].values, dr_values, RHO,
                                          self.wind_speed, elements_df['axial_induction'].values,
                                          elements_df['tangential_induction'].values,
                                          self.rotational_speed)
        total_thrust_per_blade = elements_df['dT'].sum()
        total_torque_per_blade = elements_df['dM'].sum()
        total_thrust = total_thrust_per_blade * BLADES_NO / 1000
        total_torque = total_torque_per_blade * BLADES_NO / 1000
        aero_power = fn.compute_aerodynamic_power(total_torque, self.rotational_speed)
        thrust_coeff = fn.compute_thrust_coeff(RHO, A, self.wind_speed, total_thrust)
        power_coeff = fn.compute_power_coeff(RHO, A, self.wind_speed, aero_power)
        self.total_thrust = total_thrust
        self.total_torque = total_torque
        self.aero_power = aero_power
        self.thrust_coeff = thrust_coeff
        self.power_coeff = power_coeff


# Constants
RHO = 1.225
ROTOR_RADIUS = 240 / 2
BLADES_NO = 3
A = np.pi * ROTOR_RADIUS ** 2

# Load data
DATA_DIR = Path(__file__).resolve().parent.parent
airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils')
power_curve_df = fn.read_power_curve_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')
blade_data_df = fn.read_blade_data_file(DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA-15-240-RWT_AeroDyn15_blade.dat')

# Optimization loop
optimization_instances = {}
for i in range(len(power_curve_df)):
    instance = BemOptimization(i)
    instance.optimize_induction_factors(instance.elements_df, instance.convergence_reached,
                                        instance.iteration_counter, max_iterations=100, tolerance=1e-3)
    instance.calculate_thrust_and_power(instance.elements_df)
    optimization_instances[i] = instance
    print(f'Finished optimization for wind speed: {instance.wind_speed}')

# Store results
results_df = power_curve_df.copy()
for i in range(len(optimization_instances)):
    results_df.loc[i, 'total_thrust_bem'] = optimization_instances[i].total_thrust
    results_df.loc[i, 'total_torque_bem'] = optimization_instances[i].total_torque
    results_df.loc[i, 'aero_power_bem'] = optimization_instances[i].aero_power
    results_df.loc[i, 'thrust_coeff_bem'] = optimization_instances[i].thrust_coeff
    results_df.loc[i, 'power_coeff_bem'] = optimization_instances[i].power_coeff

# Filter converged results
convergence_mask = np.array([optimization_instances[i].convergence_reached for i in range(len(optimization_instances))])
converged_results_df = results_df[convergence_mask].copy()
converged_results_df['aero_power_bem'] /= 1000
converged_results_df['aero_power'] /= 1000

# Plot results
fn.plot_results_vs_ws(converged_results_df, 'aero_power', 'Reference power curve', 'aero_power_bem',
                      'BEM power curve', 'Power [MW]')
fn.plot_results_vs_ws(converged_results_df, 'aero_thrust', 'Reference thrust curve', 'total_thrust_bem',
                      'BEM thrust curve', 'Thrust [N]')