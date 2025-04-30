"""Class for the final project"""

from pathlib import Path
import numpy as np
import pandas as pd
from src import functions as fn


class BemOptimization:
    """Class for BEM optimization"""

    def __init__(self, index, power_curve_df, blade_data_df):
        """Initialize the BEM optimization instance."""
        self.index = index
        self.wind_speed = power_curve_df['wind_speed'].iloc[index]
        self.rotational_speed = (
            power_curve_df['rot_speed'].iloc[index] * ((2 * np.pi) / 60))
        self.pitch_rad = power_curve_df['pitch'].iloc[index]
        self.pitch_deg = self.pitch_rad * (np.pi / 180)
        self.elements_df = self.initialize_elements_df(blade_data_df)
        self.convergence_reached = False
        self.iteration_counter = 0

         # Initialize attributes that will be set later
        self.total_thrust = None
        self.total_torque = None
        self.aero_power = None
        self.thrust_coeff = None
        self.power_coeff = None

    def initialize_elements_df(self, blade_data_df):
        """Initialize the elements DataFrame."""
        elements_df = blade_data_df.copy()
        elements_df['axial_induction'] = np.zeros(len(elements_df))
        elements_df['tangential_induction'] = np.zeros(len(elements_df))
        
        return elements_df

    def optimize_induction_factors(self, elements_df, airfoil_polar, 
                                   convergence_reached, BLADES_NO, 
                                   ROTOR_RADIUS, 
                                   iteration_counter,
                                   max_iterations, tolerance=1e-3):
        
        """Optimize induction factors."""
        print(f'\nInitializing induction factor optimization loop '
              f'for wind speed: {self.wind_speed}')
        while not convergence_reached and iteration_counter < max_iterations:

            (elements_df['flow_angle_rad'], 
             elements_df['flow_angle_deg']) = fn.compute_flow_angle(
                elements_df, self.wind_speed, self.rotational_speed
            )

            (elements_df['local_angle_of_attack_rad'], 
             elements_df['local_angle_of_attack_deg']) = (
                fn.compute_local_angle_of_attack(elements_df, self.pitch_deg)
            )
            elements_df['Cl'], elements_df['Cd'] = fn.interpolate_Cl_Cd_coeff(
                elements_df, airfoil_polar)

            elements_df['Cn'] = fn.compute_normal_coeff(elements_df)   

            elements_df['Ct'] = fn.compute_tangential_coeff(elements_df)
            elements_df['local_solidity'] = (
                fn.compute_local_solidity(elements_df)
            )

            elements_df['prandtl_factor'] = fn.prandtl_correction(elements_df,
                                                                  BLADES_NO,
                                                                  ROTOR_RADIUS)

            elements_df['delta_thrust_coeff'] = fn.update_delta_thrust_coeff(
                elements_df)

            elements_df['axial_induction_new'] = fn.update_axial_joe(
                elements_df)

            elements_df['tangential_induction_new'] = (
                fn.update_tangential_joe(elements_df))

            convergence_reached, elements_df, iteration_counter = (
                fn.check_convergence(elements_df, tolerance,
                                     iteration_counter, convergence_reached))

            if iteration_counter == max_iterations:
                print('Maximum iterations reached, stopping loop')

        self.elements_df = elements_df
        self.convergence_reached = convergence_reached
        self.iteration_counter = iteration_counter

    def calculate_thrust_and_power(self, elements_df, RHO, BLADES_NO, A):
        """Calculate thrust and power."""
        dr_values = np.diff(elements_df['span_position'].values,
                            prepend=elements_df['span_position'].values[0])

        elements_df['dT'] = (
            fn.compute_dT(elements_df['span_position'].values,
                          dr_values, RHO,
                          self.wind_speed,
                          elements_df['axial_induction'].values)
        )
        elements_df['dM'] = (
            fn.compute_dM(elements_df['span_position'].values,
                          dr_values, RHO,
                          self.wind_speed,
                          elements_df['axial_induction'].values,
                          elements_df['tangential_induction'].values,
                          self.rotational_speed)
        )

        total_thrust_per_blade = elements_df['dT'].sum()
        total_torque_per_blade = elements_df['dM'].sum()
        total_thrust = total_thrust_per_blade * BLADES_NO / 1000
        total_torque = total_torque_per_blade * BLADES_NO / 1000

        aero_power = fn.compute_aerodynamic_power(total_torque,
                                                  self.rotational_speed)
        thrust_coeff = fn.compute_thrust_coeff(RHO, A,
                                               self.wind_speed,
                                               total_thrust)

        power_coeff = fn.compute_power_coeff(RHO, A, self.wind_speed,
                                             aero_power)
        self.total_thrust = total_thrust
        self.total_torque = total_torque
        self.aero_power = aero_power
        self.thrust_coeff = thrust_coeff
        self.power_coeff = power_coeff
