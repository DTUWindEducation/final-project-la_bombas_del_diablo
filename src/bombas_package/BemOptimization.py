"""Class for the final project"""

# from pathlib import Path
import numpy as np
# import pandas as pd
import src.bombas_package.utils.functions as fn  # Custom functions for BEM calculations


class BemOptimization:
    """Class for BEM optimization."""

    def __init__(self, index, power_curve_df, blade_data_df):
        """Initialize the BEM optimization instance."""
        self.index = index
        self.wind_speed = power_curve_df['wind_speed'].iloc[index]
        self.rotational_speed = (
            power_curve_df['rot_speed'].iloc[index] * ((2 * np.pi) / 60)
        )
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

        # flow_angles_rad, _ = fn.compute_flow_angle(
        #     elements_df,
        #     self.wind_speed,
        #     self.rotational_speed
        # )
        # elements_df['flow_angles'] = flow_angles_rad

        return elements_df

    def optimize_induction_factors(self, elements_df, airfoil_polar,
                                   convergence_reached, BLADES_NO,
                                   ROTOR_RADIUS, iteration_counter,
                                   max_iterations, tolerance=1e-3):
        """
        Optimize induction factors using the BEM method.

        This method iteratively updates the induction factors until 
        convergence is reached.

        Parameters
        ----------
        elements_df : pd.DataFrame
            DataFrame containing blade element data.
        airfoil_polar : pd.DataFrame
            DataFrame containing airfoil performance data.
        convergence_reached : bool
            Flag to check if the solution converges.
        BLADES_NO : int
            Number of blades.
        ROTOR_RADIUS : float
            Rotor radius.
        iteration_counter : int
            Count of iterations performed.
        max_iterations : int
            Maximum number of iterations allowed.
        tolerance : float, optional
            Convergence tolerance (default is 1e-3).

        Returns
        -------
        None
            Updates the elements_df with optimized induction 
            factors and other parameters directly in the object.
        """
        print(
            f'\nInitializing induction factor optimization loop '
            f'for wind speed: {self.wind_speed}'
        )

        while not convergence_reached and iteration_counter < max_iterations:

            # Compute flow angles and put in the dataframe (df)
            elements_df['flow_angles_rad'], elements_df['flow_angles_deg'] = (
                fn.compute_flow_angle(elements_df, self.wind_speed,
                                      self.rotational_speed))

            # Compute local angle of attack and put in the df
            (elements_df['local_angle_of_attack_rad'],
             elements_df['local_angle_of_attack_deg']) = (
                 fn.compute_local_angle_of_attack(
                elements_df, self.pitch_deg
            ))

            # 3. Compute lift coefficient (Cl) and drag coefficient (Cd)
            # as function of span position (r) and angle of attack (α)
            # using the airfoil polar data and the blade geometry

        
            # Interpolate lift and drag coefficients (And put in df)
            elements_df['Cl'], elements_df['Cd'] = fn.interpolate_Cl_Cd_coeff(
                elements_df, airfoil_polar
            )

            # Compute normal and tangential coefficients (And put in df)
            elements_df['Cn'] = fn.compute_normal_coeff(elements_df)
            elements_df['Ct'] = fn.compute_tangential_coeff(elements_df)

            # Compute local solidity (And put in df)
            elements_df['local_solidity'] = fn.compute_local_solidity(elements_df)

            # Compute prandtl factor for corrections (And put in df)
            elements_df['prandtl_factor'] = fn.prandtl_correction(
                elements_df, BLADES_NO, ROTOR_RADIUS
            )

            # Compute dCT (and put in df)
            elements_df['delta_thrust_coeff'] = fn.update_delta_thrust_coeff(
                elements_df
            )


            # 4. Compute the axial (a) and tangential (a′) induction factors
            # as function of span position (r), the inflow wind speed V0,
            #  the blade pitch angle (θp) and the rotational speed ω.

            # Update the induction factors (And put in df)
            elements_df['axial_induction_new'] = fn.update_axial(elements_df)
            elements_df['tangential_induction_new'] = fn.update_tangential(elements_df)

            # Check convergence and update the induction factors in the df
            convergence_reached, elements_df, iteration_counter = fn.check_convergence(
                elements_df, tolerance, iteration_counter, convergence_reached
            )

            # Avoid infinite loop by limiting the number of iterations
            if iteration_counter == max_iterations:
                print('Maximum iterations reached, stopping loop')

        # When convergence is reached, update induction factors of the object
        self.elements_df = elements_df
        # Assign convergence status and iteration count to the object
        self.convergence_reached = convergence_reached
        self.iteration_counter = iteration_counter

        return None

    def calculate_thrust_and_power(self, elements_df, RHO, BLADES_NO, A):
        """
        Calculate thrust and power based on converged induction factors.

        This method computes the differential thrust and torque
        for each blade element, integrates them to find total thrust
        and torque, and then calculates the overall aerodynamic
        performance metrics of the turbine.

        Parameters
        ----------
        elements_df : pd.DataFrame
            DataFrame containing blade element data with converged
            induction factors.
        RHO : float
            Air density in kg/m³.
        BLADES_NO : int
            Number of blades on the turbine.
        A : float
            Rotor swept area in m².

        Returns
        -------
        None
            Updates the instance attributes with calculated thrust, torque,
            aerodynamic power, thrust coefficient, and power coefficient
            directly in the object.
        """
        # Calculate radial step size (dr) for each element using numpy diff
        # The prepend ensures we have a value for the first element
        dr_values = np.diff(
            elements_df['span_position'].values,
            prepend=elements_df['span_position'].values[0]
        )

        # Calculate differential thrust (dT) for each blade element
        # # using momentum theory and converged axial induction factors
        elements_df['dT'] = fn.compute_dT(
            elements_df['span_position'].values,
            dr_values,
            RHO,
            self.wind_speed,
            elements_df['axial_induction'].values
        )

        # Calculate differential torque (dM) for each blade element
        # using the tangential forces acting at each radial position
        elements_df['dM'] = fn.compute_dM(
            elements_df['span_position'].values,
            dr_values,
            RHO,
            self.wind_speed,
            elements_df['axial_induction'].values,
            elements_df['tangential_induction'].values,
            self.rotational_speed
        )

        # Sum up the differential thrust and torque contributions
        # for one blade
        total_thrust_per_blade = elements_df['dT'].sum()    # [N]
        total_torque_per_blade = elements_df['dM'].sum()    # [Nm]

        # Scale by number of blades and convert
        # thrust/torque to kilonewtons
        total_thrust = total_thrust_per_blade * BLADES_NO / 1000   # [kN]
        total_torque = total_torque_per_blade * BLADES_NO / 1000   # [kNm]

        # Calculate aerodynamic power from torque and rotational speed
        # P = T * ω where T is torque and ω is rotational speed
        aero_power = fn.compute_aerodynamic_power(
            total_torque,              # Torque [kNm]
            self.rotational_speed      # Rotational speed [rad/s]
        )  # [kW]

        # Calculate non-dimensional thrust coefficient
        # CT = T / (0.5 * ρ * A * V²)
        thrust_coeff = fn.compute_thrust_coeff(
            RHO,                # Air density [kg/m³]
            A,                  # Rotor area [m²]
            self.wind_speed,    # Freestream velocity [m/s]
            total_thrust        # Total thrust [kN]
        )  # [-]

        # Calculate non-dimensional power coefficient
        # CP = P / (0.5 * ρ * A * V³)
        power_coeff = fn.compute_power_coeff(
            RHO,                # Air density [kg/m³]
            A,                  # Rotor area [m²]
            self.wind_speed,    # Freestream velocity [m/s]
            aero_power          # Aerodynamic power [kW]
        )  # [-]

        # Store calculated values as instance attributes for later access
        self.total_thrust = total_thrust      # [kN]
        self.total_torque = total_torque      # [kNm]
        self.aero_power = aero_power          # [kW]
        self.thrust_coeff = thrust_coeff      # [-]
        self.power_coeff = power_coeff        # [-]
