"""Main script for BEM wind turbine simulation

This script implements the Blade Element Momentum (BEM) theory to analyze
wind turbine performance at different wind speeds. The BEM method divides the 
blade into elements and iteratively calculates the forces and flow conditions
to predict power output and thrust.
"""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import src.functions as fn  # Custom functions for BEM calculations
from src.BemOptimization import BemOptimization  # BEM implementation class

#  Physical and turbine constants
RHO = 1.225  # Air density in kg/m³
ROTOR_RADIUS = 240 / 2  # Radius of the rotor in meters (240m diameter)
BLADES_NO = 3  # Number of blades on the turbine
A = np.pi * ROTOR_RADIUS ** 2  # Swept area of the rotor in m²


def main():
    """Main function to run the BEM simulation
    
    This function orchestrates the entire BEM analysis process:
    1. Loads input data (airfoil properties, power curve, blade geometry)
    2. Runs BEM simulation for each wind speed
    3. Processes and plots results
    """
    print("initializing main.py")
    
    # Load all required input data for the simulation
    # Navigate to project root directory
    DATA_DIR = Path(__file__).resolve().parent.parent

    # Create output directories
    results_dir = DATA_DIR / 'outputs' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)  

    # Get airfoil aerodynamic data 
    # (coordinates and polar performance curves)
    airfoil_coords, airfoil_polar = fn.read_all_airfoil_files(
        DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'Airfoils')
    
    # Get wind speeds, pitch angles, rotational speeds, etc.)
    power_curve_df = fn.read_power_curve_file(
        DATA_DIR / 'inputs' / 'IEA-15-240-RWT' / 'IEA_15MW_RWT_Onshore.opt')
    
    # Get span positions, twist angles, chord lengths
    blade_data_df = fn.read_blade_data_file(
        DATA_DIR / 'inputs' / 'IEA-15-240-RWT' /
          'IEA-15-240-RWT_AeroDyn15_blade.dat')
    
    # Create a 3D visualization of the blade with all airfoil profiles
    fn.plot_airfoils_3d(airfoil_coords, blade_data_df['span_position'],
                        blade_data_df['twist_angle'])
    
    # Dictionary to store each BEM simulation instance (one per wind speed)
    optimization_instances = {}
    
    # Run BEM optimization for each wind speed in the power curve
    for i in range(len(power_curve_df)):
        # Create an instance with parameters for this wind speed
        # (including blade geometry, pitch angle, rotational speed)
        instance = BemOptimization(i, power_curve_df, blade_data_df)
        
        # Run the BEM iterative algorithm to find induction factors
        instance.optimize_induction_factors(
            instance.elements_df,  # Dataframe with blade element data
            airfoil_polar,         # Airfoil performance data
            # Flag to check if solution converges
            instance.convergence_reached,
            BLADES_NO,             # Number of blades
            ROTOR_RADIUS,          # Rotor radius
            instance.iteration_counter,  # Count iterations
            max_iterations=100,    # Maximum number of iterations
            tolerance=1e-3         # Convergence tolerance
        )
        
        # Calculate thrust and power using the converged induction factors
        instance.calculate_thrust_and_power(instance.elements_df, 
                                            RHO, BLADES_NO, A)
        
        # Store this instance for later analysis
        optimization_instances[i] = instance
        print(f'Finished optimization for wind speed: {instance.wind_speed}')
    
    # Create a dataframe to store all results for comparison with reference data
    results_df = power_curve_df.copy()  # Start with the reference power curve
    
    # Add BEM calculated values to the results dataframe
    for i in range(len(optimization_instances)):
        # Extract results from each wind speed simulation
        results_df.loc[i, 'total_thrust_bem'] = (
            optimization_instances[i].total_thrust)  # Thrust force [kN]
        results_df.loc[i, 'total_torque_bem'] = (
            optimization_instances[i].total_torque)  # Torque [kNm]
        results_df.loc[i, 'aero_power_bem'] = (
            optimization_instances[i].aero_power)  # Power [kW]
        results_df.loc[i, 'thrust_coeff_bem'] = (
            optimization_instances[i].thrust_coeff)  # Thrust coefficient [-]
        results_df.loc[i, 'power_coeff_bem'] = (
            optimization_instances[i].power_coeff)  # Power coefficient [-]
    
    # Create a mask to identify which wind speeds had converged solutions
    # (Some wind speeds might not reach convergence within the iteration limit)
    convergence_mask = np.array([optimization_instances[i].convergence_reached 
                                for i in range(len(optimization_instances))])
    
    converged_results_df = results_df[convergence_mask].copy()
    # Plot the results and compare with reference data
    # Power curve comparison
    fn.plot_results_vs_ws(converged_results_df, results_df, 'aero_power',
                           'Reference power curve', 'aero_power_bem',
                           'BEM power curve', 'aero_power_bem',
                            'BEM (not converged)', 'Power', '[kW]')
                          
    # Thrust curve comparison
    fn.plot_results_vs_ws(converged_results_df, results_df, 'aero_thrust',
                          'Reference thrust curve', 'total_thrust_bem',
                          'BEM thrust curve', 'total_thrust_bem',
                          'BEM (not converged)', 'Thrust', '[kN]')
    
    # fn.plot_results_vs_ws(converged_results_df, 'aero_power', 
    #                       'Converged Reference power curve', 
    #                       'aero_power_bem', 'BEM power curve', 'Power [kW]')
    # # Thrust curve comparison
    # fn.plot_results_vs_ws(converged_results_df, 'aero_thrust', 
    #                       'Converged Reference thrust curve', 
    #                       'total_thrust_bem', 'BEM thrust curve', 
    #                       'Thrust [kN]')
    
    # Save the results to CSV files
    results_df.to_csv(results_dir / 'results.csv', index=False)
    converged_results_df.to_csv(results_dir / 'converged_results_df.csv', index=False)


# This guard ensures the code only runs when executed directly (not imported)
if __name__ == "__main__":
    main()

