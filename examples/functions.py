"""functions for the final project"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, sqrt, tan, arctan, arcsin, arccos
import pandas as pd
import scipy as sp
import re

# %% Read airfoil data
def read_airfoil_file(file_path):
    """
    Read an airfoil coordinates file and return a DataFrame and simplified airfoil ID.
    
    Parameters:
    ----------
    file_path : Path or str
        Path to the airfoil file
        
    Returns:
    -------
    tuple
        (airfoil_number, pandas.DataFrame)
        airfoil_number is the simplified ID (e.g. "00", "01")
        DataFrame has x/c and y/c columns
    """
    # Extract airfoil number from filename
    file_path_str = str(file_path)
    airfoil_num = ""
    if "AF" in file_path_str:
        match = re.search(r'AF(\d+)', file_path_str)
        if match:
            airfoil_num = match.group(1)
    
    # Original file reading code
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    num_coords = int(re.search(r'(\d+)', lines[0]).group(1))
    
    data_start = 0
    for i, line in enumerate(lines):
        if '!  x/c        y/c' in line and i > 4:
            data_start = i + 1
            break
    
    x_coords = []
    y_coords = []
    
    for i in range(data_start, data_start + num_coords - 1):
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                x_coords.append(float(parts[0]))
                y_coords.append(float(parts[1]))
    
    df = pd.DataFrame({
        'x/c': x_coords,
        'y/c': y_coords
    })
    
    # Return both the airfoil number and the data
    return airfoil_num, df

def read_airfoil_polar_file(file_path):
    """
    Read an airfoil polar data file (.dat) and return a DataFrame.
    
    Parameters:
    ----------
    file_path : Path or str
        Path to the airfoil polar file
        
    Returns:
    -------
    tuple
        (airfoil_number, pandas.DataFrame) with Alpha, Cl, Cd, Cm columns
        and metadata added as additional columns
    """
    # Extract airfoil number from filename
    file_path_str = str(file_path)
    airfoil_num = ""
    if "Polar" in file_path_str:
        match = re.search(r'Polar_(\d+)', file_path_str)
        if match:
            airfoil_num = match.group(1)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables
    re_number = None
    data_start = 0
    num_points = 0
    
    # Extract metadata
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for Reynolds number line
        if line and len(line.split()) >= 2 and line.split()[1].startswith("Re") and "Reynolds number" in line:
            try:
                re_number = float(line.split()[0])
            except ValueError:
                pass
        
        # Look for number of data points
        elif 'NumAlf' in line:
            try:
                num_points = int(line.split()[0])
                data_start = i + 2  # Skip the column headers line
                break
            except (ValueError, IndexError):
                pass
    
    # Fallback for NumAlf
    if num_points == 0:
        for i, line in enumerate(lines):
            if 'NumAlf' in line:
                try:
                    num_points = int(line.split()[0])
                    data_start = i + 2
                    break
                except (ValueError, IndexError):
                    continue
    
    if num_points == 0:
        raise ValueError("Could not determine number of data points in file")
    
    # Read polar data
    alpha = []
    cl = []
    cd = []
    cm = []
    
    for i in range(data_start, data_start + num_points):
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                try:
                    alpha.append(float(parts[0]))
                    cl.append(float(parts[1]))
                    cd.append(float(parts[2]))
                    cm.append(float(parts[3]))
                except ValueError:
                    continue
    
    # Create DataFrame with metadata as columns
    df = pd.DataFrame({
        'Alpha': alpha,
        'Cl': cl,
        'Cd': cd,
        'Cm': cm,
        'Re': re_number if re_number else None  # Add Reynolds number as a column
    })
    
    # Return simplified structure
    return airfoil_num, df
def read_airfoil_files(airfoils_dir, file_type='coordinates'):
    """
    Read airfoil files and return data dictionary.
    
    Parameters:
    ----------
    airfoils_dir : Path
        Directory containing airfoil files
    file_type : str
        Either 'coordinates' for .txt files or 'polar' for .dat files
        
    Returns:
    -------
    dict
        Dictionary of airfoil data with airfoil numbers as keys
    """
    # Initialize dictionary to store data
    airfoil_data = {}
    
    # Set file pattern and reader function based on file type
    if file_type == 'coordinates':
        file_pattern = '*.txt'
        reader_function = read_airfoil_file  # Fixed: no more fn.
        data_description = 'airfoil coordinates'
    elif file_type == 'polar':
        file_pattern = '*.dat'
        reader_function = read_airfoil_polar_file
        data_description = 'airfoil polar'
    else:
        raise ValueError("file_type must be 'coordinates' or 'polar'")
    
    # Process files
    for file_path in airfoils_dir.glob(file_pattern):
        try:
            # Get both the airfoil number and data
            airfoil_num, df = reader_function(file_path)
            
            # Store DataFrame in dictionary using simplified key
            airfoil_data[airfoil_num] = df
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    print(f"Successfully read {data_description} data. Amount of airfoil data: {len(airfoil_data)}")
    if airfoil_data and "00" in airfoil_data:
        print(f'Head of {data_description} data \n {airfoil_data["00"].head()}')
    
    return airfoil_data

def read_all_airfoil_files(airfoils_dir):
    """
    Read both coordinate and polar airfoil files.
    
    Parameters:
    ----------
    airfoils_dir : Path
        Directory containing airfoil files
        
    Returns:
    -------
    tuple
        (coordinate_data, polar_data) tuple of dictionaries
    """
    coordinate_data = read_airfoil_files(airfoils_dir, file_type='coordinates')
    polar_data = read_airfoil_files(airfoils_dir, file_type='polar')
    return coordinate_data, polar_data


# %% read power curve data
def read_power_curve_file(file_path):

    """_summary_
    Read the power curve data from a file and return a DataFrame.

    Parameters:
    ----------
    file_path : Path or str
        Path to the power curve data file

    ----------
        
    Returns:
    -------
    power_curve_df : pandas.DataFrame
        DataFrame containing the power curve data with columns:
        - wind_speed [m/s]
        - pitch [deg]
        - rot_speed [rpm]
        - aero_power [kw]
        - aero_thrust [kn]
        - rot_speed_rad [rad/s]
    """
    # Read the data with: wind speed [m/s],          pitch [deg],     rot. speed [rpm]  ,    aero power [kw] ,    aero thrust [kn] into a df
    
    # Read the file, skipping the header line
    power_curve_df = pd.read_csv(file_path, 
                    skiprows=1,             # Skip header line
                    sep=r'\s+',  # Use whitespace as delimiter
                    names= ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust'])     # Apply column names                             

    power_curve_df['rot_speed_rad'] = power_curve_df['rot_speed']*(2*pi/60)  # Convert to rad/s

    return power_curve_df
# %% Math / Physical functions

def local_solidity(r):
    """
    Calculate local solidity based on span position (r).
    
    Parameters:
    ----------
    r : float
        Span position (r) in meters
        
    Returns:
    -------
    float
        Local solidity at span position r
    """
    B = 3 #number of blades

    sigma = span_position(r)*B/(2*pi*r)  # local solidity

    return sigma

def tip_speed_ratio(rotational_speed, ROTOR_RADIUS, V_inflow):
    """
    Calculate the tip speed ratio (TSR).
    
    Parameters:
    ----------
    omega : float
        Angular velocity in rad/s
    R : float
        Rotor radius in meters
    V_inf : float
        Freestream velocity in m/s
        
    Returns:
    -------
    float
        Tip speed ratio (TSR)
    """
    TSR = (rotational_speed * ROTOR_RADIUS) / V_inflow  # tip speed ratio

    return TSR

# %% angles
def compute_flow_angle(axial_factor, tangential_factor,
                       v_inflow, rotational_speed, r):
    """
    Calculate the flow angle at a given span position.

    Parameters:
    ----------
    axial_factor : float
        Axial induction factor
    induction_factor : float
        Induction factor
    V_inflow : float
        Inflow velocity in m/s
    rotational_speed : float
            Rotational speed in rad/s
    r : float
        Span position in meters
    ----------

    returns:
    -------
    float
        Flow angle in radians

    """

    # a = df[axial_factor] # axial induction factor
    # a_prime = df[tangential_factor]  # tangential induction factor
    # omega = rotational_speed # rotational speed in rad/s
    # V0 = df[V_inflow] # inflow velocity

    a = axial_factor # axial induction factor
    a_prime = tangential_factor  # tangential induction factor
    omega = rotational_speed # rotational speed in rad/s
    V0 = v_inflow # inflow velocity

    phi = arctan((1-a) / (1 + a_prime)*V0/(omega*r))  # flow angle in radians
    
    return phi

def compute_local_angle_of_attack(flow_angle, blade_pitch_angle, local_twist_angle, r):
    """
    Calculate the local angle of attack at a given span position.

    Parameters:
    ----------
    flow_angle : float
        Flow angle in radians
    blade_pitch_angle : float
        Blade pitch angle in radians
    local_twist_angle : float
        Local twist angle in radians

    Returns:
    -------
    float
        Local angle of attack in radians

    """
    beta = local_twist_angle(r)
    alpha = flow_angle - (blade_pitch_angle + beta)  # local angle of attack
   
    return alpha

# %% Coefficients functions
def compute_Cn(Cl, Cd, flow_angle):
    """
    Compute the normal force coefficient (Cn) based on Cl, flow angle, and Cd.

    Parameters:
    ----------
    Cl : float
        Lift coefficient
    flow_angle : float
        Flow angle in radians
    Cd : float
        Drag coefficient

    Returns:
    -------
    float
        Normal force coefficient (Cn)

    """
    Cn = Cl * cos(flow_angle) + Cd * sin(flow_angle)  # normal force coefficient

    return Cn

def compute_Ct(Cl, Cd, flow_angle):
    """
    Compute the tangential force coefficient (Ct) based on Cl, flow angle, and Cd.

    Parameters:
    ----------
    Cl : float
        Lift coefficient
    flow_angle : float
        Flow angle in radians
    Cd : float
        Drag coefficient

    Returns:
    -------
    float
        Tangential force coefficient (Ct)

    """
    Ct = Cl * sin(flow_angle) - Cd * cos(flow_angle)  # tangential force coefficient

    return Ct

def compute_Ct(rho, A, V_inflow, thrust):
    """
    Compute the thrust coefficient (Ct) based on thrust, air density, rotor area, and inflow velocity.

    Parameters:
    ----------
    rho : float
        Air density in kg/m^3
    A : float
        Rotor area in m^2
    V_inflow : float
        Inflow velocity in m/s
    thrust : float
        Thrust in Newtons

    Returns:
    -------
    float
        Thrust coefficient (Ct)

    """
    Ct = thrust / (0.5 * rho * A * V_inflow**2)  # thrust coefficient

    return Ct

def compute_Cp(rho, A, V_inflow, power):
    
    """
    Compute the thrust coefficient (Ct) based on thrust, air density, rotor area, and inflow velocity.

    Parameters:
    ----------
    rho : float
        Air density in kg/m^3
    A : float
        Rotor area in m^2
    V_inflow : float
        Inflow velocity in m/s
    thrust : float
        Thrust in Newtons

    Returns:
    -------
    float
        Thrust coefficient (Ct)

    """
    Cp = power / (0.5 * rho * A * V_inflow**3)  # thrust coefficient

    return Cp
# %% induction factors
def update_axial(flow_angle, local_solidity, Cn, r):
    """
    Update the axial induction factor based on flow angle, local solidity, and Cn.

    Parameters:
    ----------
    flow_angle : float
        Flow angle in radians
    local_solidity : float
        Local solidity at span position r
    Cn : float
        Normal force coefficient
    r : float
        Span position in meters

    Returns:
    -------
    float
        Updated axial induction factor

    """
    axial = 1/(4*sin(flow_angle)**2/(local_solidity(r)*Cn)+1)  # updated axial induction factor

    return axial

def update_tangential(flow_angle, local_solidity, Ct, r):
    """
    Update the tangential induction factor based on flow angle, local solidity, and Ct.

    Parameters:
    ----------
    flow_angle : float
        Flow angle in radians
    local_solidity : float
        Local solidity at span position r
    Ct : float
        Tangential force coefficient
    r : float
        Span position in meters

    Returns:
    -------
    float
        Updated Tangential induction factor

    """
    tangential = 1/(4*cos(flow_angle*sin(flow_angle))/(local_solidity(r)*Ct)-1)  # updated tangential induction factor

    return tangential

# %% differential functions
def compute_dT(r, rho, V_inflow, axial_factor): #REVISE THIS FUNCTION

    dr = 0.01  # differential span position
    for radius in range(0, ROTOR_RADIUS, dr):
        axial_factor = update_axial(flow_angle, local_solidity, Cn, radius)
        tangential_factor = update_tangential(flow_angle, local_solidity, Ct, radius)
        dT = 4*pi*r*rho*V_inflow**2*axial_factor*(1-axial_factor)*radius  # differential thrust

def compute_dm(r, rho, V_inflow, axial_factor, tangential_factor, rotational_speed): #REVISE THIS FUNCTION

    dr = 0.01  # differential span position
    for radius in range(0, ROTOR_RADIUS, dr):
        axial_factor = update_axial(flow_angle, local_solidity, Cn, radius)
        tangential_factor = update_tangential(flow_angle, local_solidity, Ct, radius)
        dT = 4*pi*r**3*rho*V_inflow*rotational_speed*tangential_factor*(1-axial_factor)*radius  # differential thrust

# %% Power
def compute_aerodynamic_power(torque, rotational_speed):
    """
    Calculate the aerodynamic power based on torque and rotational speed.

    Parameters:
    ----------
    torque : float
        Torque in Nm
    rotational_speed : float
        Rotational speed in rad/s

    Returns:
    -------
    float
        Aerodynamic power in Watts

    """
    P_aero = torque * rotational_speed  # aerodynamic power

    return P_aero

# %% Plot functions

def plot_airfoils(airfoil_coords, show_plot=False):
    """"
    "Plot airfoil coordinates."
    Parameters:
    ----------
    airfoil_coords : dict
        Dictionary of airfoil coordinates

    show_plot : bool
        Whether to display the plot interactively
    
    Returns:
    -------
    None
        Saves the plot to the specified directory
    
    """

    for airfoil_num in airfoil_coords:
        plt.figure(figsize=(10, 6))
        
        # Plot using the simplified airfoil numbers
        plt.scatter(airfoil_coords[airfoil_num]['x/c'], 
                airfoil_coords[airfoil_num]['y/c'], 
                s=10, 
                label=f'Airfoil {airfoil_num}')
        
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(f'Airfoil Geometry {airfoil_num}')
        plt.legend()
        plt.grid(True)
        
        # Define path to save figures
        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
        os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists
        
        save_path = os.path.join(pictures_dir, f'Airfoil_Geometry_{airfoil_num}.png')
        plt.savefig(save_path)
        print(f'Saved {airfoil_num}/{len(airfoil_coords)}')
        
        if show_plot:
            plt.show()
        plt.close()
    print(f"Saved {len(airfoil_coords)} airfoil plots to {pictures_dir}")


# %% Step 1
def flow_angle_loop(span_positions, power_curve_df):

    # Define the axial and tangential induction factors as 0
    a = 0.0  # axial induction factor
    a_prime = 0.0  # tangential induction factor

    # Create a 2D array to store flow angles (span_positions × wind_speeds)
    flow_angles = np.zeros((len(span_positions), len(power_curve_df)))

    # Loop through each span position and operational point
    for i, r in enumerate(span_positions):
        for j, (_, op_point) in enumerate(power_curve_df.iterrows()):
            # Get wind speed and rotational speed for this operational point
            V0 = op_point['wind_speed'] # m/s
            omega = op_point['rot_speed_rad'] # rad/s
            
            # Calculate flow angle for this combination
            # phi = arctan((V0 * (1-a)) / (omega * r * (1+a_prime)))
            if r > 0:  # Avoid division by zero at blade root
                phi = compute_flow_angle(a, a_prime, V0, omega, r)
                
            else:
                phi = pi/2  # 90 degrees at root
            
            flow_angles[i, j] = phi

    # Convert flow angles from radians to degrees
    flow_angles = np.degrees(flow_angles)

    # Convert the 2D array to DataFrame with named columns
    flow_angles_df = pd.DataFrame(
        flow_angles,
        index=span_positions,   # Radial positions as row indices
        columns=power_curve_df['wind_speed'].values  # Wind speeds as column names
    )

    return flow_angles_df
