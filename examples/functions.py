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
    # if airfoil_data and "00" in airfoil_data:
    #     print(f'Head of {data_description} data \n {airfoil_data["00"].head()}')
    
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

def read_blade_data_file(file_path):
    """
    Read a blade data file (.dat) and return a pandas DataFrame.
    
    Parameters:
    ----------
    file_path : Path or str
        Path to the blade data file
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame containing blade geometry data
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find number of blade nodes
    num_nodes = None
    for i, line in enumerate(lines):
        if 'NumBlNds' in line:
            parts = line.strip().split()
            num_nodes = int(parts[0])
            header_line_idx = i + 1  # Line with column headers
            units_line_idx = i + 2   # Line with units (m), (deg), etc.
            data_start_idx = i + 3   # First data line - CHANGED HERE
            break
    
    if num_nodes is None:
        raise ValueError("Could not find number of blade nodes in file")
    
    # Extract column headers
    headers = lines[header_line_idx].strip().split()
    
    # Read data - STARTING AFTER THE UNITS LINE
    data = []
    for i in range(data_start_idx, data_start_idx + num_nodes):
        if i < len(lines):
            row = lines[i].strip().split()
            # Convert all values to float
            row_data = [float(val) for val in row]
            data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Add a normalized span column (useful for calculations)
    if 'BlSpn' in df.columns:
        blade_length = df['BlSpn'].max()
        df['r/R'] = df['BlSpn'] / blade_length
    
    return df

# %% Math / Physical functions

def compute_local_solidity(df_angles, blade_data_df, chord_length, span_position):
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
    c = blade_data_df[chord_length].values  # chord length in meters
    r = df_angles[span_position].values  # span position in meters
    # Fixed: use np.clip to avoid division by zero
    r = np.clip(r, 1e-6, None)  # Avoid division by zero
    sigma = c*B/(2*pi*r)  # local solidity

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
def compute_flow_angle(df, span_position, axial_factor, tangential_factor,
                       v_inflow, rotational_speed):
    """
    Calculate the flow angle at a given span position.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing span positions
    span_position : str
        Column name for span position in df
    axial_factor : float
        Axial induction factor
    tangential_factor : float
        Tangential induction factor
    v_inflow : float
        Inflow velocity in m/s
    rotational_speed : float
        Rotational speed in rad/s

    returns:
    -------
    numpy.ndarray
        Flow angles in radians for each span position
    """
    a = axial_factor  # axial induction factor
    a_prime = tangential_factor  # tangential induction factor
    omega = rotational_speed  # rotational speed in rad/s
    V0 = v_inflow  # inflow velocity
    radius = df[span_position].values  # span position in meters (np array)
    phi = np.zeros(len(radius))  # Initialize flow angle array

    # Calculate flow angle for each radius
    for i, r in enumerate(radius):
        if r == 0:
            phi[i] = (pi/2)  # 90 degrees at root
        else:
            # Fixed: proper indexing and parentheses for correct calculation
            phi[i] = arctan(((1-a) * V0) / ((1 + a_prime) * omega * r))
    
    return phi  # np array

def compute_local_angle_of_attack(flow_angles, PITCH_ANGLE, blade_data_df, blade_twist):
    """
    Calculate the local angle of attack for a single wind speed.

    Parameters:
    ----------
    flow_angles_df : pandas.DataFrame
        DataFrame containing flow angles in degrees with shape (num_spans, 1)
    pitch_angle : float
        Pitch angle in degrees for the specific wind speed
    blade_data_df : pandas.DataFrame
        DataFrame containing blade geometry data including twist angles in degrees

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing local angles of attack in degrees with same shape as flow_angles_df
    """
    # Get the flow angles and convert to radians
    phi = flow_angles * (pi/180)  # 50x1 array
    
    # Convert pitch angle to radians
    theta = PITCH_ANGLE * (pi/180)  # SCALAR
    
    # Get twist angle (beta) from blade data for each span position in radians
    beta = blade_data_df[blade_twist].values * (pi/180)  # 50x1 array
    #beta = beta.reshape(-1, 1)  # Shape: (num_spans, 1)
    
    # Calculate local angle of attack
    # Output in radians
    alpha = phi - (theta + beta)  # 50x1 array
    # Convert to degrees
    alpha_deg = alpha * (180/pi)  # 50x1 array
    
    # # Convert back to DataFrame with the same structure as flow_angles_df
    # alpha_df = pd.DataFrame(
    #     data=alpha_deg,
    #     index=flow_angles_df.index,
    #     columns=flow_angles_df.columns
    # )
    
    # # Add descriptive headers
    # alpha_df.columns.name = 'Span Position (m)'
    # alpha_df.index.name = 'Local Angle of Attack (deg)'
    
    return alpha, alpha_deg
    
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

def compute_CT(rho, A, V_inflow, thrust):
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

# %% Total thrust and torque
def compute_total_loads(thrust_one_blade, torque_one_blade, num_blades):
    """
    Calculate total thrust and torque for all blades.
    
    Parameters:
    ----------
    thrust_one_blade : float
        Thrust force for one blade (N)
    torque_one_blade : float
        Torque for one blade (N·m)
    num_blades : int
        Number of blades
        
    Returns:
    -------
    tuple (float, float)
        Total thrust (N), Total torque (N·m)
    """
    total_thrust = thrust_one_blade * num_blades
    total_torque = torque_one_blade * num_blades
    
    return total_thrust, total_torque
    
# %% induction factors
def update_axial(df, flow_angle, local_solidity, normal_force_coeff):
    """
    Update the axial induction factor based on flow angle, local solidity, and Cn.

    Parameters:
    ----------
    flow_angle : String
        Column name for Flow angle in radians
    local_solidity : String
        Column name for Local solidity at span position r
    Cn : String
        Column name for Normal force coefficient
    
    Returns:
    -------
    float
        Updated axial induction factor

    """
    phi = df[flow_angle].values  # flow angle in radians
    sigma = df[local_solidity].values  # local solidity
    Cn = df[normal_force_coeff].values  # normal force coefficient
    axial = 1/(4*sin(phi)**2/(sigma*Cn)+1)  # updated axial induction factor

    return axial

def update_tangential(df, flow_angle, local_solidity, tangential_force_coeff):   
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
    phi = df[flow_angle].values  # flow angle in radians
    sigma = df[local_solidity].values  # local solidity
    Ct = df[tangential_force_coeff].values  # normal force coefficient
    
    tangential = 1/(4*sin(phi)*cos(phi)/(sigma*Ct)-1)  # updated tangential induction factor

    return tangential

# %% differential functions
def compute_dT(r, dr, rho, V_inflow, axial_factor):
    """
    Compute differential thrust at a blade element.
    
    Parameters:
    ----------
    r : float
        Local radius
    dr : float
        Differential element length
    rho : float
        Air density
    V_inflow : float
        Inflow velocity
    axial_factor : float
        Local axial induction factor
        
    Returns:
    -------
    float
        Differential thrust
    """
    dT = 4 * pi * r * rho * V_inflow**2 * axial_factor * (1 - axial_factor) * dr
    return dT

def compute_dM(r, dr, rho, V_inflow, axial_factor, tangential_factor, omega):
    """
    Compute differential torque at a blade element.
    
    Parameters:
    ----------
    r : float
        Local radius
    dr : float
        Differential element length
    rho : float
        Air density
    V_inflow : float
        Inflow velocity
    axial_factor : float
        Local axial induction factor
    tangential_factor : float
        Local tangential induction factor
    omega : float
        Rotational speed in rad/s
        
    Returns:
    -------
    float
        Differential torque
    """
    dM = 4 * pi * r**3 * rho * V_inflow * omega * tangential_factor * (1 - axial_factor) * dr

    return dM

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
def flow_angle_loop(span_positions, V0, omega):
    """
    Calculate flow angles at each span position for a single wind speed.

    Parameters:
    ----------
    span_positions : array-like
        Span positions along the blade (meters)
    V0 : float
        Wind speed (m/s)
    omega : float
        Rotational speed (rad/s)

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing flow angles in degrees for each span position
    """
    # Define the axial and tangential induction factors as 0
    a = 0.0  # axial induction factor
    a_prime = 0.0  # tangential induction factor

    # Create a 1D array for the flow angles
    flow_angles = np.zeros(len(span_positions))
    
    # Calculate flow angle at each span position
    for i, r in enumerate(span_positions):
        if r > 0:  # Avoid division by zero at blade root
            flow_angles[i] = compute_flow_angle(a, a_prime, V0, omega, r)
        else:
            flow_angles[i] = pi/2  # 90 degrees at root
    
    # Convert to degrees
    flow_angles_deg = np.degrees(flow_angles)
    
    # Create a DataFrame with span positions as index and wind speed as column name
    flow_angles_df = pd.DataFrame(
        data=flow_angles_deg,
        index=span_positions,
        columns=[V0]  # Use the wind speed as column name
    )
    
    # Rename the columns with more descriptive headers
    flow_angles_df.columns.name = 'flow angles (deg)'
    flow_angles_df.index.name = 'Span Position (m)'
    
    return flow_angles_df
