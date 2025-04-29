"""functions for the final pro ect"""
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
    
    # print(f"Successfully read {data_description} data. Amount of airfoil data: {len(airfoil_data)}")
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

def compute_local_solidity(blade_data_df, chord_length, span_position):
    """
    Calculate local solidity based on span position (r).
    
    Parameters:
    ----------
    df_angles : DataFrame
        DataFrame containing span positions and related blade angles.
        
    blade_data_df : DataFrame
        DataFrame containing the chord lengths at each span position.
        
    chord_length : str
        Column name in blade_data_df containing the chord lengths at each span position.
        
    span_position : str
        Column name in df_angles containing the span positions (r) in meters.

    Returns:
    -------
    float
        Local solidity at span position r
    """
    B = 3 # number of blades
    c = blade_data_df[chord_length].values  # chord length in meters
    r = blade_data_df[span_position].values  # span position in meters
    # Fixed: use np.clip to avoid division by zero
    r = np.clip(r, 1e-6, None)  # Avoid division by zero
    sigma = (c * B) / (2 * pi * r)  # Local solidity formula

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

# %% compute angles
def compute_flow_angle(angles_df, v_inflow, rotational_speed):
    """
    Calculate the flow angle at a given span position.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing span positions
    span_position : str
        Column name for span position in df
    angles_df : pandas.DataFrame
        DataFrame containing axial and tangential induction factors
    v_inflow : float
        Inflow velocity in m/s
    rotational_speed : float
        Rotational speed in rad/s

    returns:
    -------
    numpy.ndarray
        Flow angles in radians for each span position
    """
    a = angles_df['axial_induction'].values  # axial induction factor array
    a_prime = angles_df['tangential_induction'].values  # tangential induction factor array
    omega = rotational_speed  # rotational speed in rad/s
    V0 = v_inflow  # inflow velocity
    radius = angles_df['span_position'].values  # span position in meters (np array)
    phi = np.zeros(len(radius))  # Initialize flow angle array

    # Calculate flow angle for each radius
    for i, r in enumerate(radius):
        if r == 0 or omega == 0 or (1 + a_prime[i]) == 0:
            phi[i] = (pi/2)  # 90 degrees at root or when denominator would be zero
        else:
            # Ensure the denominator is not too close to zero
            denominator = (1 + a_prime[i]) * omega * r
            if abs(denominator) < 1e-10:  # Small threshold to avoid numerical issues
                phi[i] = (pi/2)
            else:
                phi[i] = arctan(((1-a[i]) * V0) / denominator)
    
    phi_deg = np.degrees(phi)  # Convert to degrees

    return phi, phi_deg  # np array of flow angles in radians

def compute_local_angle_of_attack(flow_angles, PITCH_ANGLE, blade_data_df, blade_twist):
    """
    Calculate the local angle of attack at each blade element.

    Parameters:
    ----------
    flow_angles : ndarray
        Array of flow angles in radians
    
    PITCH_ANGLE : float
        Blade pitch angle in radians
    
    blade_data_df : DataFrame
        DataFrame containing blade geometry data including twist angles
    
    blade_twist : str
        Column name in blade_data_df for the twist angle in degrees
    
    Returns:
    -------
    tuple of ndarrays
        (alpha_rad, alpha_deg) containing local angles of attack in 
        radians and degrees respectively
    """
    # Get the flow angles 
    phi = flow_angles   # 50x1 array
    
    # Convert pitch angle to radians
    theta = PITCH_ANGLE  # pitch angle in radians
    
    # Get twist angle (beta) from blade data for each span position in radians
    beta = blade_data_df[blade_twist].values * (pi/180)  # 50x1 array
    
    # Calculate local angle of attack
    # Output in radians
    alpha_rad = phi - (theta + beta)  # 50x1 array
    # Convert to degrees
    alpha_deg = alpha_rad * (180/pi)  # 50x1 array
        
    return alpha_rad, alpha_deg    
# %% Coefficients functions
def interpolate_Cl_Cd_coeff(angles_df, airfoil_polar):
    """
    Compute lift coefficients by interpolating airfoil polar data for each blade element.
    
    Parameters:
    ----------
    angles_df : pandas.DataFrame
        DataFrame containing local angles of attack and span positions
    airfoil_polar : dict
        Dictionary of airfoil polar data, keys are airfoil identifiers
        
    Returns:
    -------
    numpy.ndarray
        Array of lift coefficients for each blade element
    """
    # Get the local angles of attack
    alpha_deg = angles_df['local_angle_of_attack_deg'].values
    
    # Available airfoil IDs (sorted numerically)
    airfoil_ids = sorted(list(airfoil_polar.keys()), key=lambda x: int(x))
    
    # Initialize array for lift coefficients
    Cl = np.zeros_like(alpha_deg)
    Cd = Cl.copy()  # Initialize drag coefficients as well
    
    # For each blade element, match with corresponding airfoil by index
    for i in range(len(alpha_deg)):
        # Simple direct mapping - use index to select airfoil
        # Convert index to two-digit string (00, 01, 02, ..., 49)
        airfoil_idx = str(i).zfill(2)
        
        # If this exact airfoil exists, use it, otherwise find closest
        if airfoil_idx in airfoil_ids:
            airfoil_id = airfoil_idx
        else:
            # This is a fallback if there's not exactly 50 airfoils
            airfoil_id = airfoil_ids[min(i, len(airfoil_ids)-1)]
        
        # Get airfoil data
        airfoil_data = airfoil_polar[airfoil_id]
        
        # Interpolate Cl using local angle of attack
        Cl[i] = np.interp(alpha_deg[i], airfoil_data['Alpha'], airfoil_data['Cl'])
        Cd[i] = np.interp(alpha_deg[i], airfoil_data['Alpha'], airfoil_data['Cd'])
        

    
    return Cl, Cd

def compute_normal_coeff(Cl, Cd, flow_angle):
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

def compute_tangential_coeff(Cl, Cd, flow_angle):
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

def compute_thrust_coeff(rho, A, V_inflow, thrust):
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

def compute_power_coeff(rho, A, V_inflow, power):
    
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
def update_axial(df):
    """
    Update the axial induction factor based on flow angle, local solidity, and Cn.

    Parameters:
    ----------
    df : DataFrame
        The input dataframe containing the necessary columns for flow angle, 
        local solidity, and normal force coefficient.
    
    flow_angle : str
        Column name for flow angle in radians.
    
    local_solidity : str
        Column name for local solidity at span position r.
    
    normal_force_coeff : str
        Column name for normal force coefficient (Cn).
    
    Returns:
    -------
    ndarray
        Array of updated axial induction factors for each span position.

    """
    phi = df['flow_angle_rad'].values  # flow angle in radians
    sigma = df['local_solidity'].values  # local solidity
    Cn = df['Cn'].values  # normal force coefficient
    axial = 1 / (4 * (sin(phi) ** 2) / (sigma * Cn) + 1) # updated axial induction factor

    return axial

def update_tangential(df):   
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
    phi = df['flow_angle_rad'].values  # flow angle in radians
    sigma = df['local_solidity'].values  # local solidity
    Ct = df['Ct'].values  # normal force coefficient
    
    tangential = 1 / (4 * (sin(phi) * cos(phi)) / (sigma * Ct) - 1)

    return tangential

def update_axial_joe(elements_df):
    """Update axial induction factor with correction"""
    dC_T = elements_df['delta_thrust_coeff'].values
    F = elements_df['prandtl_factor'].values
    
    # Clip dC_T to reasonable values to prevent overflow
    dC_T = np.clip(dC_T, 0.0, 1.815)  # Clip to just below 1.816 to avoid sqrt of negative
    
    # Different formulas for different CT ranges
    a_updated = np.zeros_like(dC_T)
    
    # Glauert correction for highly loaded elements
    high_load_idx = dC_T > 0.96
    normal_idx = ~high_load_idx
    
    # Normal BEM for most elements
    a_updated[normal_idx] = 0.246 * dC_T[normal_idx] + 0.0586 * dC_T[normal_idx]**2 + 0.0883 * dC_T[normal_idx]**3
    
    # Glauert correction for high load - ensure we don't take sqrt of negative
    sqrt_term = np.maximum(0, 1.816 - dC_T[high_load_idx])
    a_updated[high_load_idx] = 1 - 0.5 * np.sqrt(sqrt_term)
    
    # Apply Prandtl factor
    a_updated = a_updated * F
    
    # Clip to physical range
    a_updated = np.clip(a_updated, 0.0, 0.95)
    
    # Handle any NaN values
    a_updated = np.nan_to_num(a_updated, nan=0.0)
    
    return a_updated

def update_tangential_joe(elements_df):
    """Update tangential induction factor with correction"""
    sigma = elements_df['local_solidity'].values
    C_t = elements_df['Ct'].values
    phi = elements_df['flow_angle_rad'].values
    a = elements_df['axial_induction'].values
    F = elements_df['prandtl_factor'].values
    
    # Add safety for small sin values
    sin_phi_safe = np.clip(np.abs(np.sin(phi)), 1e-6, None)
    cos_phi_safe = np.clip(np.abs(np.cos(phi)), 1e-6, None)
    
    # Safe calculation
    a_prime_updated = np.zeros_like(a)
    denominator = 4 * F * sin_phi_safe * cos_phi_safe
    
    # Only update where denominator is significant
    valid_idx = denominator > 1e-6
    a_prime_updated[valid_idx] = ((sigma[valid_idx] * C_t[valid_idx]) * (1 + a[valid_idx])) / denominator[valid_idx]
    
    # Clip to physical range
    a_prime_updated = np.clip(a_prime_updated, -0.5, 0.5)
    
    # Handle any NaN values
    a_prime_updated = np.nan_to_num(a_prime_updated, nan=0.0)
    
    return a_prime_updated

def prandtl_correction(angles_df, B, R):
    """Calculate Prandtl's tip loss factor"""
    r = angles_df['span_position'].values
    phi = angles_df['flow_angle_rad'].values
    
    # Add safety values to prevent division by zero
    r_safe = np.clip(r, 1e-6, None)  # Minimum value of 1e-6
    sin_phi_safe = np.clip(np.abs(np.sin(phi)), 1e-6, None)
    
    intermediate_term = (B / 2) * (R - r_safe) / (r_safe * sin_phi_safe)
    F = (2 / pi) * np.arccos(np.clip(np.exp(-intermediate_term), 0, 1))
    
    # Handle any remaining NaN values
    F = np.nan_to_num(F, nan=0.1)
    return F

def update_delta_thrust_coeff(df):
    sigma = df['local_solidity'].values
    C_n = df['Cn'].values
    F = df['prandtl_factor'].values
    phi = df['flow_angle_rad'].values
    a_1 = df['axial_induction'].values

    # Initialize the output array
    delta_thrust_coeff = np.zeros_like(a_1)
    
    # Vectorized calculation without conditional
    # Add small epsilon to avoid division by zero
    delta_thrust_coeff = ((1 - a_1)**2 * sigma * C_n) / (F * np.sin(phi)**2)
    
    # Handle any NaN or infinite values that might result from division by zero
    delta_thrust_coeff = np.nan_to_num(delta_thrust_coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return delta_thrust_coeff

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
    dT = pi * r * rho * V_inflow**2 * axial_factor * (1 - axial_factor) * dr # [N]

    return dT

def compute_dM(r, dr, rho, V_inflow, axial_factor, tangential_factor, rotational_speed):
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
    
    dM = pi * r**3 * rho * V_inflow * rotational_speed * tangential_factor * (1 - axial_factor) * dr # [N*m]

    return dM

# %% Power
def compute_aerodynamic_power(torque, rotational_speed):
    """
    Calculate the aerodynamic power based on torque and rotational speed.

    Parameters:
    ----------
    torque : float
        Torque in kNm
    rotational_speed : float
        Rotational speed in rad/s

    Returns:
    -------
    float
        Aerodynamic power in Watts

    """
    
    P_aero = torque * rotational_speed  #kNm/s = kW

    return P_aero # [kW]

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

def plot_airfoils_3d(airfoil_coords, blade_span, blade_twist, show_plot=False):
    """
    Plot airfoil coordinates in 3D, incorporating blade span and twist angle.
    
    Parameters:
    ----------
    airfoil_coords : dict
        Dictionary of airfoil coordinates
    blade_span : array-like
        Blade span positions from root to tip
    blade_twist : array-like
        Twist angles at corresponding blade span positions
    show_plot : bool
        Whether to display the plot interactively
    
    Returns:
    -------
    None
        Saves the plot to the specified directory
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, airfoil_num in enumerate(airfoil_coords):
        # Get airfoil coordinates
        x = airfoil_coords[airfoil_num]['x/c']
        y = airfoil_coords[airfoil_num]['y/c']
        z = np.full_like(x, blade_span.values[i])  # Set z-axis as blade span position

        # Apply twist angle (rotation around z-axis)
        twist_angle_rad = np.radians(blade_twist.values[i])
        x_rot = x * np.cos(twist_angle_rad) - y * np.sin(twist_angle_rad)
        y_rot = x * np.sin(twist_angle_rad) + y * np.cos(twist_angle_rad)

        # Plot the airfoil in 3D
        ax.plot(x_rot, y_rot, z, label=f'Airfoil {airfoil_num}')

    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_zlabel('Blade Span (m)')
    ax.set_title('3D Airfoil Geometry with Blade Span and Twist')
    #ax.legend()
    ax.grid(True)

    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, '3D_Airfoil_Geometry.png')
    plt.savefig(save_path)
    print(f'Saved 3D airfoil plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_flow_angles(blade_data_df, flow_angles_deg, show_plot=False):

    plt.figure(figsize=(10, 6))
    # Exclude root and tip elements (positions 0 and -1) for clearer visualization
    plt.plot(blade_data_df['BlSpn'].iloc[1:-1], flow_angles_deg[1:-1], 'bo-', linewidth=2)
    plt.xlabel('Blade Span Position (m)', fontsize=12)
    plt.ylabel('Flow Angle (degrees)', fontsize=12)
    plt.title('Flow Angle vs Blade Span Position', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, 'Flow_Angle_vs_Blade_Span_Position.png')
    plt.savefig(save_path)
    print(f'Saved Flow Angle vs Blade Span Position plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()
    


def plot_local_angle_of_attack(angles_df, blade_data_df, show_plot=False):
    plt.figure(figsize=(10, 6))
    # Exclude root and tip elements (positions 0 and -1) for clearer visualization
    plt.plot(blade_data_df['BlSpn'].iloc[1:-1], angles_df['local_angle_of_attack_deg'].iloc[1:-1], 'bo-', linewidth=2)
    plt.xlabel('Blade Span (m)', fontsize=12)
    plt.ylabel('Local Angle of Attack (degrees)', fontsize=12)
    plt.title('Local Angle of Attack vs Blade Span', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, 'Local_Angle_of_Attack_Blade_Twist_Position.png')
    plt.savefig(save_path)
    print(f'Saved Local Angle of Attack vs Blade Twist plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_val_vs_local_angle_of_attack(angles_df, parameter, show_plot=False):
    plt.figure(figsize=(10, 6))
    # Exclude root and tip elements (positions 0 and -1) for clearer visualization
    plt.plot(angles_df['local_angle_of_attack_deg'].iloc[1:-1], angles_df[parameter].iloc[1:-1], 'bo-', linewidth=2, label = parameter)
    plt.xlabel('Local Angle of Attack (degrees)', fontsize=12)
    plt.ylabel(parameter, fontsize=12)
    plt.title(f'{parameter} vs Local Angle of Attack', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, f'{parameter}_vs_Local_Angle_of_Attack.png')
    plt.savefig(save_path)
    print(f'Saved {parameter} vs Local Angle of Attack plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_scatter(df,x,y, parameter, xlabel, ylabel, show_plot=False):
    plt.figure(figsize=(10, 6))
    # Exclude root and tip elements (positions 0 and -1) for clearer visualization
    plt.plot(df[x].iloc[1:-1], df[y].iloc[1:-1], 'bo-', linewidth=2, label = parameter)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{parameter}_vs{xlabel}', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, f'{parameter}_vs_{xlabel}.png')
    plt.savefig(save_path)
    print(f'Saved {parameter} vs {xlabel} plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()

def plot_results_vs_ws(df_results, y1, label1,
                  y2, label2,
                    ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['wind_speed'], df_results[y1], label=label1, color='blue')
    plt.plot(df_results['wind_speed'], df_results[y2], label=label2, color='orange')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel(ylabel)
    plt.title(f'{label1} vs {label2}')
    plt.legend()
    plt.grid()
    #save the figure
    # Define path to save figures
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(pictures_dir, f'{label1}_and_{label2}_vs_wind_speed.png')
    plt.savefig(save_path)
    print(f'Saved {label1} vs {label2} plot to {save_path}')
    plt.close()       

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

# %% Convergence Check

def check_convergence(elements_df, tolerance, iteration_counter, convergence_reached):
    """
    Check if the induction factors have converged.
    
    Parameters:
    -----------
    angles_df : DataFrame
        DataFrame containing induction factors
    tolerance : float
        Convergence tolerance
    iteration_counter : int
        Current iteration count
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    tuple : (converged, updated_df, stop_iterations)
        converged: Boolean indicating if convergence was reached
        updated_df: DataFrame with updated induction factors if not converged
        stop_iterations: Boolean indicating if iterations should stop
    """
    # Check if convergence has been reached
    if (np.abs(elements_df['axial_induction_new'] - elements_df['axial_induction']) < tolerance).all() and \
       (np.abs(elements_df['tangential_induction_new'] - elements_df['tangential_induction']) < tolerance).all():
        print(f'Convergence reached at iteration: {iteration_counter}')
        convergence_reached = True
    else:
        # Apply relaxation factor to update induction factors
        relax = 0.25  # Relaxation factor (adjust as needed)
        elements_df['axial_induction'] = (1-relax) * elements_df['axial_induction'] + relax * elements_df['axial_induction_new']
        elements_df['tangential_induction'] = (1-relax) * elements_df['tangential_induction'] + relax * elements_df['tangential_induction_new']
        
        iteration_counter += 1
    
    return convergence_reached, elements_df, iteration_counter