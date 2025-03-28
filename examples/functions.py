"""functions for the final project"""
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
def flow_angle(axial_factor, tangential_factor,
               V_inflow, rotational_speed, r):
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
    phi = arctan((1-axial_factor) / (1 + tangential_factor)
                      *V_inflow/(rotational_speed*r))  # flow angle in radians
    
    return phi

def local_angle_of_attack(flow_angle, blade_pitch_angle, local_twist_angle, r):
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
def aerodynamic_power(torque, rotational_speed):
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


