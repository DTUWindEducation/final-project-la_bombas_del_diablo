# pylint: disable=C0103

"""functions for the final project"""


import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import (arctan, cos, pi, sin)

# %% Read airfoil data
def read_airfoil_file(file_path):
    """
    Read an airfoil coordinates file and return a DataFrame and simplified airfoil ID.

    Parameters
    ----------
    file_path : Path or str
        Path to the airfoil file

    Returns
    -------
    tuple
        (airfoil_number, pandas.DataFrame)
        airfoil_number is the simplified ID (e.g. "00", "01")
        DataFrame has x/c and y/c columns
    """
    file_path_str = str(file_path)
    airfoil_num = ""
    if "AF" in file_path_str:
        match = re.search(r'AF(\d+)', file_path_str)
        if match:
            airfoil_num = match.group(1)

    with open(file_path, 'r', encoding='utf-8') as f:
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

    df = pd.DataFrame({'x/c': x_coords, 'y/c': y_coords})
    return airfoil_num, df


def read_airfoil_polar_file(file_path):
    """
    Read an airfoil polar data file (.dat) and return a DataFrame.

    Parameters
    ----------
    file_path : Path or str
        Path to the airfoil polar file

    Returns
    -------
    tuple
        (airfoil_number, pandas.DataFrame) with Alpha, Cl, Cd, Cm columns
        and metadata added as additional columns
    """
    file_path_str = str(file_path)
    airfoil_num = ""
    if "Polar" in file_path_str:
        match = re.search(r'Polar_(\d+)', file_path_str)
        if match:
            airfoil_num = match.group(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    re_number = None
    data_start = 0
    num_points = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line and len(line.split()) >= 2 and line.split()[1].startswith("Re") and "Reynolds number" in line:
            try:
                re_number = float(line.split()[0])
            except ValueError:
                pass
        elif 'NumAlf' in line:
            try:
                num_points = int(line.split()[0])
                data_start = i + 2
                break
            except (ValueError, IndexError):
                pass

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

    df = pd.DataFrame({
        'Alpha': alpha,
        'Cl': cl,
        'Cd': cd,
        'Cm': cm,
        'Re': re_number if re_number else None
    })

    return airfoil_num, df


def read_airfoil_files(airfoils_dir, file_type='coordinates'):
    """
    Read airfoil files and return data dictionary.

    Parameters
    ----------
    airfoils_dir : Path
        Directory containing airfoil files
    file_type : str
        Either 'coordinates' for .txt files or 'polar' for .dat files

    Returns
    -------
    dict
        Dictionary of airfoil data with airfoil numbers as keys
    """
    airfoil_data = {}

    if file_type == 'coordinates':
        file_pattern = '*.txt'
        reader_function = read_airfoil_file
    elif file_type == 'polar':
        file_pattern = '*.dat'
        reader_function = read_airfoil_polar_file
    else:
        raise ValueError("file_type must be 'coordinates' or 'polar'")

    for file_path in airfoils_dir.glob(file_pattern):
        try:
            airfoil_num, df = reader_function(file_path)
            airfoil_data[airfoil_num] = df
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    return airfoil_data


def read_all_airfoil_files(airfoils_dir):
    """
    Read both coordinate and polar airfoil files.

    Parameters
    ----------
    airfoils_dir : Path
        Directory containing airfoil files

    Returns
    -------
    tuple
        (coordinate_data, polar_data) tuple of dictionaries
    """
    coordinate_data = read_airfoil_files(airfoils_dir, file_type='coordinates')
    polar_data = read_airfoil_files(airfoils_dir, file_type='polar')
    return coordinate_data, polar_data


# %% Read power curve data
def read_power_curve_file(file_path):
    """
    Read the power curve data from a file and return a DataFrame.

    Parameters
    ----------
    file_path : Path or str
        Path to the power curve data file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the power curve data with columns:
        - wind_speed [m/s]
        - pitch [deg]
        - rot_speed [rpm]
        - aero_power [kw]
        - aero_thrust [kn]
        - rot_speed_rad [rad/s]
    """
    power_curve_df = pd.read_csv(
        file_path,
        skiprows=1,
        sep=r'\s+',
        names=['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
    )

    power_curve_df['rot_speed_rad'] = power_curve_df['rot_speed'] * (2 * pi / 60)
    return power_curve_df


def read_blade_data_file(file_path):
    """
    Read a blade data file (.dat) and return a pandas DataFrame.

    Parameters
    ----------
    file_path : Path or str
        Path to the blade data file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing blade geometry data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    num_nodes = None
    for i, line in enumerate(lines):
        if 'NumBlNds' in line:
            parts = line.strip().split()
            num_nodes = int(parts[0])
            header_line_idx = i + 1
            data_start_idx = i + 3
            break

    if num_nodes is None:
        raise ValueError("Could not find number of blade nodes in file")

    headers = lines[header_line_idx].strip().split()
    data = []
    for i in range(data_start_idx, data_start_idx + num_nodes):
        if i < len(lines):
            row = lines[i].strip().split()
            row_data = [float(val) for val in row]
            data.append(row_data)

    df = pd.DataFrame(data, columns=headers)
    if 'BlSpn' in df.columns:
        blade_length = df['BlSpn'].max()
        df['r/R'] = df['BlSpn'] / blade_length

    df = df.rename(columns={
        'BlSpn': 'span_position',
        'BlTwist': 'twist_angle',
        'BlChord': 'chord_length'
    })

    return df
# %% Math / Physical functions

def compute_local_solidity(elements_df):
    """
    Calculate local solidity based on span position (r).

    Parameters
    ----------
    elements_df : DataFrame
        DataFrame with chord lengths and span_positions at each span position

    Returns
    -------
    float
        Local solidity at span position r
    """
    B = 3
    c = elements_df['chord_length'].values
    r = np.clip(elements_df['span_position'].values, 1e-6, None)
    sigma = (c * B) / (2 * pi * r)
    return sigma


def tip_speed_ratio(rotational_speed, rotor_radius, v_inflow):
    """
    Calculate the tip speed ratio (TSR).

    Parameters
    ----------
    rotational_speed : float
        Angular velocity in rad/s
    rotor_radius : float
        Rotor radius in meters
    v_inflow : float
        Freestream velocity in m/s

    Returns
    -------
    float
        Tip speed ratio (TSR)
    """
    return (rotational_speed * rotor_radius) / v_inflow


def compute_flow_angle(elements_df, v_inflow, rotational_speed):
    """
    Calculate the flow angle at a given span position.

    Parameters
    ----------
    elements_df : pandas.DataFrame
        DataFrame containing axial and tangential induction factors
    v_inflow : float
        Inflow velocity in m/s
    rotational_speed : float
        Rotational speed in rad/s

    Returns
    -------
    tuple
        Flow angles in radians and degrees for each span position
    """
    a = elements_df['axial_induction'].values
    a_prime = elements_df['tangential_induction'].values
    omega = rotational_speed
    V0 = v_inflow
    radius = elements_df['span_position'].values
    phi = np.zeros(len(radius))

    for i, r in enumerate(radius):
        if r == 0 or omega == 0 or (1 + a_prime[i]) == 0:
            phi[i] = pi / 2
        else:
            denominator = (1 + a_prime[i]) * omega * r
            if abs(denominator) < 1e-10:
                phi[i] = pi / 2
            else:
                phi[i] = arctan(((1 - a[i]) * V0) / denominator)

    phi_deg = np.degrees(phi)
    return phi, phi_deg


def compute_local_angle_of_attack(elements_df, pitch_angle):
    """
    Calculate the local angle of attack at each blade element.

    Parameters
    ----------
    elements_df : DataFrame
        DataFrame containing blade geometry data including twist angles
    pitch_angle : float
        Blade pitch angle in radians

    Returns
    -------
    tuple
        Local angle of attack in radians and degrees
    """
    phi = elements_df['flow_angles']
    theta = pitch_angle
    beta = elements_df['twist_angle'].values * (pi / 180)
    alpha_rad = phi - (theta + beta)
    alpha_deg = alpha_rad * (180 / pi)
    return alpha_rad, alpha_deg


# %% Coefficients functions
def interpolate_Cl_Cd_coeff(elements_df, airfoil_polar):
    """
    Compute lift coefficients by interpolating airfoil polar data for each blade element.

    Parameters
    ----------
    elements_df : pandas.DataFrame
        DataFrame containing local angles of attack and span positions
    airfoil_polar : dict
        Dictionary of airfoil polar data, keys are airfoil identifiers

    Returns
    -------
    numpy.ndarray
        Array of lift coefficients for each blade element
    """
    alpha_deg = elements_df['local_angle_of_attack_deg'].values
    airfoil_ids = sorted(list(airfoil_polar.keys()), key=lambda x: int(x))

    Cl = np.zeros_like(alpha_deg)
    Cd = Cl.copy()

    for i in range(len(alpha_deg)):
        airfoil_idx = str(i).zfill(2)

        if airfoil_idx in airfoil_ids:
            airfoil_id = airfoil_idx
        else:
            airfoil_id = airfoil_ids[min(i, len(airfoil_ids) - 1)]

        airfoil_data = airfoil_polar[airfoil_id]
        Cl[i] = np.interp(alpha_deg[i], airfoil_data['Alpha'], airfoil_data['Cl'])
        Cd[i] = np.interp(alpha_deg[i], airfoil_data['Alpha'], airfoil_data['Cd'])

    return Cl, Cd


def compute_normal_coeff(elements_df):
    """
    Compute the normal force coefficient (Cn) based on Cl, flow angle, and Cd.

    Parameters
    ----------
    elements_df : DataFrame

    Returns
    -------
    float
        Normal force coefficient (Cn)
    """
    Cl = elements_df['Cl'].values
    Cd = elements_df['Cd'].values
    flow_angle = elements_df['flow_angle_rad'].values
    Cn = Cl * cos(flow_angle) + Cd * sin(flow_angle)

    return Cn


def compute_tangential_coeff(elements_df):
    """
    Compute the tangential force coefficient (Ct) based on Cl, flow angle, and Cd.

    Parameters
    ----------
    elements_df : DataFrame

    Returns
    -------
    float
        Tangential force coefficient (Ct)
    """
    Cl = elements_df['Cl'].values
    Cd = elements_df['Cd'].values
    flow_angle = elements_df['flow_angle_rad'].values

    Ct = Cl * sin(flow_angle) - Cd * cos(flow_angle)

    return Ct


def compute_thrust_coeff(rho, A, V_inflow, thrust):
    """
    Compute the thrust coefficient (Ct) based on thrust, air density, rotor area, and inflow velocity.

    Parameters
    ----------
    rho : float
        Air density in kg/m^3
    A : float
        Rotor area in m^2
    V_inflow : float
        Inflow velocity in m/s
    thrust : float
        Thrust in Newtons

    Returns
    -------
    float
        Thrust coefficient (Ct)
    """
    Ct = thrust / (0.5 * rho * A * V_inflow**2)

    return Ct


def compute_power_coeff(rho, A, V_inflow, power):
    """
    Compute the thrust coefficient (Cp) based on power, air density, rotor area, and inflow velocity.

    Parameters
    ----------
    rho : float
        Air density in kg/m^3
    A : float
        Rotor area in m^2
    V_inflow : float
        Inflow velocity in m/s
    power : float
        Power in Watts

    Returns
    -------
    float
        Power coefficient (Cp)
    """
    Cp = power / (0.5 * rho * A * V_inflow**3)

    return Cp


# %% Total thrust and torque
def compute_total_loads(thrust_one_blade, torque_one_blade, num_blades):
    """
    Calculate total thrust and torque for all blades.

    Parameters
    ----------
    thrust_one_blade : float
        Thrust force for one blade (N)
    torque_one_blade : float
        Torque for one blade (N·m)
    num_blades : int
        Number of blades

    Returns
    -------
    tuple (float, float)
        Total thrust (N), Total torque (N·m)
    """
    total_thrust = thrust_one_blade * num_blades
    total_torque = torque_one_blade * num_blades

    return total_thrust, total_torque


# %% Induction factors
def update_axial(df):
    """
    Update the axial induction factor based on flow angle, local solidity, and Cn.

    Parameters
    ----------
    df : DataFrame
        The input dataframe containing the necessary columns for flow angle,
        local solidity, and normal force coefficient.

    Returns
    -------
    ndarray
        Array of updated axial induction factors for each span position.
    """
    phi = df['flow_angle_rad'].values
    sigma = df['local_solidity'].values
    Cn = df['Cn'].values

    axial = 1 / (4 * (sin(phi) ** 2) / (sigma * Cn) + 1)

    return axial


def update_tangential(df):
    """
    Update the tangential induction factor based on flow angle, local solidity, and Ct.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    float
        Updated tangential induction factor
    """
    phi = df['flow_angle_rad'].values
    sigma = df['local_solidity'].values
    Ct = df['Ct'].values

    tangential = 1 / (4 * (sin(phi) * cos(phi)) / (sigma * Ct) - 1)

    return tangential


def update_axial_joe(elements_df):
    """Update axial induction factor with correction.
    
    Parameters
    ----------
    elements_df : DataFrame
        DataFrame containing blade element data including delta thrust coefficient
    
        
    Returns
    -------
    numpy.ndarray
        Array of updated axial induction factors for each blade element
    
    """
    dC_T = elements_df['delta_thrust_coeff'].values
    F = elements_df['prandtl_factor'].values

    # Clip dC_T to reasonable values to prevent overflow
    dC_T = np.clip(dC_T, 0.0, 1.815)

    a_updated = np.zeros_like(dC_T)

    # Glauert correction for highly loaded elements
    high_load_idx = dC_T > 0.96
    normal_idx = ~high_load_idx

    # Normal BEM for most elements
    a_updated[normal_idx] = (
        0.246 * dC_T[normal_idx]
        + 0.0586 * dC_T[normal_idx] ** 2
        + 0.0883 * dC_T[normal_idx] ** 3
    )

    # Glauert correction for high load
    sqrt_term = np.maximum(0, 1.816 - dC_T[high_load_idx])
    a_updated[high_load_idx] = 1 - 0.5 * np.sqrt(sqrt_term)

    # Apply Prandtl factor and clip to physical range
    a_updated *= F
    a_updated = np.clip(a_updated, 0.0, 0.95)
    a_updated = np.nan_to_num(a_updated, nan=0.0)

    return a_updated


def update_tangential_joe(elements_df):
    """Update tangential induction factor with correction

    Parameters
    ----------
    elements_df : DataFrame
        DataFrame containing blade element data including 
        local solidity, Ct, and flow angle
 
    Returns
    -------
    numpy.ndarray
        Array of updated tangential induction factors for each blade element

    """
    sigma = elements_df['local_solidity'].values
    C_t = elements_df['Ct'].values
    phi = elements_df['flow_angle_rad'].values
    a = elements_df['axial_induction'].values
    F = elements_df['prandtl_factor'].values

    sin_phi_safe = np.clip(np.abs(np.sin(phi)), 1e-6, None)
    cos_phi_safe = np.clip(np.abs(np.cos(phi)), 1e-6, None)

    a_prime_updated = np.zeros_like(a)
    denominator = 4 * F * sin_phi_safe * cos_phi_safe

    valid_idx = denominator > 1e-6
    a_prime_updated[valid_idx] = (
        (sigma[valid_idx] * C_t[valid_idx]) * (1 + a[valid_idx])
    ) / denominator[valid_idx]

    a_prime_updated = np.clip(a_prime_updated, -0.5, 0.5)
    a_prime_updated = np.nan_to_num(a_prime_updated, nan=0.0)

    return a_prime_updated


def prandtl_correction(elements_df, B, R):
    """Calculate Prandtl's tip loss factor."""
    r = elements_df['span_position'].values
    phi = elements_df['flow_angle_rad'].values

    r_safe = np.clip(r, 1e-6, None)
    sin_phi_safe = np.clip(np.abs(np.sin(phi)), 1e-6, None)

    intermediate_term = (B / 2) * (R - r_safe) / (r_safe * sin_phi_safe)
    F = (2 / pi) * np.arccos(np.clip(np.exp(-intermediate_term), 0, 1))

    F = np.nan_to_num(F, nan=0.1)
    
    return F


def update_delta_thrust_coeff(df):
    """Compute delta thrust coefficient based on BEM relations."""
    sigma = df['local_solidity'].values
    C_n = df['Cn'].values
    F = df['prandtl_factor'].values
    phi = df['flow_angle_rad'].values
    a_1 = df['axial_induction'].values

    delta_thrust_coeff = ((1 - a_1) ** 2 * sigma * C_n) / (F * np.sin(phi) ** 2)
    delta_thrust_coeff = np.nan_to_num(delta_thrust_coeff, nan=0.0,
                                       posinf=0.0, neginf=0.0)

    return delta_thrust_coeff


# %% Differential functions
def compute_dT(r, dr, rho, V_inflow, axial_factor):
    """
    Compute differential thrust at a blade element.

    Parameters
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

    Returns
    -------
    float
        Differential thrust
    """
    dT = pi * r * rho * V_inflow ** 2 * axial_factor * (1 - axial_factor) * dr
    return dT


def compute_dM(r, dr, rho, V_inflow, axial_factor,
               tangential_factor, rotational_speed):
    """
    Compute differential torque at a blade element.

    Parameters
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
    rotational_speed : float
        Rotational speed in rad/s

    Returns
    -------
    float
        Differential torque
    """
    dM = (pi * r ** 3 * rho * V_inflow * rotational_speed *
          tangential_factor * (1 - axial_factor) * dr)
    return dM
# %% Power
def compute_aerodynamic_power(torque, rotational_speed):
    """
    Calculate the aerodynamic power based on torque and rotational speed.

    Parameters
    ----------
    torque : float
        Torque in kNm
    rotational_speed : float
        Rotational speed in rad/s

    Returns
    -------
    float
        Aerodynamic power in Watts
    """
    P_aero = torque * rotational_speed  # kNm/s = kW
    return P_aero  # [kW]


# %% Plot functions
def plot_airfoils(airfoil_coords, show_plot=False):
    """
    Plot airfoil coordinates.

    Parameters
    ----------
    airfoil_coords : dict
        Dictionary of airfoil coordinates
    show_plot : bool
        Whether to display the plot interactively

    Returns
    -------
    None
        Saves the plot to the specified directory
    """
    for airfoil_num in airfoil_coords:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            airfoil_coords[airfoil_num]['x/c'],
            airfoil_coords[airfoil_num]['y/c'],
            s=10,
            label=f'Airfoil {airfoil_num}'
        )

        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(f'Airfoil Geometry {airfoil_num}')
        plt.legend()
        plt.grid(True)

        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pictures_dir = os.path.join(
            main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
        os.makedirs(pictures_dir, exist_ok=True)

        save_path = os.path.join(
            pictures_dir, f'Airfoil_Geometry_{airfoil_num}.png')
        plt.savefig(save_path)
        print(f'Saved {airfoil_num}/{len(airfoil_coords)}')

        if show_plot:
            plt.show()
        plt.close()

    print(f"Saved {len(airfoil_coords)} airfoil plots to {pictures_dir}")


def plot_airfoils_3d(airfoil_coords, blade_span, blade_twist, show_plot=False):
    """
    Plot airfoil coordinates in 3D, incorporating blade span and twist angle.

    Parameters
    ----------
    airfoil_coords : dict
        Dictionary of airfoil coordinates
    blade_span : array-like
        Blade span positions from root to tip
    blade_twist : array-like
        Twist angles at corresponding blade span positions
    show_plot : bool
        Whether to display the plot interactively

    Returns
    -------
    None
        Saves the plot to the specified directory
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, airfoil_num in enumerate(airfoil_coords):
        x = airfoil_coords[airfoil_num]['x/c']
        y = airfoil_coords[airfoil_num]['y/c']
        z = np.full_like(x, blade_span.values[i])

        twist_angle_rad = np.radians(blade_twist.values[i])
        x_rot = x * np.cos(twist_angle_rad) - y * np.sin(twist_angle_rad)
        y_rot = x * np.sin(twist_angle_rad) + y * np.cos(twist_angle_rad)

        ax.plot(x_rot, y_rot, z, label=f'Airfoil {airfoil_num}')

    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_zlabel('Blade Span (m)')
    ax.set_title('3D Airfoil Geometry with Blade Span and Twist')
    ax.grid(True)

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(pictures_dir, '3D_Airfoil_Geometry.png')
    plt.savefig(save_path)
    print(f'Saved 3D airfoil plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_flow_angles(elements_df, flow_angles_deg, show_plot=False):
    """
    Plot flow angle vs blade span position.

    Parameters
    ----------
    elements_df : DataFrame
        DataFrame with span positions
    flow_angles_deg : array-like
        Flow angles in degrees
    show_plot : bool
        Whether to display the plot interactively
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        elements_df['span_position'].iloc[1:-1],
        flow_angles_deg[1:-1],
        'bo-',
        linewidth=2
    )
    plt.xlabel('Blade Span Position (m)', fontsize=12)
    plt.ylabel('Flow Angle (degrees)', fontsize=12)
    plt.title('Flow Angle vs Blade Span Position', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(
        pictures_dir, 'Flow_Angle_vs_Blade_Span_Position.png')
    plt.savefig(save_path)
    print(f'Saved Flow Angle vs Blade Span Position plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()

def plot_local_angle_of_attack(elements_df, show_plot=False):
    """
    Plot the local angle of attack along the blade span and save the figure.

    This function creates a 2D line plot of the local angle of attack (in degrees)
    versus blade span position. The plot is saved to the 'outputs/pictures' directory
    of the project. Optionally, the plot can be shown interactively.

    Parameters
    ----------
    elements_df : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'span_position': positions along the blade (m)
        - 'local_angle_of_attack_deg': local angle of attack (degrees)

    show_plot : bool, optional
        If True, displays the plot interactively. Default is False.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        elements_df['span_position'].iloc[1:-1],
        elements_df['local_angle_of_attack_deg'].iloc[1:-1],
        'bo-',
        linewidth=2
    )
    plt.xlabel('Blade Span (m)', fontsize=12)
    plt.ylabel('Local Angle of Attack (degrees)', fontsize=12)
    plt.title('Local Angle of Attack vs Blade Span', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(
        pictures_dir, 'Local_Angle_of_Attack_Blade_Twist_Position.png')
    plt.savefig(save_path)
    print(f'Saved Local Angle of Attack vs Blade Twist plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_val_vs_local_angle_of_attack(elements_df, parameter, show_plot=False):
    """
    Plot a specified parameter against the local angle of attack and save the figure.

    This function creates a 2D line plot of a given blade element parameter (e.g., Cl, Cd, Cn)
    versus the local angle of attack (in degrees). The plot is saved to the 
    'outputs/pictures' directory of the project. Optionally, the plot can be shown interactively.

    Parameters
    ----------
    elements_df : pandas.DataFrame
        DataFrame containing at least:
        - 'local_angle_of_attack_deg': local angle of attack in degrees
        - The specified `parameter` column to plot against angle of attack.

    parameter : str
        The name of the column in `elements_df` to be plotted on the y-axis.

    show_plot : bool, optional
        If True, displays the plot interactively. Default is False.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        elements_df['local_angle_of_attack_deg'].iloc[1:-1],
        elements_df[parameter].iloc[1:-1],
        'bo-',
        linewidth=2,
        label=parameter
    )
    plt.xlabel('Local Angle of Attack (degrees)', fontsize=12)
    plt.ylabel(parameter, fontsize=12)
    plt.title(f'{parameter} vs Local Angle of Attack', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(
        pictures_dir, f'{parameter}_vs_Local_Angle_of_Attack.png')
    plt.savefig(save_path)
    print(f'Saved {parameter} vs Local Angle of Attack plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_scatter(df, x, y, parameter, xlabel, ylabel, show_plot=False):
    """
    Plot a scatter-style line plot of a parameter versus another variable and save the figure.

    This function generates a 2D line plot of the specified `y` values versus `x` values 
    from the given DataFrame. The plot is labeled according to the `parameter`, `xlabel`, 
    and `ylabel` arguments, and saved to the 'outputs/pictures' directory. Optionally, 
    the plot can be shown interactively.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted.

    x : str
        Column name in `df` to use for the x-axis.

    y : str
        Column name in `df` to use for the y-axis.

    parameter : str
        Name of the parameter being plotted, used for legend and filename.

    xlabel : str
        Label for the x-axis.

    ylabel : str
        Label for the y-axis.

    show_plot : bool, optional
        If True, displays the plot interactively. Default is False.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        df[x].iloc[1:-1],
        df[y].iloc[1:-1],
        'bo-',
        linewidth=2,
        label=parameter
    )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{parameter}_vs{xlabel}', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(pictures_dir, f'{parameter}_vs_{xlabel}.png')
    plt.savefig(save_path)
    print(f'Saved {parameter} vs {xlabel} plot to {save_path}')

    if show_plot:
        plt.show()
    plt.close()


def plot_results_vs_ws(df_results, y1, label1, y2, label2, ylabel):
    """
    Plot two result series against wind speed and save the figure.

    This function plots two specified columns from a results DataFrame as functions
    of wind speed, with separate labels for each line. The resulting plot is saved
    to the 'outputs/pictures' directory and shows how two parameters vary with wind speed.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing a 'wind_speed' column and at least the `y1` and `y2` columns.

    y1 : str
        Name of the first column to plot on the y-axis.

    label1 : str
        Legend label for the first data series.

    y2 : str
        Name of the second column to plot on the y-axis.

    label2 : str
        Legend label for the second data series.

    ylabel : str
        Label for the y-axis.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['wind_speed'], df_results[y1],
             label=label1, color='blue')
    plt.plot(df_results['wind_speed'], df_results[y2],
             label=label2, color='orange')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel(ylabel)
    plt.title(f'{label1} vs {label2}')
    plt.legend()
    plt.grid()

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pictures_dir = os.path.join(
        main_dir, 'final-project-la_bombas_del_diablo', 'outputs', 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    save_path = os.path.join(
        pictures_dir, f'{label1}_and_{label2}_vs_wind_speed.png')
    plt.savefig(save_path)
    print(f'Saved {label1} vs {label2} plot to {save_path}')
    plt.close()


# %% Step 1
def flow_angle_loop(span_positions, V0, omega):
    """
    Calculate flow angles at each span position for a single wind speed.

    Parameters
    ----------
    span_positions : array-like
        Span positions along the blade (meters)
    V0 : float
        Wind speed (m/s)
    omega : float
        Rotational speed (rad/s)

    Returns
    -------
    pandas.DataFrame
        DataFrame containing flow angles in degrees for each span position
    """
    a = 0.0  # axial induction factor
    a_prime = 0.0  # tangential induction factor

    flow_angles = np.zeros(len(span_positions))

    for i, r in enumerate(span_positions):
        if r > 0:
            flow_angles[i] = compute_flow_angle(a, a_prime, V0, omega, r)
        else:
            flow_angles[i] = pi / 2

    flow_angles_deg = np.degrees(flow_angles)

    flow_elements_df = pd.DataFrame(
        data=flow_angles_deg,
        index=span_positions,
        columns=[V0]
    )
    flow_elements_df.columns.name = 'flow angles (deg)'
    flow_elements_df.index.name = 'Span Position (m)'

    return flow_elements_df


# %% Convergence Check
def check_convergence(elements_df, tolerance, iteration_counter, convergence_reached):
    """
    Check if the induction factors have converged.

    Parameters
    ----------
    elements_df : DataFrame
        DataFrame containing induction factors
    tolerance : float
        Convergence tolerance
    iteration_counter : int
        Current iteration count
    convergence_reached : bool
        Current convergence status

    Returns
    -------
    tuple
        (converged, updated_df, updated_iteration_counter)
    """
    diff_axial = np.abs(
        elements_df['axial_induction_new'] - elements_df['axial_induction'])
    diff_tangential = np.abs(
        elements_df['tangential_induction_new'] - elements_df['tangential_induction'])

    if (diff_axial < tolerance).all() and (diff_tangential < tolerance).all():
        print(f'Convergence reached at iteration: {iteration_counter}')
        convergence_reached = True
    else:
        relax = 0.25
        elements_df['axial_induction'] = (
            1 - relax) * elements_df['axial_induction'] + relax * elements_df['axial_induction_new']
        elements_df['tangential_induction'] = (
            1 - relax) * elements_df['tangential_induction'] + relax * elements_df['tangential_induction_new']
        iteration_counter += 1

    return convergence_reached, elements_df, iteration_counter
