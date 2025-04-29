import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the examples directory to the Python path so we can import the functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
import functions as fn

@pytest.fixture
def sample_airfoil_data():
    coords = {
        '00': pd.DataFrame({
            'x/c': [0, 0.5, 1],
            'y/c': [0, 0.1, 0]
        })
    }
    polar = {
        '00': pd.DataFrame({
            'Alpha': [-5, 0, 5],
            'Cl': [-0.5, 0, 0.5],
            'Cd': [0.01, 0.01, 0.02],
            'Cm': [0, 0, 0],
            'Re': [1e6, 1e6, 1e6]
        })
    }
    return coords, polar

@pytest.fixture
def sample_blade_data():
    return pd.DataFrame({
        'BlSpn': [0, 5, 10],
        'BlTwist': [13, 10, 7],
        'BlChord': [4, 3, 2]
    })

@pytest.fixture
def sample_angles_df():
    return pd.DataFrame({
        'axial_induction': [0.2, 0.25, 0.3],
        'tangential_induction': [0.01, 0.02, 0.03],
        'span_position': [0, 5, 10],
        'flow_angle_rad': [0.5, 0.4, 0.3],
        'local_angle_of_attack_rad': [0.1, 0.2, 0.3],
        'local_angle_of_attack_deg': [5.7, 11.4, 17.2],
        'Cl': [0.5, 0.6, 0.7],
        'Cd': [0.01, 0.015, 0.02],
        'Cn': [0.4, 0.5, 0.6],
        'Ct': [0.1, 0.15, 0.2],
        'local_solidity': [0.3, 0.2, 0.1],
        'prandtl_factor': [0.9, 0.95, 0.98]
    })

def test_compute_local_solidity(sample_blade_data):
    result = fn.compute_local_solidity(sample_blade_data, 'BlChord', 'BlSpn')
    assert len(result) == len(sample_blade_data)
    assert np.all(result >= 0)  # Solidity should be non-negative
    assert np.isfinite(result).all()  # No infinities or NaNs

def test_tip_speed_ratio():
    rotational_speed = 1.5  # rad/s
    rotor_radius = 50.0     # m
    v_inflow = 10.0        # m/s
    tsr = fn.tip_speed_ratio(rotational_speed, rotor_radius, v_inflow)
    assert isinstance(tsr, float)
    assert tsr == pytest.approx(7.5)  # Expected TSR = (1.5 * 50) / 10

def test_compute_flow_angle(sample_angles_df):
    v_inflow = 10.0
    rotational_speed = 1.5
    phi, phi_deg = fn.compute_flow_angle(sample_angles_df, v_inflow, rotational_speed)
    assert len(phi) == len(sample_angles_df)
    assert len(phi_deg) == len(sample_angles_df)
    assert np.all(phi >= 0)  # Flow angles should be non-negative
    assert np.all(phi_deg >= 0)
    assert np.isfinite(phi).all()
    assert np.isfinite(phi_deg).all()

def test_compute_local_angle_of_attack(sample_blade_data):
    flow_angles = np.array([0.5, 0.4, 0.3])
    pitch_angle = 0.1
    alpha_rad, alpha_deg = fn.compute_local_angle_of_attack(flow_angles, pitch_angle, 
                                                          sample_blade_data, 'BlTwist')
    assert len(alpha_rad) == len(flow_angles)
    assert len(alpha_deg) == len(flow_angles)
    assert np.isfinite(alpha_rad).all()
    assert np.isfinite(alpha_deg).all()
    assert np.allclose(alpha_deg, alpha_rad * 180 / np.pi, rtol=1e-10)

def test_interpolate_Cl_Cd_coeff(sample_angles_df, sample_airfoil_data):
    _, airfoil_polar = sample_airfoil_data
    Cl, Cd = fn.interpolate_Cl_Cd_coeff(sample_angles_df, airfoil_polar)
    assert len(Cl) == len(sample_angles_df)
    assert len(Cd) == len(sample_angles_df)
    assert np.isfinite(Cl).all()
    assert np.isfinite(Cd).all()
    assert np.all(Cd > 0)  # Drag coefficient should be positive

def test_compute_normal_coeff():
    Cl = 0.5
    Cd = 0.01
    flow_angle = 0.2
    Cn = fn.compute_normal_coeff(Cl, Cd, flow_angle)
    assert isinstance(Cn, float)
    assert np.isfinite(Cn)
    # Test with arrays
    Cl_arr = np.array([0.5, 0.6])
    Cd_arr = np.array([0.01, 0.02])
    flow_angle_arr = np.array([0.2, 0.3])
    Cn_arr = fn.compute_normal_coeff(Cl_arr, Cd_arr, flow_angle_arr)
    assert len(Cn_arr) == len(Cl_arr)
    assert np.isfinite(Cn_arr).all()

def test_compute_tangential_coeff():
    Cl = 0.5
    Cd = 0.01
    flow_angle = 0.2
    Ct = fn.compute_tangential_coeff(Cl, Cd, flow_angle)
    assert isinstance(Ct, float)
    assert np.isfinite(Ct)
    # Test with arrays
    Cl_arr = np.array([0.5, 0.6])
    Cd_arr = np.array([0.01, 0.02])
    flow_angle_arr = np.array([0.2, 0.3])
    Ct_arr = fn.compute_tangential_coeff(Cl_arr, Cd_arr, flow_angle_arr)
    assert len(Ct_arr) == len(Cl_arr)
    assert np.isfinite(Ct_arr).all()

def test_compute_thrust_coeff():
    rho = 1.225
    A = 100.0
    V_inflow = 10.0
    thrust = 1000.0
    Ct = fn.compute_thrust_coeff(rho, A, V_inflow, thrust)
    assert isinstance(Ct, float)
    assert Ct > 0
    assert np.isfinite(Ct)

def test_compute_power_coeff():
    rho = 1.225
    A = 100.0
    V_inflow = 10.0
    power = 1000.0
    Cp = fn.compute_power_coeff(rho, A, V_inflow, power)
    assert isinstance(Cp, float)
    assert Cp > 0
    assert np.isfinite(Cp)

def test_compute_total_loads():
    thrust_one_blade = 1000.0
    torque_one_blade = 2000.0
    num_blades = 3
    total_thrust, total_torque = fn.compute_total_loads(thrust_one_blade, torque_one_blade, num_blades)
    assert total_thrust == pytest.approx(3000.0)
    assert total_torque == pytest.approx(6000.0)

def test_update_axial(sample_angles_df):
    axial = fn.update_axial(sample_angles_df)
    assert len(axial) == len(sample_angles_df)
    assert np.all(axial >= 0)  # Axial induction factor should be non-negative
    assert np.isfinite(axial).all()

def test_update_tangential(sample_angles_df):
    tangential = fn.update_tangential(sample_angles_df)
    assert len(tangential) == len(sample_angles_df)
    assert np.isfinite(tangential).all()

def test_prandtl_correction(sample_angles_df):
    B = 3  # number of blades
    R = 50.0  # rotor radius
    F = fn.prandtl_correction(sample_angles_df, B, R)
    assert len(F) == len(sample_angles_df)
    assert np.all(F >= 0)  # Prandtl factor should be non-negative
    assert np.all(F <= 1)  # Prandtl factor should be <= 1
    assert np.isfinite(F).all()

def test_update_delta_thrust_coeff(sample_angles_df):
    delta_ct = fn.update_delta_thrust_coeff(sample_angles_df)
    assert len(delta_ct) == len(sample_angles_df)
    assert np.isfinite(delta_ct).all()

def test_compute_dT():
    r = 10.0
    dr = 1.0
    rho = 1.225
    V_inflow = 10.0
    axial_factor = 0.3
    dT = fn.compute_dT(r, dr, rho, V_inflow, axial_factor)
    assert isinstance(dT, float)
    assert dT > 0
    assert np.isfinite(dT)

def test_compute_dM():
    r = 10.0
    dr = 1.0
    rho = 1.225
    V_inflow = 10.0
    axial_factor = 0.3
    tangential_factor = 0.1
    rotational_speed = 1.5
    dM = fn.compute_dM(r, dr, rho, V_inflow, axial_factor, tangential_factor, rotational_speed)
    assert isinstance(dM, float)
    assert np.isfinite(dM)

def test_compute_aerodynamic_power():
    torque = 1000.0
    rotational_speed = 1.5
    power = fn.compute_aerodynamic_power(torque, rotational_speed)
    assert isinstance(power, float)
    assert power > 0
    assert np.isfinite(power)
    assert power == pytest.approx(1500.0)  # Expected power = 1000 * 1.5