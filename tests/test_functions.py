# tests/test_functions.py
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.functions import *

# ----------- Core math/physics functions -----------

def test_compute_flow_angle_simple():
    df = pd.DataFrame({
        'axial_induction': [0.2],
        'tangential_induction': [0.05],
        'span_position': [10]
    })
    phi_rad, phi_deg = compute_flow_angle(df, v_inflow=10, rotational_speed=1)
    assert phi_rad.shape == (1,)
    assert phi_deg.shape == (1,)
    assert np.isfinite(phi_rad).all()

def test_compute_local_angle_of_attack_simple():
    flow_angles = np.array([0.2])
    blade_data = pd.DataFrame({'BlTwist': [5.0]})
    alpha_rad, alpha_deg = compute_local_angle_of_attack(flow_angles, 0.0, blade_data, 'BlTwist')
    assert alpha_rad.shape == (1,)
    assert np.isfinite(alpha_deg).all()

def test_interpolate_Cl_Cd_coeff_simple():
    angles_df = pd.DataFrame({'local_angle_of_attack_deg': [5.0]})
    airfoil_polar = {'00': pd.DataFrame({'Alpha': [0, 10], 'Cl': [0.5, 1.0], 'Cd': [0.01, 0.02]})}
    Cl, Cd = interpolate_Cl_Cd_coeff(angles_df, airfoil_polar)
    assert Cl.shape == (1,)
    assert Cd.shape == (1,)
    assert Cl[0] > 0
    assert Cd[0] > 0

def test_compute_normal_and_tangential_coeff():
    Cl, Cd, phi = 1.0, 0.05, np.pi/6
    Cn = compute_normal_coeff(Cl, Cd, phi)
    Ct = compute_tangential_coeff(Cl, Cd, phi)
    assert isinstance(Cn, float)
    assert isinstance(Ct, float)
    assert Cn > 0
    assert Ct > 0

def test_compute_thrust_and_power_coeff():
    rho, A, V_inflow, thrust, power = 1.225, 100.0, 10.0, 500.0, 1000.0
    Ct = compute_thrust_coeff(rho, A, V_inflow, thrust)
    Cp = compute_power_coeff(rho, A, V_inflow, power)
    assert 0 <= Ct <= 2
    assert 0 <= Cp <= 1

def test_compute_total_loads():
    total_thrust, total_torque = compute_total_loads(100, 50, 3)
    assert total_thrust == 300
    assert total_torque == 150

# ----------- Induction factor updates -----------

def test_update_axial_and_tangential():
    df = pd.DataFrame({
        'flow_angle_rad': [0.2],
        'local_solidity': [0.05],
        'Cn': [1.0],
        'Ct': [0.2]
    })
    a = update_axial(df)
    a_prime = update_tangential(df)
    assert np.isfinite(a).all()
    assert np.isfinite(a_prime).all()

def test_prandtl_correction_safe():
    df = pd.DataFrame({'span_position': [5.0], 'flow_angle_rad': [0.1]})
    F = prandtl_correction(df, 3, 50)
    assert (0 <= F[0] <= 1)

def test_update_delta_thrust_coeff_basic():
    df = pd.DataFrame({
        'local_solidity': [0.05],
        'Cn': [1.0],
        'prandtl_factor': [0.9],
        'flow_angle_rad': [0.2],
        'axial_induction': [0.3]
    })
    delta_Ct = update_delta_thrust_coeff(df)
    assert delta_Ct.shape == (1,)

def test_update_axial_joe_and_tangential_joe():
    df = pd.DataFrame({
        'delta_thrust_coeff': [0.5],
        'prandtl_factor': [0.9],
        'local_solidity': [0.05],
        'Ct': [0.2],
        'flow_angle_rad': [0.2],
        'axial_induction': [0.3]
    })
    a_joe = update_axial_joe(df)
    a_prime_joe = update_tangential_joe(df)
    assert np.isfinite(a_joe).all()
    assert np.isfinite(a_prime_joe).all()

# ----------- Differential thrust/torque and power -----------

def test_compute_dT_and_dM():
    dT = compute_dT(10, 1, 1.225, 10, 0.3)
    dM = compute_dM(10, 1, 1.225, 10, 0.3, 0.05, 1)
    assert dT > 0
    assert dM > 0

def test_compute_aerodynamic_power_simple():
    P = compute_aerodynamic_power(10, 1)
    assert P == 10

# ----------- Convergence logic -----------

def test_check_convergence_stops_fast():
    df = pd.DataFrame({
        'axial_induction': [0.2],
        'tangential_induction': [0.05],
        'axial_induction_new': [0.2001],
        'tangential_induction_new': [0.0501]
    })
    conv, df_new, counter = check_convergence(df, tolerance=0.01, iteration_counter=0, convergence_reached=False)
    assert isinstance(conv, bool)

# ----------- Flow Angle logic -----------

def test_flow_angle_loop_basic():
    # Create fake input data matching what compute_flow_angle expects
    df = pd.DataFrame({
        'axial_induction': [0.2, 0.2, 0.2, 0.2],
        'tangential_induction': [0.05, 0.05, 0.05, 0.05],
        'span_position': [0, 5, 10, 15]
    })
    V0 = 10
    omega = 2
    phi_rad, phi_deg = compute_flow_angle(df, V0, omega)
    assert len(phi_rad) == len(df)
    assert len(phi_deg) == len(df)


# ----------- File reading tests -----------

def test_read_airfoil_file(tmp_path):
    file = tmp_path / "AF01.txt"
    content = (
        "50  ! Number of points\n"
        "Header 1\nHeader 2\nHeader 3\nHeader 4\n"
        "!  x/c        y/c\n"
        "0.0 0.0\n0.1 0.1\n0.2 0.2\n0.3 0.3\n0.4 0.4\n"
    )
    file.write_text(content)
    airfoil_num, df = read_airfoil_file(file)
    assert airfoil_num == "01"
    assert not df.empty

def test_read_airfoil_polar_file(tmp_path):
    file = tmp_path / "Polar_01.dat"
    content = (
        "1000 Reynolds number\n"
        "2 NumAlf\n"
        "Alpha Cl Cd Cm\n"
        "0 1.0 0.01 0\n"
        "10 1.2 0.02 0\n"
    )
    file.write_text(content)
    airfoil_num, df = read_airfoil_polar_file(file)
    assert airfoil_num == "01"
    assert not df.empty
    assert 'Cl' in df.columns

def test_read_all_airfoil_files(tmp_path):
    af_file = tmp_path / "AF01.txt"
    af_file.write_text(
        "50  ! Number of points\n"
        "Header 1\nHeader 2\nHeader 3\nHeader 4\n"
        "!  x/c        y/c\n"
        "0.0 0.0\n"
        "0.1 0.1\n"
        "0.2 0.2\n"
        "0.3 0.3\n"
        "0.4 0.4\n"
    )
    polar_file = tmp_path / "Polar_01.dat"
    polar_file.write_text(
        "1000 Reynolds number\n"
        "2 NumAlf\n"
        "Alpha Cl Cd Cm\n"
        "0 1.0 0.01 0\n"
        "10 1.2 0.02 0\n"
    )
    coord, polar = read_all_airfoil_files(tmp_path)
    assert "01" in coord
    assert "01" in polar
    assert not coord["01"].empty
    assert not polar["01"].empty

def test_read_blade_data_file(tmp_path):
    file = tmp_path / "blade.dat"
    file.write_text(
        "2 NumBlNds\n"
        "BlSpn BlChord BlTwist\n"
        "(m) (m) (deg)\n"
        "5 0.5 2\n"
        "10 1.0 4\n"
    )
    df = read_blade_data_file(file)
    assert 'BlSpn' in df.columns
    assert 'r/R' in df.columns
    assert len(df) == 2
    assert np.isclose(df['r/R'].iloc[-1], 1.0)

# ----------- Plotting tests (mock savefig and show) -----------

def test_plot_airfoils_mock(monkeypatch):
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    airfoil_coords = {"00": pd.DataFrame({"x/c": [0, 1], "y/c": [0, 0]})}
    plot_airfoils(airfoil_coords, show_plot=False)

def test_plot_airfoils_3d_mock(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend to avoid Tkinter
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    airfoil_coords = {"00": pd.DataFrame({"x/c": [0, 1], "y/c": [0, 0]})}
    span = pd.Series([1])
    twist = pd.Series([0])
    plot_airfoils_3d(airfoil_coords, span, twist, show_plot=False)
    
