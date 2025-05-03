# tests/test_functions.py
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.bombas_package.utils.functions import *
from src.bombas_package.BemOptimization import BemOptimization  # BEM implementation class

# ------------------ Core Math & Physics ------------------

class TestCorePhysics:
    def test_flow_and_local_angle_and_coefficients(self):
        df = pd.DataFrame({
            'axial_induction': [0.2],
            'tangential_induction': [0.05],
            'span_position': [10],
            'flow_angles_rad': [0.2],
            'twist_angle': [5.0]
        })
        phi_rad, phi_deg = compute_flow_angle(df, 10, 1)
        alpha_rad, alpha_deg = compute_local_angle_of_attack(df, 0.0)
        assert np.isfinite(phi_rad).all()
        assert np.isfinite(alpha_deg).all()

    def test_lift_drag_and_force_coeffs(self):
        df = pd.DataFrame({'local_angle_of_attack_deg': [5.0]})
        polar = {'00': pd.DataFrame({'Alpha': [0, 10], 'Cl': [0.5, 1.0], 'Cd': [0.01, 0.02]})}
        Cl, Cd = interpolate_Cl_Cd_coeff(df, polar)
        assert Cl[0] > 0 and Cd[0] > 0

        df2 = pd.DataFrame({'Cl': [1.0], 'Cd': [0.05], 'flow_angles_rad': [np.pi / 6]})
        assert compute_normal_coeff(df2)[0] > 0
        assert compute_tangential_coeff(df2)[0] > 0

    def test_power_thrust_coeff_and_total(self):
        Ct = compute_thrust_coeff(1.225, 100.0, 10.0, 500.0)
        Cp = compute_power_coeff(1.225, 100.0, 10.0, 1000.0)
        thrust, torque = compute_total_loads(100, 50, 3)
        assert 0 <= Ct <= 2 and 0 <= Cp <= 1
        assert thrust == 300 and torque == 150

# ------------------ Induction Factor Logic ------------------

class TestInduction:
    def test_update_all_induction_types(self):
        df = pd.DataFrame({
            'flow_angles_rad': [0.2],
            'local_solidity': [0.05],
            'Cn': [1.0],
            'Ct': [0.2]
        })
        assert np.isfinite(update_axial(df)).all()
        assert np.isfinite(update_tangential(df)).all()

    def test_prandtl_and_delta_thrust(self):
        df = pd.DataFrame({'span_position': [5.0], 'flow_angles_rad': [0.1]})
        F = prandtl_correction(df, 3, 50)
        assert 0 <= F[0] <= 1

        df2 = pd.DataFrame({
            'local_solidity': [0.05], 'Cn': [1.0], 'prandtl_factor': [0.9],
            'flow_angles_rad': [0.2], 'axial_induction': [0.3]
        })
        assert update_delta_thrust_coeff(df2).shape == (1,)

    def test_joe_induction_updates(self):
        df = pd.DataFrame({
            'delta_thrust_coeff': [0.5], 'prandtl_factor': [0.9],
            'local_solidity': [0.05], 'Ct': [0.2], 'flow_angles_rad': [0.2],
            'axial_induction': [0.3]
        })
        assert np.isfinite(update_axial_joe(df)).all()
        assert np.isfinite(update_tangential_joe(df)).all()

# ------------------ Differentials & Power ------------------

class TestDifferentials:
    def test_dT_dM_and_power(self):
        dT = compute_dT(10, 1, 1.225, 10, 0.3)
        dM = compute_dM(10, 1, 1.225, 10, 0.3, 0.05, 1)
        P = compute_aerodynamic_power(10, 1)
        assert dT > 0 and dM > 0 and P == 10

# ------------------ Convergence & Flow Logic ------------------

class TestConvergenceFlow:
    def test_convergence_behavior(self):
        df = pd.DataFrame({
            'axial_induction': [0.1, 0.2],
            'tangential_induction': [0.05, 0.06],
            'axial_induction_new': [0.15, 0.25],
            'tangential_induction_new': [0.1, 0.1]
        })
        conv, df_new, iters = check_convergence(df, 1e-5, 2, False)
        assert not conv and iters == 3

    def test_compute_spanwise_flow(self):
        df = pd.DataFrame({
            'axial_induction': [0.2] * 4,
            'tangential_induction': [0.05] * 4,
            'span_position': [0, 5, 10, 15]
        })
        phi_rad, phi_deg = compute_flow_angle(df, 10, 2)
        assert len(phi_rad) == 4 and len(phi_deg) == 4

# ------------------ File Reading ------------------

class TestFileReaders:
    def test_read_valid_airfoil_and_blade_files(self, tmp_path):
        af = tmp_path / "AF01.txt"
        af.write_text("50  ! Number of points\nHeader\nHeader\nHeader\nHeader\n!  x/c        y/c\n0.0 0.0\n0.1 0.1\n")
        num, df = read_airfoil_file(af)
        assert num == "01" and not df.empty

        polar = tmp_path / "Polar_01.dat"
        polar.write_text("1000 Reynolds number\n2 NumAlf\nAlpha Cl Cd Cm\n0 1.0 0.01 0\n10 1.2 0.02 0\n")
        num2, df2 = read_airfoil_polar_file(polar)
        assert num2 == "01" and 'Cl' in df2.columns

        blade = tmp_path / "blade.dat"
        blade.write_text("2 NumBlNds\nBlSpn BlChord BlTwist\n(m) (m) (deg)\n5 0.5 2\n10 1.0 4\n")
        df3 = read_blade_data_file(blade)
        assert 'span_position' in df3.columns and np.isclose(df3['r/R'].iloc[-1], 1.0)

    def test_handle_missing_or_invalid_files(self, tmp_path):
        polar = tmp_path / "Polar_99.dat"
        polar.write_text("1000 Reynolds number\n2 NumAlf\nAlpha Cm\n0 0.1\n10 0.2\n")
        num, df = read_airfoil_polar_file(polar)
        assert num == "99" and isinstance(df, pd.DataFrame)

        af = tmp_path / "AF02.txt"
        af.write_text("0.0 0.0\n0.1 0.1\n0.2 0.2\n")
        coord, polar = read_all_airfoil_files(tmp_path)
        assert "02" in coord and "02" not in polar

# ------------------ Plotting Tests ------------------

class TestPlotting:
    @pytest.fixture(autouse=True)
    def patch_plt(self, monkeypatch):
        monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    def test_all_plot_functions(self):
        airfoil_coords = {"00": pd.DataFrame({"x/c": [0, 1], "y/c": [0, 0]})}
        span = pd.Series([1])
        twist = pd.Series([0])
        plot_airfoils(airfoil_coords, show_plot=False)
        plot_airfoils_3d(airfoil_coords, span, twist, show_plot=False)

        df = pd.DataFrame({
            'span_position': [0, 1, 2],
            'local_angle_of_attack_deg': [5, 10, 15],
            'Cl': [0.5, 0.7, 0.8]
        })
        plot_val_vs_local_angle_of_attack(df, 'Cl', show_plot=False)
        plot_local_angle_of_attack(df, show_plot=False)

        result_df = pd.DataFrame({
            'wind_speed': [5, 10, 15],
            'Cp': [0.3, 0.4, 0.5],
            'Ct': [0.7, 0.6, 0.5]
        })

        non_converged_df = pd.DataFrame({'wind_speed': [], 'Cp': [], 'Ct': []})

        plot_results_vs_ws(
            result_df,
            non_converged_df,
            'Cp',
            'Power Coeff',
            'Ct',
            'Thrust Coeff',
            non_converged_df,
            'Non-converged',
            'Coefficient value',
            '-'
        )

        df2 = pd.DataFrame({'x': [0, 1, 2], 'y': [0.1, 0.2, 0.3]})
        plot_scatter(df2, 'x', 'y', 'test', 'x', 'y', show_plot=False)

# ------------------ BEM Optimization ------------------

class TestBemOptimization:
    def test_optimization_and_thrust(self):
        power_df = pd.DataFrame({'wind_speed': [8], 'rot_speed': [12.1], 'pitch': [0.0]})
        blade_df = pd.DataFrame({'span_position': [2, 4], 'chord_length': [1.0, 0.8], 'twist_angle': [5.0, 3.0]})
        
        polar = {'00': pd.DataFrame({
            'Alpha': [-10, 0, 10],
            'Cl': [-0.5, 0.0, 0.5],
            'Cd': [0.01, 0.01, 0.01]
        })}

        bem = BemOptimization(0, power_df, blade_df)

        bem.optimize_induction_factors(
            bem.elements_df,
            polar,
            False,
            BLADES_NO=3,
            ROTOR_RADIUS=50.0,
            iteration_counter=0,
            max_iterations=100,
            tolerance=1e-3
        )

        bem.calculate_thrust_and_power(bem.elements_df, RHO=1.225, BLADES_NO=3, A=np.pi * 50**2)

        assert bem.iteration_counter > 0
        assert bem.total_thrust is not None
        assert bem.total_torque is not None
        assert bem.aero_power is not None
        assert 0 <= bem.power_coeff <= 0.593
        assert bem.thrust_coeff >= 0
