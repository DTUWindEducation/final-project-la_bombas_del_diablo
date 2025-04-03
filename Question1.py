# Script for Question 1
# %% Load libraries
import numpy as np
import math
import matplotlib.pylab as plt
import pandas as pd
import sys

# %% Load in Blade Data
# Blade data
blade_data = np.loadtxt("bladedat.txt")
r = blade_data[:, 0]
beta_deg = blade_data[:, 1]
beta_rad = np.deg2rad(beta_deg)
c = blade_data[:, 2]
thick = blade_data[:, 3]


# %% Parameters with constant values
R = 89.17  # Position of the blade we are interested in [m]
B = 3  # Number of blades
rho = 1.225  # Air density [kg/m^3]
# V_o = 5  # Free stream velocity [m/s]
# tip_speed_ratio = 11.205432676824074
# theta_p = math.radians(1.8)  # Global pitch angle [radians]
tolerance = 1e-8  # Convergence tolerance
f = 0.1  # Relaxation parameter

# # Tip speed and pitch angle data
# # Changing parameters we define
# n = 10
# tip_speed_ratios = np.linspace(5,11,n)
# pitch_angles = np.linspace(5,10,n)
# Free_Stream_Velocities = np.linspace(4,24,n)  # Free stream velocity [m/s]

# for testing
# tip_speed_ratio = tip_speed_ratios[0]
# theta_p = pitch_angles[0]
# V_o = Free_Stream_Velocities[0]
# r = 24.5


# Load data for different airfoils (thickness profiles)
files = ['FFA-W3-241.txt', 'FFA-W3-301.txt', 'FFA-W3-360.txt',
         'FFA-W3-480.txt', 'FFA-W3-600.txt', 'cylinder.txt']

# Initializing tables
cl_tab = np.zeros([105, 6])
cd_tab = np.zeros([105, 6])
cm_tab = np.zeros([105, 6])
aoa_tab = np.zeros([105, ])
# Readin of tables. Only do this once at startup of simulation
for i in range(np.size(files)):
    aoa_tab[:], cl_tab[:, i], cd_tab[:, i], cm_tab[:,
                                                   i] = np.loadtxt(files[i], skiprows=0).T

# Thickness of the airfoils considered
# NOTE THAT IN PYTHON THE INTERPOLATION REQUIRES THAT THE VALUES INCREASE IN THE VECTOR!

thick_prof = np.zeros(6)
thick_prof[0] = 24.1
thick_prof[1] = 30.1
thick_prof[2] = 36
thick_prof[3] = 48
thick_prof[4] = 60
thick_prof[5] = 100

# %% Interpolation Function


def force_coeffs_10MW(angle_of_attack, thick, aoa_tab, cl_tab, cd_tab, cm_tab):
    cl_aoa = np.zeros([1, 6])
    cd_aoa = np.zeros([1, 6])
    cm_aoa = np.zeros([1, 6])

    # Interpolate to the current angle of attack
    for i in range(len(files)):
        cl_aoa[0, i] = np.interp(angle_of_attack, aoa_tab, cl_tab[:, i])
        cd_aoa[0, i] = np.interp(angle_of_attack, aoa_tab, cd_tab[:, i])
        cm_aoa[0, i] = np.interp(angle_of_attack, aoa_tab, cm_tab[:, i])

    # Interpolate to the current thickness
    cl = np.interp(thick, thick_prof, cl_aoa[0, :])
    cd = np.interp(thick, thick_prof, cd_aoa[0, :])
    cm = np.interp(thick, thick_prof, cm_aoa[0, :])

    return cl, cd, cm


# %% Defining BEM
def BEM(R, r, B, rho, V_o, omega, theta_p, beta, c, thick, tolerance=1e-8, f=0.1):
    """
    Iteratively calculate the induction factors a_1 and a_2 using the BEM algorithm until convergence.

    We are writing the function to work with individual values and will use it later on for varying values

    Arguments:
    a_1 -- Initial axial induction factor
    a_2 -- Initial tangential induction factor
    V_o -- Free stream velocity [m/s]
    omega -- Rotational velocity of blades [rad/s]
    r -- Radial position along the blade [m]
    theta_p -- Global pitch angle [radians]
    beta -- Twist angle [radians]
    C_l -- Lift coefficient
    C_d -- Drag coefficient
    B -- Number of blades
    c -- Chord length at position r [m]
    R -- Total blade radius [m]
    tolerance -- Convergence criterion for change in induction factors (default: 1e-6)
    max_iterations -- Maximum number of iterations allowed (default: 100)
    f -- Relaxation parameter

    Returns:
    a_1 -- Final converged axial induction factor
    a_2 -- Final converged tangential induction factor
    phi -- Final flow angle [radians]
    alpha -- Final angle of attack [radians]
    iterations -- Number of iterations taken to 
    F -- Prandtl's correction factor
    """
    # Step 1: Initialise induction factors
    a_1 = 0
    a_2 = 0

    # compute sigma
    sigma = c * B / (2 * np.pi * r)

    # step 4
    # compute a_1 and a_2
    for i in range(10000):
        # compute phi
        phi = np.arctan((V_o * (1 - a_1)) / ((1+a_2) * omega * r))

        # Get the right data from the thickness using angle of attack and interpolation
        alpha = phi - theta_p - beta
        alpha_deg = np.rad2deg(alpha)
        C_l, C_d, C_m = force_coeffs_10MW(
            alpha_deg, thick, aoa_tab, cl_tab, cd_tab, cm_tab)

        # Compute C_t and C_l
        C_t = C_l * np.sin(phi) - C_d * np.cos(phi)
        C_n = C_l * np.cos(phi) + C_d * np.sin(phi)

        # Apply Prindtl correction
        F_1 = (B / 2) * (R - r) / (r * np.abs(np.sin(phi)))
        F = (2 / np.pi) * np.arccos(np.exp(-F_1))

        # compute C_T
        dC_T = ((1 - a_1)**2 * sigma * C_n) / (F * np.sin(phi)**2)

        # Compute a_1
        if a_1 < (1/3):
            a_1_new = dC_T / (4 * (1 - a_1))
        if a_1 >= (1/3):
            a_1_new = dC_T / (4 * (1 - (1 / 4) * (5 - 3 * a_1) * a_1))
        a_1_new = f * a_1_new + (1 - f) * a_1

        # compute a_2
        a_2_new = ((sigma * C_t) * (1 + a_2)) / \
            (4 * F * np.sin(phi) * np.cos(phi))
        a_2_new = f * a_2_new + (1 - f) * a_2

        if np.abs(a_1_new - a_1) < tolerance and np.abs(a_2_new - a_2) < tolerance:
            a_1 = a_1_new
            a_2 = a_2_new
            break

        else:
            # if np.abs(a_1_new - a_1) > tolerance or np.abs(a_2_new - a_2) > tolerance:
            a_1 = a_1_new
            a_2 = a_2_new

    # Compute Vrel
    V_rel = (V_o * (1 - a_1)) / np.sin(phi)
    V_rel = (omega * r * (1 + a_2)) / np.cos(phi)

    # Compute lift and drag
    l = 0.5 * rho * V_rel**2 * C_l * c
    d = 0.5 * rho * V_rel**2 * C_d * c

    # Compute Pt and Pn
    P_t = -d * np.cos(phi) + l * np.sin(phi)
    P_n = d * np.sin(phi) + l * np.cos(phi)
    return a_1, a_2, P_t, P_n, F


def BEM_Madsen(R, r, B, rho, V_o, omega, theta_p, beta, c, thick, tolerance=1e-8, f=0.1):
    # Step 1: Initial Guess for a_1 and a_2
    a_1 = 0
    a_2 = 0

    # compute sigma
    sigma = c * B / (2 * np.pi * r)

    # step 4
    # compute a_1 and a_2
    for i in range(10000):
        # compute phi
        phi = np.arctan((V_o * (1 - a_1)) / ((1+a_2) * omega * r))

        # Get the right data from the thickness using angle of attack and interpolation
        alpha = phi - theta_p - beta
        C_l, C_d, C_m = force_coeffs_10MW(
            alpha * 360/(2*np.pi), thick, aoa_tab, cl_tab, cd_tab, cm_tab)

        # Compute C_t and C_l
        C_t = C_l * np.sin(phi) - C_d * np.cos(phi)
        C_n = C_l * np.cos(phi) + C_d * np.sin(phi)

        # Apply Prindtl correction
        F_1 = (B / 2) * (R - r) / (r * np.abs(np.sin(phi)))
        F = (2 / np.pi) * np.arccos(np.exp(-F_1))

        # compute C_T
        dC_T = ((1 - a_1)**2 * sigma * C_n) / (F * np.sin(phi)**2)

        # Compute a_1
        a_1_new = 0.246 * dC_T + 0.0586 * dC_T**2 + 0.0883*dC_T**3
        a_1_new = f * a_1_new + (1 - f) * a_1

        # compute a_2
        a_2_new = ((sigma * C_t) * (1 + a_2)) / \
            (4 * F * np.sin(phi) * np.cos(phi))
        a_2_new = f * a_2_new + (1 - f) * a_2

        if np.abs(a_1_new - a_1) < tolerance and np.abs(a_2_new - a_2) < tolerance:
            a_1 = a_1_new
            a_2 = a_2_new
            break

        else:
            # if np.abs(a_1_new - a_1) > tolerance or np.abs(a_2_new - a_2) > tolerance:
            a_1 = a_1_new
            a_2 = a_2_new

    # Compute Vrel
    V_rel = (V_o * (1 - a_1)) / np.sin(phi)
    V_rel = (omega * r * (1 + a_2)) / np.cos(phi)

    # Compute lift and drag
    l = 0.5 * rho * V_rel**2 * C_l * c
    d = 0.5 * rho * V_rel**2 * C_d * c

    # Compute Pt and Pn
    P_t = -d * np.cos(phi) + l * np.sin(phi)
    P_n = d * np.sin(phi) + l * np.cos(phi)
    return a_1, a_2, P_t, P_n, F


# %% Defining a range of values for wind speeds
s = 10  # Length of ranges
V_o_range = np.linspace(4, 25, s)
theta_p_range_deg = np.linspace(-4, 3, s)
theta_p_range_rad = np.deg2rad(theta_p_range_deg)
tip_speed_ratios = np.linspace(4, 10, s)

X_coordinates = np.zeros((s, s))
Y_coordinates = np.zeros((s, s))
Cp_Values = np.zeros((s, s))
Ct_Values = np.zeros((s, s))

# %% Making a loop for calculating Cpmax Glauert
for x in range(s):
    for y in range(s):
        X_coordinates[x][y] = theta_p_range_rad[y]
        Y_coordinates[x][y] = tip_speed_ratios[x]
        R = 89.17  # Position of the blade we are interested in [m]
        B = 3  # Number of blades
        rho = 1.225  # Air density [kg/m^3]
        V_o = 4  # Free stream velocity [m/s]
        theta_p = theta_p_range_rad[y]  # Global pitch angle [radians]
        omega = (tip_speed_ratios[x] * V_o) / R   # Rotational velocity [rad/s]
        tolerance = 1e-8  # Convergence tolerance
        f = 0.1  # Relaxation parameter

        a_1_results = np.zeros(len(r))
        a_2_results = np.zeros(len(r))
        P_t_results = np.zeros(len(r))
        P_n_results = np.zeros(len(r))
        F_results = np.zeros(len(r))

        for i in range(len(r) - 1):
            a_1_results[i], a_2_results[i], P_t_results[i], P_n_results[i], F_results[i] = BEM(
                R, r[i], B, rho, V_o, omega, theta_p, beta_rad[i], c[i], thick[i], tolerance=1e-8, f=0.1)
        T = B * np.trapz(P_n_results, r)
        P = B * omega * np.trapz(P_t_results * r, r)
        C_p = P / (0.5 * rho * V_o**3 * np.pi * R**2)
        C_t = T / (0.5 * rho * V_o**2 * np.pi * R**2)

        Cp_Values[x][y] = C_p
        Ct_Values[x][y] = C_t

# %% Using Madsen

V_o_range = np.linspace(4, 25, s)
theta_p_range_deg = np.linspace(-4, 3, s)
theta_p_range_deg = np.deg2rad(theta_p_range_deg)
tip_speed_ratios = np.linspace(4, 10, s)

X_coordinates = np.zeros((s, s))
Y_coordinates = np.zeros((s, s))
Cp_Values_Madsen = np.zeros((s, s))
Ct_Values_Madsen = np.zeros((s, s))

for x in range(s):
    for y in range(s):
        X_coordinates[x][y] = theta_p_range_deg[y]
        Y_coordinates[x][y] = tip_speed_ratios[x]
        R = 89.17  # Position of the blade we are interested in [m]
        B = 3  # Number of blades
        rho = 1.225  # Air density [kg/m^3]
        V_o = 24  # Free stream velocity [m/s]
        theta_p = theta_p_range_deg[y]  # Global pitch angle [radians]
        omega = (tip_speed_ratios[x] * V_o) / R   # Rotational velocity [rad/s]
        tolerance = 1e-8  # Convergence tolerance
        f = 0.1  # Relaxation parameter

        a_1_results = np.zeros(len(r))
        a_2_results = np.zeros(len(r))
        P_t_results = np.zeros(len(r))
        P_n_results = np.zeros(len(r))
        F_results = np.zeros(len(r))

        for i in range(len(r) - 1):
            a_1_results[i], a_2_results[i], P_t_results[i], P_n_results[i], F_results[i] = BEM_Madsen(
                R, r[i], B, rho, V_o, omega, theta_p, beta_rad[i], c[i], thick[i], tolerance=1e-8, f=0.1)
        T = B * np.trapz(P_n_results, r)
        P = B * omega * np.trapz(P_t_results * r, r)
        C_p = P / (0.5 * rho * V_o**3 * np.pi * R**2)
        C_t = T / (0.5 * rho * V_o**2 * np.pi * R**2)

        Cp_Values_Madsen[x][y] = C_p
        Ct_Values_Madsen[x][y] = C_t

# %% Save as a dataframe
C_p_df = pd.DataFrame(Cp_Values)
C_p_df.to_pickle(f'Cp_Values_Vo_{V_o}_{s}x{s}.pickle')

C_t_df = pd.DataFrame(Ct_Values)
C_t_df.to_pickle(f'Ct_Values_Vo_{V_o}_{s}x{s}.pickle')

X_coordinates_df = pd.DataFrame(X_coordinates)
X_coordinates_df.to_pickle(f'X_cor_{V_o}_{s}x{s}.pickle')
Y_coordinates_df = pd.DataFrame(Y_coordinates)
Y_coordinates_df.to_pickle(f'Y_cor_{V_o}_{s}x{s}.pickle')

# %% Saving Madsen Data
C_p_Madsen_df = pd.DataFrame(Cp_Values_Madsen)
# C_p_Madsen_df.to_parquet(f'Madsen_Cp_Values_Vo_{V_o}_{s}x{s}.parquet')

C_t_Madsen_df = pd.DataFrame(Ct_Values_Madsen)
# C_t_Madsen_df.to_parquet(f'Madsen_Ct_Values_Vo_{V_o}_{s}x{s}.parquet')
# %% Plotting the values for Ct and Cp
n_levels = 50
# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_coordinates, Y_coordinates,
                       Cp_Values, levels=n_levels, cmap='viridis')
# Gives the position of the maximum value when y,x
x, y = np.unravel_index(np.argmax(C_p_df), C_p_df.shape)
X_coordinate = X_coordinates[:, y]
Y_coordinate = Y_coordinates[x, :]
plt.plot(X_coordinate, Y_coordinate, 'r+')
# Add a color bar to show the color scale
cbar = plt.colorbar(contour)
cbar.set_label('Cp Values')

# Label the axes
plt.xlabel('Degrees (Pitch Angle)')
plt.ylabel('Tip Speed (m/s)')

# Add a title
title_Cp = 'Contour Plot of Cp Values'
plt.title(title_Cp)
plt.savefig(f'{title_Cp} with V_o = {V_o} with {s}x{s}')
# Show the plot
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(X_coordinates, Y_coordinates,
                       Ct_Values, levels=n_levels, cmap='viridis')
x, y = np.unravel_index(np.argmax(C_t_df), C_t_df.shape)
X_coordinate = X_coordinates[:, y]
Y_coordinate = Y_coordinates[x, :]
plt.plot(X_coordinate, Y_coordinate, 'r+')
# Add a color bar to show the color scale
cbar = plt.colorbar(contour)
cbar.set_label('Ct Values')

# Label the axes
plt.xlabel('Degrees (Pitch Angle)')
plt.ylabel('Tip Speed (m/s)')

# Add a title
title_Ct = 'Contour Plot of Ct Values'
plt.title(title_Ct)
plt.savefig(f'{title_Ct} with V_o = {V_o} with {s}x{s}')
# Show the plot
plt.show()

# %% Plotting for Madsen
# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_coordinates, Y_coordinates,
                       Cp_Values_Madsen, levels=n_levels, cmap='viridis')
# Gives the position of the maximum value when y,x
x, y = np.unravel_index(np.argmax(Cp_Values_Madsen), Cp_Values_Madsen.shape)
X_coordinate = X_coordinates[:, y]
Y_coordinate = Y_coordinates[x, :]
plt.plot(X_coordinate, Y_coordinate, 'r+')
# Add a color bar to show the color scale
cbar = plt.colorbar(contour)
cbar.set_label('Cp Values')

# Label the axes
plt.xlabel('Degrees (Pitch Angle)')
plt.ylabel('Tip Speed (m/s)')

# Add a title
title_Cp = 'Contour Plot of Cp Values Madsen'
plt.title(title_Cp)
plt.savefig(f'{title_Cp} with V_o = {V_o} with {s}x{s}')
# Show the plot
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(X_coordinates, Y_coordinates,
                       Ct_Values_Madsen, levels=n_levels, cmap='viridis')
x, y = np.unravel_index(np.argmax(Ct_Values_Madsen), C_t_df.shape)
X_coordinate = X_coordinates[:, y]
Y_coordinate = Y_coordinates[x, :]
plt.plot(X_coordinate, Y_coordinate, 'r+')
# Add a color bar to show the color scale
cbar = plt.colorbar(contour)
cbar.set_label('Ct Values')

# Label the axes
plt.xlabel('Degrees (Pitch Angle)')
plt.ylabel('Tip Speed (m/s)')

# Add a title
title_Ct = 'Contour Plot of Ct Values Madsen'
plt.title(title_Ct)
plt.savefig(f'{title_Ct} with V_o = {V_o} with {s}x{s}')
# Show the plot
plt.show()

#%%
#Q2


