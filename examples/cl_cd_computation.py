# Define span positions and angles of attack
# n_r = len(power_curve_df)  # number of radial positions
# print(f'\n n_r = {n_r}')

# n_alpha = len(airfoil_polar)  # number of angle of attack points
# print(f'\n n_alpha = {n_alpha}')

# r_positions = blade_data_df['BlSpn']
# alpha_range = [airfoil_polar['00']['Alpha'].min, airfoil_polar['00']['Alpha'].max]   # degrees

# # Initialize matrices to store results
# Cl_matrix = np.zeros((len(r_positions), len(alpha_range)))
# Cd_matrix = np.zeros((len(r_positions), len(alpha_range)))

# # Get sorted list of all available airfoil sections
# available_sections = sorted(list(airfoil_polar.keys()))
# n_sections = len(available_sections)
# print(f"Using {n_sections} airfoil sections")

# # Compute Cl and Cd for each position and angle
# for i, r in enumerate(r_positions):
#     # Normalize radius for airfoil selection
#     r_norm = r / ROTOR_RADIUS
    
#     # Map normalized radius to available airfoil section
#     section_index = int(round(r_norm * (n_sections - 1)))
#     section = available_sections[section_index]
    
#     # Get polar data for this section
#     polar = airfoil_polar[section]
    
#     # Interpolate Cl and Cd for each angle of attack
#     Cl_matrix[i, :] = np.interp(alpha_range, polar['Alpha'], polar['Cl'])
#     Cd_matrix[i, :] = np.interp(alpha_range, polar['Alpha'], polar['Cd'])

# # Plot results
# plt.figure(figsize=(15, 6))

# # Plot Cl
# plt.subplot(121)
# contour_cl = plt.contourf(alpha_range, r_positions/ROTOR_RADIUS, Cl_matrix, levels=20, cmap='viridis')
# plt.colorbar(contour_cl, label='Cl')
# plt.xlabel('Angle of Attack (degrees)')
# plt.ylabel('r/R')
# plt.title('Lift Coefficient Distribution')
# plt.grid(True)

# # Plot Cd
# plt.subplot(122)
# contour_cd = plt.contourf(alpha_range, r_positions/ROTOR_RADIUS, Cd_matrix, levels=20, cmap='viridis')
# plt.colorbar(contour_cd, label='Cd')
# plt.xlabel('Angle of Attack (degrees)')
# plt.ylabel('r/R')
# plt.title('Drag Coefficient Distribution')
# plt.grid(True)

# plt.tight_layout()

# # Save the plot
# save_path = os.path.join(pictures_dir, 'Cl_Cd_Distribution.png')
# plt.savefig(save_path)
# if show_plot:
#     plt.show()
# plt.close()