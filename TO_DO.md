
Overall Requirements:
- class in src
- a package that implements the required functions
- A README
- <10 min runtime
- tests for package, and test coverage >80%
- pylint score > 8.0

 Specific TO DO
 - develop steady state blade element momentum model to predict aerodynamic performance of WT
 - model to compute key performance metrics: power output, thrust, torque as functions of WS, rotor speed, blade pitch angle

- Write a function to load, parse and plot the provided dataset in inputs, 
  generate at least one figure. Ideally also with test in tests
 - write mathematical functions in init.py
 - Loop over all blade elements
 - compute the power output of the rotor
 - thrust coef

- Functional requirements (to pass)

1. Load and parse the provided turbine data
2. Plot the provided airfoil shapes in one figure
3. Compute lift coefficient (Cl) and drag coefficient (Cd) as function of span position (r) and angle of attack (α)
4. Compute the axial (a) and tangential (a′) induction factors as function of span position (r), the inflow wind speed V0, the blade pitch angle (θp) and the rotational speed ω.
5. Compute the thrust (T), torque (M), and power (P) of the rotor as function of the inflow wind speed V0, the blade pitch angle (θp) and the rotational speed ω.
6. Compute optimal operational strategy, i.e., blade pitch angle (θp) and rotational speed (ω), as function of wind speed (V0), based on the provided operational strategy in IEA_15MW_RWT_Onshore.opt
7. Compute and plot power curve ($P(V_0)$) and thrust curve ($T(V_0)$) based on the optimal operational strategy obtained in the previous function.

Important note: Your package should also implement at least two extra functions besides the required ones listed above.

 

 • Think about the structure/architecture of your package.
