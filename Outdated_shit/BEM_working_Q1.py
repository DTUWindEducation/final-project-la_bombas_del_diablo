#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:02:23 2024

@author: josephabbott
"""

import numpy as np
import math

beta=np.loadtxt("bladedat.txt",usecols=1).astype(float)#beta vector
c = np.loadtxt("bladedat.txt",usecols=2).astype(float)#chord vector
r= np.loadtxt("bladedat.txt",usecols=0).astype(float)#radius vector
t = (np.loadtxt("bladedat.txt",usecols=3)*c/100).astype(float)

def BEM_steady(V0, Theta_p, omega):
    """
    Steady Blade Element Momentum (BEM) method for calculating local loads on a wind turbine blade.
    """
    
  
    
    # Initialize axial and tangential induction factors
    a = 0
    a_prime = 0
    stopA = []#loop stopper 
    stopA.append([a,a_prime])
    tol = 1e-6
    beta = 2
    iterations=0 
    ratio_o = []
    ratio_t = []
    PT=[]
    CTV=[]
    LIFT=[]
    DRAG=[]
    

    # Other fixed inputs
    R = 86  # Rotor radius (m)
    B = 3  # Number of blades
    rho = 1.225  # Air density (kg/m^3)
    V0 = 8  # Freestream wind speed (m/s)
    Theta_p =  -3 # Global pitch angle (degrees)
    omega = 2.61  # Rotor angular velocity (rad/s)
    cl = 0.5  # Lift coefficient
    cd = 0.01 
    
    for i in range(len(r)):
        
        while True:
            iterations += 1
      
        # Compute the flow angle φ
            phi = (np.arctan(1 - a) * V0) / ((1 + a_prime) * omega * r[i])
        
        #Comput the angle of attack
            phi_deg= phi*360/(2*np.pi)
        
            alpha = phi_deg - beta - Theta_p
        
        # Compute the normal (Cn) and tangential (Ct) force coefficients
            Cn = cl * np.cos(phi) + cd * np.sin(phi)
            Ct = cl * np.sin(phi) - cd * np.cos(phi)
        
        # Compute the local solidity
            sigma = B * c[i] / (2 * np.pi * r[i])
        
        # Compute correction coefficient F
            Cte = -B/2 * (R-r[i])/(r[i]*math.sin(abs(phi)))
            F = 2/math.pi * math.acos(math.exp(Cte))
        
        # Compute the thrust
            CT = ((1 - a)**2 * Cn * sigma) / (F * np.sin(phi)**2)
        
        # Update axial and tangential induction factors
            if a<0.33:
                a = sigma*Cn*(1-a)/4*F*np.sin(phi)
        
            else:
                a_s = CT/(4*(1-0.25*(5-3*a)*a))
                a=0.1*a_s+(1-0.1)*a_s
            
            a_prime= sigma*Ct*(1+a_prime)/4*np.sin(phi)*np.cos(phi)
        
            stopA.append([a,a_prime])
            
            ratio_o = abs((stopA[iterations][0]-stopA[iterations-1][0])/stopA[iterations][0])
            ratio_t = abs((stopA[iterations][1]-stopA[iterations-1][1])/stopA[iterations][1])
            if ratio_o and ratio_t < tol : 
				#print("Convergence at: "+str(iterations-1)+" iterations ")
                break
            elif iterations > 2000:
                print("Couldn't converge")
                break
            
        
        #work on covonvergence.
            Vr = math.sqrt((V0-a*V0)**2 + (omega * r[i] + a_prime * omega *r[0])**2)
            pt= 1/2 * rho * Vr**2 * c[i] * Ct
            pn= 1/2 * rho * Vr**2 * c[i] * Cn
            l = cl * 0.5 *  rho * Vr**2 * c[i]
            d = cd * 0.5 * rho * Vr**2 * c[i]
        
        PT.append(pt)
        CTV.append(pn)
        LIFT.append(l)
        DRAG.append(d)
        
        lam = omega * R / V0
        CTV = np.array(CTV)
        PT = np.array(PT)
        
        Thrust = B*np.trapz(CTV,r)
        POWER = (B * omega * np.trapz(pt*r,r))
        
        
        
        
        return(POWER,Thrust)
        
        
        
V0=8
Theta_p=-3
omega=2.61      

       
# Define the inputs



BEM_steady(V0, Theta_p, omega)
# Print the results
