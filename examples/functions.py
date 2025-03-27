"""functions for the final project"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import re

def read_airfoil_file(file_path):
    """
    Read an airfoil coordinates file and return a DataFrame.
    
    Parameters:
    ----------
    file_path : Path or str
        Path to the airfoil file
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with x/c and y/c columns
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract number of coordinates from first line
    num_coords = int(re.search(r'(\d+)', lines[0]).group(1))
    
    # Skip header and comments, find data start
    data_start = 0
    for i, line in enumerate(lines):
        if '!  x/c        y/c' in line and i > 4:  # Find the coordinate header line
            data_start = i + 1
            break
    
    # Read coordinates
    x_coords = []
    y_coords = []
    
    for i in range(data_start, data_start + num_coords - 1):  # -1 because num_coords includes reference point
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                x_coords.append(float(parts[0]))
                y_coords.append(float(parts[1]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'x/c': x_coords,
        'y/c': y_coords
    })
    
    return df


def read_airfoil_polar_file(file_path):
    """
    Read an airfoil polar data file (.dat) and return a DataFrame.
    
    Parameters:
    ----------
    file_path : Path or str
        Path to the airfoil polar file
        
    Returns:
    -------
    dict
        Dictionary containing metadata and pandas DataFrame with Alpha,
        Cl, Cd, Cm columns
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables to store metadata
    metadata = {}
    data_start = 0
    num_points = 0
    
    # Extract metadata
    for i, line in enumerate(lines):
        line = line.strip()
        if 'Re' in line and '!' in line:
            parts = line.split()
            try:
                metadata['Re'] = float(parts[0])  # Extract just the number
            except ValueError:
                print(f"Warning: Could not parse Reynolds number from line: {line}")
        
        elif 'NumAlf' in line:
            # Extract the number of data points
            parts = line.split()
            try:
                num_points = int(parts[0])
                data_start = i + 2  # Skip the column headers line
                break
            except (ValueError, IndexError):
                print(f"Warning: Could not parse NumAlf from line: {line}")
    
    # If we didn't find the number of points, we can't proceed
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
                    # Skip lines we can't parse
                    continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'Alpha': alpha,
        'Cl': cl,
        'Cd': cd,
        'Cm': cm
    })
    
    return {'metadata': metadata, 'data': df}