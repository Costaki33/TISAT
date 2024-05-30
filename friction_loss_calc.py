import math
import colebrook
import pandas as pd
import datetime
import numpy as np
from fractions import Fraction

tubing_information = '/home/skevofilaxc/Documents/earthquake_data/2024-03-06T08.47.26.270812_2024-03-05 cmpl tubing.py.csv'
tubing_dimensions = [
    # [Outer Diameter (in), Inner Diameter (in)]
    # Corresponding OD, ID
    [2.063, 1.751],
    [2.375, 2],
    [2.875, 2.4],
    [3.5, 2.9],
    [4, 3.5],
    [4.5, 3.9],
    [5, 4.2],
    [5.5, 4.7],
    [7, 6.3]
]


def parse_tubing_size(size_str):
    # Split the string into parts
    parts = size_str.split()
    if len(parts) == 1:
        # If only one part, assume it's a whole number
        return float(parts[0])
    elif len(parts) == 2:
        # If two parts, assume first part is whole number and second part is fraction
        whole_part = float(parts[0])
        fraction_part = Fraction(parts[1])
        return whole_part + fraction_part
    return None


def friction_loss(api_number, injection_date, injected_bbl, packer_depth_ft):
    # For Newtonian Fluids
    pipe_data_df = pd.read_csv(tubing_information)

    # Look for the rows whose API numbers match the one provided
    # Find the row with the closest 'modified_dt' to injection_date
    pipe_data_df['modified_dt'] = pd.to_datetime(pipe_data_df['modified_dt'])
    injection_date = pd.to_datetime(injection_date)  # temp
    matching_rows = pipe_data_df[pipe_data_df['api_no'] == api_number]

    if matching_rows.empty:
        # This means that the API number wasn't found, instead we are going to find the row with the
        # closest packer_set to the inputted packer_depth
        print(f"API Number not found")
        closest_row_index = (pipe_data_df['packer_set'] - packer_depth_ft).abs().idxmin()
        closest_row = pipe_data_df.loc[[closest_row_index]]
    else:
        print("API Number Found")
        closest_row = matching_rows.iloc[(matching_rows['modified_dt'] - injection_date).abs().argsort()[:1]]
        # print(f"Closest Row:\n{closest_row}")
        # print(f"Packer depth: {packer_depth_ft}")

    tubing_size = closest_row['tubing_size'].iloc[0]
    tubing_size = parse_tubing_size(tubing_size)  # Parse tubing size

    outer_diameter_inches = min(tubing_dimensions, key=lambda x: abs(x[0] - tubing_size))[0]  # in inches
    inner_diameter_inches = next(item[1] for item in tubing_dimensions if item[0] == outer_diameter_inches)  # in inches

    # roh = 64.3 lbm/ft^3 = 1.03-specific gravity water
    # Viscosity of Water at bottomhole condiitons = 0.6 cp
    # print(f"Injected BBL: {injected_bbl}\nInner Diameter (in): {inner_diameter_inches}")
    newtonian_reynolds = (1.48 * injected_bbl * 64.3) / (0.6 * inner_diameter_inches)
    # print(f"Newtonian Reynolds: {newtonian_reynolds}")

    if newtonian_reynolds < 2100:  # Laminar Flow
        print("Laminar Flow")
        friction_factor = 16 / newtonian_reynolds
    else:  # Turbulent Flow
        print("Turbulent Flow")
        # Pipe roughness "We have been using 0.0001 (dimensionless) as an estimate, which is a fairly smooth value."
        friction_factor = colebrook.sjFriction(newtonian_reynolds, roughness=0.0001)

    # print(f"Friction Factor: {friction_factor}")
    # Fanning Equation - Frictional Pressure Drop Equation
    fluid_velocity_ft_s = (4 * injected_bbl * 5.615 * (1 / 86400)) / (math.pi * (inner_diameter_inches / 12)**2)  # ft/s
    deltaP = ((2 * friction_factor * 64.3 * (fluid_velocity_ft_s**2) * packer_depth_ft)
              / (32.17 * (inner_diameter_inches / 12))) * (1/144) # PSI
    return deltaP


# API NUM: 32938683
# Injected BBL: 4372.0
# Injection Date: 2020-09-30 00:00:00
# Packer Depth: 6130

# API NUM: 13506723
# Injected BBL: 299.0
# Injection Date: 2020-10-01 00:00:00
# Packer Depth: 13316

# API NUM: 32936971
# Injected BBL: 278.0
# Injection Date: 2020-09-30 00:00:00
# Packer Depth: 6000

# deltaP_psi = friction_loss(api_number=32936971, injection_date='2020-09-30 00:00:00', injected_bbl=278.0,
#                            packer_depth_ft=6000)
# print(f"Delta P (psi): {deltaP_psi}")
