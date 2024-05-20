import math
import pandas as pd
import datetime
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize_scalar

tubing_information = '/home/skevofilaxc/Documents/earthquake_data/2024-03-06T08.47.26.270812_2024-03-05 cmpl tubing.py.csv'
tubing_dimensions = [
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


def calculate_objective(friction_factor_guess, epsilon, hydraulic_diameter, reynolds_num):
    colebrook_left_side = 1 / np.sqrt(friction_factor_guess)
    colebrook_right_side = -2 * np.log10(
        (epsilon / (3.7 * hydraulic_diameter)) + (2.51 / (reynolds_num * np.sqrt(friction_factor_guess))))
    objective = (colebrook_left_side - colebrook_right_side) ** 2
    return objective


def friction_loss(api_number, injection_date, injected_bbl):
    # For Newtonian Fluids
    inch_to_meter = 0.0254
    epsilon = 0.0001  # "We have been using 0.0001 (dimensionless) as an estimate, which is a fairly smooth value."
    mu = 0.001  # kg / (m * s), viscosity of water
    fluid_density = 1071.248  # roh: kg/m^3, was given as 8.94 ppg, converted to kg/m^3

    pipe_data_df = pd.read_csv(tubing_information)

    # Change datetime types to make processing easier
    pipe_data_df['modified_dt'] = pd.to_datetime(pipe_data_df['modified_dt'])
    injection_date = datetime.datetime.strptime(injection_date, '%Y-%m-%d %H:%M:%S')

    # Look for the rows whose API numbers match the one provided
    # Find the row with the closest 'modified_dt' to injection_date
    matching_rows = pipe_data_df[pipe_data_df['api_no'] == api_number]
    closest_row = matching_rows.iloc[(matching_rows['modified_dt'] - injection_date).abs().argsort()[:1]]

    # Extract tubing_size (inch) and packer_set (ft) from closest_row
    # Advised by Jim Moore to use packer_set depth in calculations
    tubing_size = closest_row['tubing_size'].iloc[0]
    tubing_size = parse_tubing_size(tubing_size)  # Parse tubing size
    packer_set = closest_row['packer_set'].iloc[0] # Depth of the packer (ft), asked to use by Jim Moore
    packer_depth_m = packer_set * 0.3048  # convert ft to m

    outer_diameter_inches = min(tubing_dimensions, key=lambda x: abs(x[0] - tubing_size))[0]  # in inches
    inner_diameter_inches = next(item[1] for item in tubing_dimensions if item[0] == outer_diameter_inches)  # in inches
    inner_diameter_meters: float = inner_diameter_inches * inch_to_meter


    # Calculate pipe flow area using formula from sample problem on Petrowiki link (Q.4)
    pipe_flow_area_interior = .25 * math.pi * (inner_diameter_meters)**2 # m^2
    fluid_flow_rate = round((injected_bbl * 0.159) / (24 * 3600), 5)  # bbl to m^3/s
    fluid_velocity = fluid_flow_rate / pipe_flow_area_interior  # m/s

    print(f"Inner Diameter inches: {inner_diameter_inches}\nOuter Diameter inches: {outer_diameter_inches}\nInjected BBL: {injected_bbl}\npacker_set: {packer_set}")
    reynolds_num = (fluid_density * fluid_velocity * inner_diameter_meters) / mu
    print(f"Reynolds num: {reynolds_num}")

    # Dh = D2 - D1 (eqn. from https://neutrium.net/fluid-flow/hydraulic-diameter/)
    hydraulic_diameter =inner_diameter_meters

    result = minimize_scalar(calculate_objective, bounds=(0.001, 0.1),
                             args=(epsilon, hydraulic_diameter, reynolds_num))
    f = result.x  # optimized friction factor
    print(f"F: {f}")

    deltaP = ((2 * f * fluid_density * (fluid_velocity ** 2)) / inner_diameter_meters) * packer_depth_m
    deltaP_psi = deltaP / 6895  # Convert Pa to psi
    return deltaP_psi


# Call the function
api_number = 31741196
injection_date = "2021-07-30 00:00:00"
bbl = 8079.0

deltaP = friction_loss(api_number, injection_date, bbl)
print(f"Delta P (psi): {deltaP}")
