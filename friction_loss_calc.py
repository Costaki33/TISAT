import math
import colebrook
import pandas as pd
from pint import UnitRegistry
from fractions import Fraction

# Constants
SEC_IN_DAY = 86400
BBL_TO_CUBIC_METERS = 0.158987
FEET_TO_METERS = 0.3048
PA_TO_PSI = 6894.76

# Load data
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
tubing_dim_df = pd.DataFrame(tubing_dimensions, columns=['outer_diameter', 'inner_diameter'])
pipedata_df = pd.read_csv(tubing_information)
pipedata_df['modified_dt'] = pd.to_datetime(pipedata_df['modified_dt'])
ureg = UnitRegistry()


def parse_tubing_size(size_str):
    parts = size_str.split()
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        whole_part = float(parts[0])
        fraction_part = Fraction(parts[1])
        return whole_part + fraction_part
    raise ValueError("Invalid tubing size format")


def friction_loss(api_number, injection_date, injected_bbl, packer_depth_ft, b3, pipe_data_df=pipedata_df):
    api_number = int(api_number)
    density = 997  # kg/m^3
    viscosity = 0.001  # Pa·s

    # Find the closest matching row
    matching_rows = pipe_data_df[pipe_data_df['api_no'] == api_number]
    if matching_rows.empty:
        closest_row = pipe_data_df.loc[(pipe_data_df['packer_set'] - packer_depth_ft).abs().idxmin()]
    else:
        closest_row = matching_rows.iloc[
            (matching_rows['modified_dt'] - pd.to_datetime(injection_date)).abs().argsort()[:1]]

    packer_depth_ft = closest_row['packer_set'] if isinstance(closest_row['packer_set'], float) else closest_row['packer_set'].iloc[0]
    tubing_size_str = closest_row['tubing_size'] if isinstance(closest_row['tubing_size'], str) else closest_row['tubing_size'].iloc[0]
    tubing_size = parse_tubing_size(tubing_size_str)
    tubing_dim_row = tubing_dim_df.iloc[(tubing_dim_df['outer_diameter'] - tubing_size).abs().idxmin()]

    outer_diameter_inches = tubing_dim_row['outer_diameter']
    inner_diameter_inches = tubing_dim_row['inner_diameter']
    D_outer = (outer_diameter_inches * ureg.inch).to('meter').magnitude
    D_inner = (inner_diameter_inches * ureg.inch).to('meter').magnitude
    area_m2 = (math.pi / 4) * (D_inner ** 2)

    sec_in_day = SEC_IN_DAY * pd.Period(year=injection_date.year, month=injection_date.month, freq='M').days_in_month if b3 == 1 else SEC_IN_DAY
    flow_rate_m3_s = (injected_bbl * BBL_TO_CUBIC_METERS) / sec_in_day

    fluid_velocity_m_s = flow_rate_m3_s / area_m2
    reynolds_number = (density * fluid_velocity_m_s * D_inner) / viscosity

    if reynolds_number < 2100:
        # print(f"Laminar Flow")
        friction_factor = 64 / reynolds_number
    else:
        # print(f"Turbulent Flow")
        friction_factor = colebrook.sjFriction(reynolds_number, roughness=0.0001)

    packer_depth_m = packer_depth_ft * FEET_TO_METERS
    deltaP_pa = (2 * friction_factor * density * (fluid_velocity_m_s ** 2) * packer_depth_m) / D_inner
    deltaP_psi = deltaP_pa / PA_TO_PSI

    return deltaP_psi
