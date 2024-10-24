import math
import os
import sys
import csv
import datetime
import colorsys
import warnings
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from matplotlib.lines import Line2D
from friction_loss_calc import friction_loss
from individual_plots import gather_well_data
from pandas.errors import SettingWithCopyWarning
from subplot_dirs import create_indiv_subplot_dirs
from well_data_query import closest_wells_to_earthquake
from read_b3 import clean_csv, b3_data_quality_histogram, calculate_b3_total_bh_pressure, plot_b3_bhp, plot_b3_ijv

# GLOBAL VARIABLES AND FILE PATHS
STRAWN_FORMATION_DATA_FILE_PATH = '/home/skevofilaxc/Documents/earthquake_data/TopStrawn_RD_GCSWGS84.csv'
OUTPUT_DIR = '/home/skevofilaxc/Documents/earthquake_plots'

# Filter out SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def get_earthquake_info_from_csv(csv_string):
    # Parse the CSV string and extract earthquake information
    reader = csv.reader(csv_string.splitlines())
    rows = list(reader)
    event_id, origin_datetime, latitude, longitude, _, magnitude, _ = rows[0]
    origin_datetime = origin_datetime.replace('Z', '')
    origin_date, origin_time = origin_datetime.split('T')

    earthquake_info = {
        'Event ID': event_id,
        'Latitude': float(latitude),
        'Longitude': float(longitude),
        'Origin Date': origin_date,
        'Origin Time': datetime.datetime.strptime(origin_time, '%H:%M:%S.%f').strftime('%H:%M:%S'),
        'Local Magnitude': math.ceil(float(magnitude) * 10) / 10
    }
    return earthquake_info


def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    # print(f"Earthquake Center: {lat1}, {lon2}\nWell: {lat2}, {lon2}")
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    # print(f"Earthquake Center Rad: {lat1_rad}, {lon2_rad}\nWell Rad: {lat2_rad}, {lon2_rad}")
    # Calculate differences in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula to calculate distance
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    # print(f"Distance: {distance}\n")
    return distance


def convert_dates(some_earthquake_origin_date, year_cutoff):
    """
    Converts some earthquake origin date to one year before said earthquake origin date to datetime objects for calcs
    """
    # Convert some_earthquake_origin_date to a datetime object
    year_to_days = 365 * year_cutoff
    some_earthquake_origin_date = datetime.datetime.strptime(some_earthquake_origin_date, '%Y-%m-%d')

    # Convert some_earthquake_origin_date to a datetime object to calculate cutoff before earthquake
    cutoff_before_earthquake_date = some_earthquake_origin_date - datetime.timedelta(days=year_to_days)

    return some_earthquake_origin_date, cutoff_before_earthquake_date


def classify_well_type(well_lat, well_lon, well_depth, strawn_formation_data):
    """
    Function classifys well type between either Shallow or Deep based on the Z-depth of the well in comparison to the
    Z-depth of the closest position of the Strawn Formation
    :param well_lat: Latitude of the well
    :param well_lon: Longitude of the well
    :param well_depth: Depth of the well
    :param strawn_formation_data: DataFrame containing the Strawn Formation data
    :return: 1 for Deep, 0 for Shallow
    """
    # Extract latitude and longitude columns from the DataFrame
    dataset_latitudes = strawn_formation_data['lat_wgs84'].values
    dataset_longitudes = strawn_formation_data['lon_wgs84'].values

    # Calculate the Haversine distance between the well's position and each position in the dataset
    distances = np.array([
        haversine_distance(well_lat, well_lon, lat, lon)
        for lat, lon in zip(dataset_latitudes, dataset_longitudes)
    ])

    # Find the index of the position with the minimum distance
    closest_index = np.argmin(distances)

    # Get Straw Formation Depth
    closest_strawn_depth = strawn_formation_data['Zft_sstvd'].values[closest_index]

    # Classify the well type based on the depth comparison
    if abs(well_depth) + closest_strawn_depth > 0:  # It's deeper
        return 1  # DEEP well type
    else:  # It's above the S.F.
        return 0  # Shallow well type


def is_within_cutoff(injection_date, earthquake_date, cutoff_before_earthquake):
    """
    Checks to see if a given well injection date falls within a given cutoff prior to the earthquake occurring.

    Parameters:
    injection_date (datetime): The date of the well injection.
    earthquake_date (datetime): The date of the earthquake.
    cutoff_before_earthquake (int): The year cutoff investigating prior to the earthquake

    Returns:
    bool: True if the injection date falls within the one-year range before the earthquake date, False otherwise.
    """
    #print(f"Injection Date: {type(injection_date)}")
    if cutoff_before_earthquake <= injection_date <= earthquake_date:
        return True
    else:
        return False


def hsl_to_rgb(h, s, l):
    """Convert HSL to RGB color space."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    r, g, b = (r + m), (g + m), (b + m)
    return int(r * 255), int(g * 255), int(b * 255)


def hex_to_hsl(hex_color):
    """Convert HEX color to HSL."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)


def hsl_to_hex(h, s, l):
    """Convert HSL color to HEX."""
    r, g, b = hsl_to_rgb(h, s, l)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def adjust_lightness(hex_color, adjustment_factor):
    """Adjust the lightness of a color."""
    h, l, s = hex_to_hsl(hex_color)
    l = max(0, min(1, l * adjustment_factor))
    return hsl_to_hex(h, s, l)


def generate_gradient_colors(num_colors, start_color, lightness_adjustment):
    """Generate a list of distinct gradient colors starting from a given color."""
    # Convert start color to HSL
    h, l, s = hex_to_hsl(start_color)
    # Adjust lightness
    l = max(0.1, min(0.9, l * lightness_adjustment))

    # Generate distinct colors by varying the hue
    hue_step = 360 / num_colors
    colors = []
    for i in range(num_colors):
        new_hue = (h * 360 + i * hue_step) % 360
        colors.append(hsl_to_hex(new_hue, s, l))

    return colors


def data_preperation(closest_wells_data_df, earthquake_lat, earthquake_lon, some_earthquake_origin_date,
                     strawn_formation_data, year_cutoff):
    """
    Function checks the injection data for the closest wells to a given earthquake to see if the injection data
    is 'valid', IE. it falls between the origin date of the injection dataset and up to a year post-earthquake,
    so we can see if we should throw out a well in our plotting

    Function also adds the 'WELL TYPE' for a given well, shallow or deep

    :param closest_wells_data_df: dataframe that contains the total well information that's within a given range
    :param earthquake_lat: latitude of the earthquake
    :param earthquake_lon: longitude of the earthquake
    :param some_earthquake_origin_date: the earthquake origin date
    :param current_earthquake_index: the number earthquake that is being reviewed/iterated
    :return: cleaned_df: cleaned dataframe
    """
    api_numbers_to_remove = []

    some_earthquake_origin_date, cutoff_before_earthquake_date = convert_dates(some_earthquake_origin_date, year_cutoff)

    # Group by 'API Number' and find the earliest 'Date of Injection' for each group
    earliest_injection_dates = closest_wells_data_df.groupby('API Number')['Date of Injection'].min()

    # Find all the wells who don't have a valid injection and remove from dataframe
    current_date = datetime.datetime.now()
    current_date_difference = current_date - some_earthquake_origin_date
    # print(f"Earthquake Origin Date: {some_earthquake_origin_date}\nCutoff: {cutoff_before_earthquake_date}"
    #       f"\nDate delta: {current_date_difference}")
    for api_number, injection_date in earliest_injection_dates.items():
        if not is_within_cutoff(injection_date, (some_earthquake_origin_date + current_date_difference),
                                cutoff_before_earthquake_date):
            print(f"Earliest injection date for well #{api_number}, is not within cutoff of the earthquake date."
                  f" Earliest date was: {injection_date}.\nWill omit from computation.\n"
                  f"------------------------------------")
            api_numbers_to_remove.append(api_number)
    cleaned_df = closest_wells_data_df[~closest_wells_data_df['API Number'].isin(api_numbers_to_remove)]

    # Dictionary to store API numbers and their corresponding well types and distances from earthquake
    well_types_map = {}
    distances_map = {}

    # Classify well types for each unique API number
    for api_number, injection_date in earliest_injection_dates.items():
        if api_number not in api_numbers_to_remove:
            well_lat = cleaned_df.loc[cleaned_df['API Number'] == api_number, 'Surface Latitude'].iloc[0]
            well_lon = cleaned_df.loc[cleaned_df['API Number'] == api_number, 'Surface Longitude'].iloc[0]
            well_depth = cleaned_df.loc[cleaned_df['API Number'] == api_number, 'Well Total Depth ft'].iloc[0]
            well_types_map[api_number] = classify_well_type(well_lat, well_lon, well_depth, strawn_formation_data)

            distance = round(haversine_distance(earthquake_lat, earthquake_lon, well_lat, well_lon), 2)
            distances_map[api_number] = distance

    # Add the 'Well Type' column based on the well types map
    cleaned_df['Well Type'] = cleaned_df['API Number'].map(well_types_map)
    # Add the 'Distance from Earthquake (km)' column based on the distances map
    cleaned_df['Distance from Earthquake (km)'] = cleaned_df['API Number'].map(distances_map)

    return cleaned_df


def calculate_total_bottomhole_pressure(cleaned_well_data_df):
    # Method provided by Jim Moore at RRCT that includes friction loss in the tubing string
    # Bottomhole pressure = surface pressure + hydrostatic pressure – flowing tubing friction loss
    # Mud weight: JP Nicot, Jun Ge

    deltaP = 0
    for index, row in cleaned_well_data_df.iterrows():
        api_number = row['API Number']
        volume_injected = row['Volume Injected (BBLs)']
        injection_pressure_avg_psig = row['Injection Pressure Average PSIG']
        injection_pressure_avg_psi = injection_pressure_avg_psig + 14.7
        well_total_depth_ft = row['Well Total Depth ft']
        depth_of_tubing_packer = row['Depth of Tubing Packer']  # ft
        injection_date = row['Date of Injection']

        # if pd.isna(injection_pressure_avg_psig) or float(injection_pressure_avg_psig) == 0:
        #     total_bottomhole_pressure = 0
        # else:
        if pd.isna(depth_of_tubing_packer):
            hydrostatic_pressure = float(0.465 * well_total_depth_ft)  # 0.465 psi/ft X depth (ft)
            if volume_injected == 0:
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure
            elif volume_injected > 0:
                deltaP = friction_loss(api_number, injection_date, volume_injected, well_total_depth_ft,
                                       b3=None)  # in psi
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure - deltaP

        else:
            hydrostatic_pressure = float(0.465 * depth_of_tubing_packer)  # 0.465 psi/ft X depth (ft)
            if volume_injected == 0:
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure
            elif volume_injected > 0:
                deltaP = friction_loss(api_number, injection_date, volume_injected,
                                       depth_of_tubing_packer, b3=None)  # in psi
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure - deltaP

        # Append the total_bottomhole_pressure value to the DataFrame as a new column
        cleaned_well_data_df.loc[index, 'Bottomhole Pressure'] = total_bottomhole_pressure
        cleaned_well_data_df.loc[index, 'deltaP'] = deltaP

    return cleaned_well_data_df


def prepare_total_pressure_data_from_df(finalized_df):
    total_pressure_data = defaultdict(dict)
    distance_data = {}

    for index, row in finalized_df.iterrows():
        api_number = row['API Number']
        date_of_injection = row['Date of Injection']
        bottomhole_pressure = row['Bottomhole Pressure']
        well_type = row['Well Type']
        distance = row['Distance from Earthquake (km)']

        # Convert date_of_injection to a datetime object
        if isinstance(date_of_injection, str):
            date_of_injection = datetime.datetime.strptime(date_of_injection, '%Y-%m-%d')

        if api_number not in total_pressure_data:
            total_pressure_data[api_number] = {'TYPE': well_type}
            distance_data[api_number] = distance  # Store the distance for each API number

        total_pressure_data[api_number][date_of_injection] = bottomhole_pressure

    return total_pressure_data, distance_data


def prepare_listed_pressure_data_from_df(df):
    listed_pressure_data = defaultdict(dict)
    distance_data = {}

    for index, row in df.iterrows():
        api_number = row['API Number']
        date_of_injection = row['Date of Injection']
        listed_pressure = row['Injection Pressure Average PSIG']
        well_type = row['Well Type']
        distance = row['Distance from Earthquake (km)']

        # Convert date_of_injection to a datetime object
        if isinstance(date_of_injection, str):
            date_of_injection = datetime.datetime.strptime(date_of_injection, '%Y-%m-%d')

        if api_number not in listed_pressure_data:
            listed_pressure_data[api_number] = {'TYPE': well_type}
            distance_data[api_number] = distance  # Store the distance for each API number

        listed_pressure_data[api_number][date_of_injection] = listed_pressure

    return listed_pressure_data, distance_data


def prepare_daily_injection_data_from_df(finalized_df):
    daily_injection_data = defaultdict(dict)
    distance_data = {}

    for index, row in finalized_df.iterrows():
        api_number = row['API Number']
        date_of_injection = row['Date of Injection']
        volume_injected = row['Volume Injected (BBLs)']
        well_type = row['Well Type']
        distance = row['Distance from Earthquake (km)']

        # Convert date_of_injection to a datetime object
        if isinstance(date_of_injection, str):
            date_of_injection = datetime.datetime.strptime(date_of_injection, '%Y-%m-%d')

        if api_number not in daily_injection_data:
            daily_injection_data[api_number] = {'TYPE': well_type}
            distance_data[api_number] = distance  # Store the distance for each API number

        if date_of_injection not in daily_injection_data[api_number]:
            daily_injection_data[api_number][date_of_injection] = 0

        daily_injection_data[api_number][date_of_injection] += volume_injected

    return daily_injection_data, distance_data


def plot_total_pressure(total_pressure_data, distance_data, earthquake_info, output_directory, range_km):
    # Create a defaultdict to store the total pressure for each date
    total_pressure_by_date = defaultdict(float)
    deep_pressure_data = defaultdict(list)
    shallow_pressure_data = defaultdict(list)
    all_api_nums = []  # list to store all the api numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)

    if not total_pressure_data:
        print("No data to plot.")
        return

    # Check if total_pressure_data is a dictionary
    if not isinstance(total_pressure_data, dict):
        print("Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in total_pressure_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, pressures = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, total_pressure in zip(unconverted_tuple_dates, pressures):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            total_pressure_by_date[date] += total_pressure

    dates, total_pressure_values = zip(*total_pressure_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding pressures by date
    sorted_data = sorted(zip(dates, total_pressure_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_total_pressure_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in total_pressure_data.items():
        if data['TYPE'] == 1:
            for date, pressure in data.items():
                if date != 'TYPE':
                    deep_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
        elif data['TYPE'] == 0:
            for date, pressure in data.items():
                if date != 'TYPE':
                    shallow_pressure_data[date].append((api_number, pressure))  # Include API number with pressure

    # Save deep well pressure data to a text file
    deep_filename = os.path.join(output_directory,
                                 f'deep_well_bottomholepressure_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(deep_filename, 'w') as f:
        f.write("Date\tAPI Number\tPressure (PSI)\n")
        for date, pressure_points in deep_pressure_data.items():
            for api_number, pressure in pressure_points:
                f.write(f"{date}\t{api_number}\t{pressure}\n")

    # Save shallow well pressure data to a text file
    shallow_filename = os.path.join(output_directory,
                                    f'shallow_well_bottomholepressure_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(shallow_filename, 'w') as f:
        f.write("Date\tAPI Number\tPressure (PSI)\n")
        for date, pressure_points in shallow_pressure_data.items():
            for api_number, pressure in pressure_points:
                f.write(f"{date}\t{api_number}\t{pressure}\n")

    # Combine all API numbers from shallow and deep data
    all_api_numbers = list(set(all_api_nums))
    all_distances = {api_number: distance_data.get(api_number, float('inf')) for api_number in all_api_numbers}
    sorted_all_distances = sorted(all_distances.items(), key=lambda x: x[1])

    # Generate gradient colors for all API numbers
    slightly_darker_shallow = adjust_lightness("#FFDAB9", 0.55)  # FF91A4/7FFFD4
    slightly_darker_deep = adjust_lightness("#E6E6FA", 0.65)
    shallow_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_shallow,
                                              lightness_adjustment=1)
    deep_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_deep, lightness_adjustment=.95)

    # Create a color map for all API numbers
    color_map_shallow = {api_number: color for (api_number, _), color in zip(sorted_all_distances, shallow_colors)}
    color_map_deep = {api_number: color for (api_number, _), color in zip(sorted_all_distances, deep_colors)}

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))  # Create a 2x1 grid for shallow and deep plots

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_pressure_shallow = {}  # Dictionary to store median pressure for each API number over a 3-day span

    for date, pressure_points in shallow_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_shallow:
                api_median_pressure_shallow[api_number] = []
            api_median_pressure_shallow[api_number].append((date, median_pressure))

    all_shallow_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_shallow.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_shallow[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the shallow well data points
        ax1.plot(dates, pressures, marker='o', linestyle='', color=color_map_shallow[api_number], markersize=2)

        # Separate the data points for the category 'Only Volume Injected Provided'
        category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
                                             (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        category_pressures = category_data['Bottomhole Pressure']

        # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        ax1.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
                 markeredgecolor='black', markeredgewidth=0.5, markersize=2.5)

        all_shallow_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(f'event_{earthquake_info["Event ID"]} Bottomhole Pressure Data - Shallow Well ({range_km} KM Range)')
    ax1.set_ylabel('Total Bottomhole Pressure (PSI)')
    ax1.grid(True)
    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    ax1.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_bps:
        # print(f"shallow median bps: {all_shallow_median_bps}")

        # Ensure all values in the list are finite numbers
        valid_bps = [bp for bp in all_shallow_median_bps if np.isfinite(bp)]

        if not valid_bps:
            print("No valid data points found in all_shallow_median_bps.")
        else:
            # Calculate percentiles only with valid data points
            shallow_min, shallow_max = np.percentile(valid_bps, [5, 95])
            # print(f"shallow_min: {shallow_min}, shallow_max: {shallow_max}")

            # Validate the calculated percentiles
            if not np.isfinite(shallow_min) or not np.isfinite(shallow_max):
                print(f"Invalid axis limits: shallow_min={shallow_min}, shallow_max={shallow_max}")
            else:
                ax1.set_ylim(shallow_min, shallow_max)
    else:
        print("No data points available to calculate shallow well pressure limits.")

    # Set major locator and formatter to display ticks for each month
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}  # Reset
    api_median_pressure_deep = {}

    for date, pressure_points in deep_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_deep:
                api_median_pressure_deep[api_number] = []
            api_median_pressure_deep[api_number].append((date, median_pressure))

    all_deep_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_deep.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_deep[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the deep well data points
        ax2.plot(dates, pressures, marker='o', linestyle='', color=color_map_deep[api_number], markersize=2)

        # Separate the data points for the category 'Only Volume Injected Provided'
        category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
                                             (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        category_pressures = category_data['Bottomhole Pressure']

        # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        ax2.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
                 markeredgecolor='black', markeredgewidth=0.5, markersize=2.5)
        all_deep_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(f'event_{earthquake_info["Event ID"]} Bottomhole Pressure Data - Deep Well ({range_km} KM Range)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Bottomhole Pressure (PSI)')
    ax2.grid(True)
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    ax2.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for deep wells using the 5th and 95th percentiles
    if all_deep_median_bps:
        # Convert to a numpy array if it's not already
        all_deep_median_bps = np.array(all_deep_median_bps)

        # Remove NaN values
        filtered_data = all_deep_median_bps[~np.isnan(all_deep_median_bps)]

        # Check if there is any data left after filtering
        if filtered_data.size > 0:
            # Calculate the 5th and 95th percentiles
            deep_min, deep_max = np.percentile(filtered_data, [5, 95])
            ax2.set_ylim(deep_min, deep_max)
        else:
            print("Warning: No valid data available after removing NaNs. Cannot set axis limits.")
    else:
        print("No deep median bps data available.")

    # Set major locator and formatter to display ticks for each month
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save the plot as an image file
    output_filename = os.path.join(output_directory,
                                   f'event_{earthquake_info["Event ID"]}_bottomhole_pressure_range{range_km}km.png')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"Daily bottomhole plots for earthquake: {earthquake_info['Event ID']} were successfully created.")


def create_pressure_txt(listed_pressure_data, distance_data, earthquake_info, output_directory, range_km,
                        cleaned_well_data_df):
    # Create a defaultdict to store the total pressure for each date
    total_pressure_by_date = defaultdict(float)
    deep_pressure_data = defaultdict(list)
    shallow_pressure_data = defaultdict(list)
    all_api_nums = []  # list to store all the api numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)

    if not listed_pressure_data:
        print("No data to plot.")
        return

    # Check if total_pressure_data is a dictionary
    if not isinstance(listed_pressure_data, dict):
        print("Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in listed_pressure_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, pressures = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, total_pressure in zip(unconverted_tuple_dates, pressures):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            total_pressure_by_date[date] += total_pressure

    dates, total_pressure_values = zip(*total_pressure_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding pressures by date
    sorted_data = sorted(zip(dates, total_pressure_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_total_pressure_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in listed_pressure_data.items():
        if data['TYPE'] == 1:
            for date, pressure in data.items():
                if date != 'TYPE':
                    deep_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
        elif data['TYPE'] == 0:
            for date, pressure in data.items():
                if date != 'TYPE':
                    shallow_pressure_data[date].append((api_number, pressure))  # Include API number with pressure

    # Save deep well pressure data to a text file
    deep_filename = os.path.join(output_directory,
                                 f'deep_well_listed_pressure_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(deep_filename, 'w') as f:
        f.write("Date\tAPI Number\tAverage Pressure (PSIG)\n")
        for date, pressure_points in deep_pressure_data.items():
            for api_number, pressure in pressure_points:
                f.write(f"{date}\t{api_number}\t{pressure}\n")

    # Save shallow well pressure data to a text file
    shallow_filename = os.path.join(output_directory,
                                    f'shallow_well_listed_pressure_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(shallow_filename, 'w') as f:
        f.write("Date\tAPI Number\tAverage Pressure (PSIG)\n")
        for date, pressure_points in shallow_pressure_data.items():
            for api_number, pressure in pressure_points:
                f.write(f"{date}\t{api_number}\t{pressure}\n")

    # Combine all API numbers from shallow and deep data
    all_api_numbers = list(set(all_api_nums))
    all_distances = {api_number: distance_data.get(api_number, float('inf')) for api_number in all_api_numbers}
    sorted_all_distances = sorted(all_distances.items(), key=lambda x: x[1])

    # Generate gradient colors for all API numbers
    slightly_darker_shallow = adjust_lightness("#FFDAB9", 0.55)  # FF91A4/7FFFD4
    slightly_darker_deep = adjust_lightness("#E6E6FA", 0.65)
    shallow_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_shallow,
                                              lightness_adjustment=1)
    deep_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_deep, lightness_adjustment=.95)

    # Create a color map for all API numbers
    color_map_shallow = {api_number: color for (api_number, _), color in zip(sorted_all_distances, shallow_colors)}
    color_map_deep = {api_number: color for (api_number, _), color in zip(sorted_all_distances, deep_colors)}

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))  # Create a 2x1 grid for shallow and deep plots

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_pressure_shallow = {}  # Dictionary to store median pressure for each API number over a 3-day span

    for date, pressure_points in shallow_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_shallow:
                api_median_pressure_shallow[api_number] = []
            api_median_pressure_shallow[api_number].append((date, median_pressure))

    all_shallow_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_shallow.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_shallow[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the shallow well data points
        ax1.plot(dates, pressures, marker='o', linestyle='', color=color_map_shallow[api_number], markersize=2)

        # Separate the data points for the category 'Only Volume Injected Provided'
        # category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
        #                                      (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        # category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        # category_pressures = category_data['Injection Pressure Average PSIG']
        #
        # # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        # ax1.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
        #          markeredgecolor='black', markeredgewidth=0.5, markersize=2.5)

        all_shallow_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(f'event_{earthquake_info["Event ID"]} Listed Pressure Data - Shallow Well ({range_km} KM Range)')
    ax1.set_ylabel('Listed Average Pressure (PSIG)')
    ax1.grid(True)
    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    ax1.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_bps:
        # print(f"shallow median bps: {all_shallow_median_bps}")

        # Ensure all values in the list are finite numbers
        valid_bps = [bp for bp in all_shallow_median_bps if np.isfinite(bp)]

        if not valid_bps:
            print("No valid data points found in all_shallow_median_bps.")
        else:
            # Calculate percentiles only with valid data points
            shallow_min, shallow_max = np.percentile(valid_bps, [5, 95])
            # print(f"shallow_min: {shallow_min}, shallow_max: {shallow_max}")

            # Validate the calculated percentiles
            if not np.isfinite(shallow_min) or not np.isfinite(shallow_max):
                print(f"Invalid axis limits: shallow_min={shallow_min}, shallow_max={shallow_max}")
            else:
                ax1.set_ylim(shallow_min, shallow_max)
    else:
        print("No data points available to calculate shallow well pressure limits.")

    # Set major locator and formatter to display ticks for each month
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}  # Reset
    api_median_pressure_deep = {}

    for date, pressure_points in deep_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_deep:
                api_median_pressure_deep[api_number] = []
            api_median_pressure_deep[api_number].append((date, median_pressure))

    all_deep_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_deep.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_deep[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the deep well data points
        ax2.plot(dates, pressures, marker='o', linestyle='', color=color_map_deep[api_number], markersize=2)

        # Separate the data points for the category 'Only Volume Injected Provided'
        # category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
        #                                      (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        # category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        # category_pressures = category_data['Injection Pressure Average PSIG']
        #
        # # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        # ax2.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
        #          markeredgecolor='black', markeredgewidth=0.5, markersize=2.5)
        all_deep_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(f'event_{earthquake_info["Event ID"]} Listed Pressure Data - Deep Well ({range_km} KM Range)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Listed Average Pressure (PSIG)')
    ax2.grid(True)
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    ax2.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for deep wells using the 5th and 95th percentiles
    if all_deep_median_bps:
        # Convert to a numpy array if it's not already
        all_deep_median_bps = np.array(all_deep_median_bps)

        # Remove NaN values
        filtered_data = all_deep_median_bps[~np.isnan(all_deep_median_bps)]

        # Check if there is any data left after filtering
        if filtered_data.size > 0:
            # Calculate the 5th and 95th percentiles
            deep_min, deep_max = np.percentile(filtered_data, [5, 95])
            ax2.set_ylim(deep_min, deep_max)
        else:
            print("Warning: No valid data available after removing NaNs. Cannot set axis limits.")
    else:
        print("No deep median bps data available.")

    # Set major locator and formatter to display ticks for each month
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save the plot as an image file
    output_filename = os.path.join(output_directory,
                                   f'event_{earthquake_info["Event ID"]}_listed_avg_pressure_range{range_km}km.png')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"Daily AVG PSIG Pressure plots for earthquake: {earthquake_info['Event ID']} were successfully created.")


def plot_daily_injection(daily_injection_data, distance_data, earthquake_info, output_directory, range_km):
    # Create a defaultdict to store the daily injection for each date
    daily_injection_by_date = defaultdict(float)
    deep_injection_data = defaultdict(list)
    shallow_injection_data = defaultdict(list)
    all_api_nums = []  # list to store all the API numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)

    if not daily_injection_data:
        print("No data to plot.")
        return

    # Check if daily_injection_data is a dictionary
    if not isinstance(daily_injection_data, dict):
        print("Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in daily_injection_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, injections = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, daily_injection in zip(unconverted_tuple_dates, injections):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            daily_injection_by_date[date] += daily_injection

    dates, daily_injection_values = zip(*daily_injection_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding injections by date
    sorted_data = sorted(zip(dates, daily_injection_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_daily_injection_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in daily_injection_data.items():
        if data['TYPE'] == 1:
            for date, injection in data.items():
                if date != 'TYPE':
                    deep_injection_data[date].append((api_number, injection))  # Include API number with injection
        elif data['TYPE'] == 0:
            for date, injection in data.items():
                if date != 'TYPE':
                    shallow_injection_data[date].append((api_number, injection))  # Include API number with injection

    # Save deep well injection data to a text file
    deep_filename = os.path.join(output_directory,
                                 f'deep_well_injection_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(deep_filename, 'w') as f:
        f.write("Date\tAPI Number\tInjection (BBLs)\n")
        for date, injection_points in deep_injection_data.items():
            for api_number, injection in injection_points:
                f.write(f"{date}\t{api_number}\t{injection}\n")

    # Save shallow well injection data to a text file
    shallow_filename = os.path.join(output_directory,
                                    f'shallow_well_injection_data_{earthquake_info["Event ID"]}_range{range_km}km.txt')
    with open(shallow_filename, 'w') as f:
        f.write("Date\tAPI Number\tInjection (BBLs)\n")
        for date, injection_points in shallow_injection_data.items():
            for api_number, injection in injection_points:
                f.write(f"{date}\t{api_number}\t{injection}\n")

    # Combine all API numbers from shallow and deep data
    all_api_numbers = list(set(all_api_nums))
    all_distances = {api_number: distance_data.get(api_number, float('inf')) for api_number in all_api_numbers}
    sorted_all_distances = sorted(all_distances.items(), key=lambda x: x[1])

    # Generate gradient colors for all API numbers
    slightly_darker_shallow = adjust_lightness("#FFDAB9", 0.55)  # FF91A4/7FFFD4
    slightly_darker_deep = adjust_lightness("#E6E6FA", 0.65)
    shallow_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_shallow,
                                              lightness_adjustment=1)
    deep_colors = generate_gradient_colors(len(sorted_all_distances), slightly_darker_deep, lightness_adjustment=.95)

    # Create a color map for all API numbers
    color_map_shallow = {api_number: color for (api_number, _), color in zip(sorted_all_distances, shallow_colors)}
    color_map_deep = {api_number: color for (api_number, _), color in zip(sorted_all_distances, deep_colors)}

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_injection = {}  # Dictionary to store median injection for each API number over a 3-day span

    for date, injection_points in shallow_injection_data.items():
        api_injection_values = {}
        for api_number, injection in injection_points:
            if api_number not in api_injection_values:
                api_injection_values[api_number] = []
            api_injection_values[api_number].append(injection)

        for api_number, injection_values in api_injection_values.items():
            median_injection = np.median(injection_values)
            if api_number not in api_median_injection:
                api_median_injection[api_number] = []
            api_median_injection[api_number].append((date, median_injection))

    all_shallow_median_injections = []

    for api_number, median_injection_points in api_median_injection.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_shallow[api_number])
        dates, injections = zip(*median_injection_points)
        ax1.plot(dates, injections, marker='o', linestyle='', color=color_map_shallow[api_number], markersize=2)
        all_shallow_median_injections.extend(injections)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(f'event_{earthquake_info["Event ID"]} Daily Injection Data - Shallow Well ({range_km} KM Range)')
    ax1.set_ylabel('Daily Injection (BBLs)')
    ax1.set_xlabel('Date')
    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='medium', ncol=2)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_injections:
        shallow_min, shallow_max = np.percentile(all_shallow_median_injections, [5, 95])
        ax1.set_ylim(shallow_min, shallow_max)

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_injection = {}  # Dictionary to store median injection for each API number over a 3-day span

    for date, injection_points in deep_injection_data.items():
        api_injection_values = {}
        for api_number, injection in injection_points:
            if api_number not in api_injection_values:
                api_injection_values[api_number] = []
            api_injection_values[api_number].append(injection)

        for api_number, injection_values in api_injection_values.items():
            median_injection = np.median(injection_values)
            if api_number not in api_median_injection:
                api_median_injection[api_number] = []
            api_median_injection[api_number].append((date, median_injection))

    all_deep_median_injections = []

    for api_number, median_injection_points in api_median_injection.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_deep[api_number])
        dates, injections = zip(*median_injection_points)
        ax2.plot(dates, injections, marker='o', linestyle='', color=color_map_deep[api_number], markersize=2)
        all_deep_median_injections.extend(injections)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if x_min <= origin_date_num <= x_max:
        ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(f'event_{earthquake_info["Event ID"]} Daily Injection Data - Deep Well ({range_km} KM Range)')
    ax2.set_ylabel('Daily Injection (BBLs)')
    ax2.set_xlabel('Date')
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='medium', ncol=2)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_deep_median_injections:
        deep_min, deep_max = np.percentile(all_deep_median_injections, [5, 95])
        ax2.set_ylim(deep_min, deep_max)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save the plot to a file
    output_file_path = os.path.join(output_directory,
                                    f"daily_injection_plot_{earthquake_info['Event ID']}_range{range_km}km.png")
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"Daily injection plots for earthquake: {earthquake_info['Event ID']} were successfully created.")


def create_well_histogram_per_api(cleaned_well_data_df, range_km, output_directory=None):
    # Convert 'Date of Injection' to datetime
    cleaned_well_data_df['Date of Injection'] = pd.to_datetime(cleaned_well_data_df['Date of Injection'],
                                                               errors='coerce')

    # Define the conditions and categories
    conditions = [
        (cleaned_well_data_df['Injection Pressure Average PSIG'].notna() & (
                cleaned_well_data_df['Injection Pressure Average PSIG'] != 0) &
         cleaned_well_data_df['Volume Injected (BBLs)'].notna()),
        (cleaned_well_data_df['Volume Injected (BBLs)'].notna() &
         (cleaned_well_data_df['Injection Pressure Average PSIG'].isna() | (
                 cleaned_well_data_df['Injection Pressure Average PSIG'] == 0))),
        ((cleaned_well_data_df['Injection Pressure Average PSIG'].isna() | (
                cleaned_well_data_df['Injection Pressure Average PSIG'] == 0)) &
         cleaned_well_data_df['Volume Injected (BBLs)'].isna())]
    categories = ['Both Volume Injected and Pressure Provided', 'Only Volume Injected Provided',
                  'Neither Value Provided']

    # Apply the conditions to create a new 'Category' column
    cleaned_well_data_df['Category'] = np.select(conditions, categories, default='Unknown')
    cleaned_well_data_df['Month-Year'] = cleaned_well_data_df['Date of Injection'].dt.to_period('M')
    cleaned_well_data_df.sort_values(by='Date of Injection', inplace=True)

    # Group by 'API Number'
    grouped = cleaned_well_data_df.groupby('API Number')
    histograms = {}

    for api_number, group in grouped:
        # Ensure chronological order within each 'Month-Year' by sorting by 'Date of Injection'
        group = group.sort_values(by='Date of Injection')

        # Count the occurrences in each category for each date within each month-year
        category_counts = group.groupby(['Month-Year', 'Date of Injection', 'Category']).size().unstack(
            fill_value=0).reindex(columns=categories, fill_value=0)

        category_counts = category_counts.reset_index()
        category_counts.set_index(['Month-Year', 'Date of Injection'], inplace=True)
        category_counts = category_counts.sort_index(level=['Month-Year', 'Date of Injection'])
        category_counts = category_counts.groupby(level=['Month-Year']).cumsum()

        monthly_totals = category_counts.groupby(level='Month-Year').last()
        monthly_totals = monthly_totals.reindex(columns=categories)

        total_counts = category_counts.sum()
        total_sum = total_counts.sum()
        percentages = (total_counts / total_sum) * 100

        # Create legend labels with percentages
        legend_labels = [f'{category} ({count} records, {percent:.2f}%)'
                         for category, count, percent in zip(categories, total_counts, percentages)]

        # Plot the histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_totals.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Well Data for API #{api_number} (Total Records: {total_sum}) ({range_km} KM Range)')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Days')
        ax.legend(legend_labels, title='Category', loc='upper right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()

        histograms[api_number] = fig

        # Save the plot to a file
        if output_directory:
            plot_filename = os.path.join(output_directory, f'well_data_histogram_{api_number}_range{range_km}km.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', format='png')
            plt.close()

    return histograms


if len(sys.argv) > 1:
    if sys.argv[1] == '0':
        """
        B3 CODE 
        """
        # Prompt user to input the output directory file path
        output_dir = input("Enter the output directory file path: ")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # B3 Data Input
        b3csvfile_path = input("Please provide B3 data filepath (In CSV format): ")

        # Earthquake Info in CSV format
        print("\nClick on the following link to fetch earthquake data:")
        earthquake_info_url = "http://scdb.beg.utexas.edu/fdsnws/event/1/builder"
        print(earthquake_info_url)

        webbrowser.open(earthquake_info_url)
        csv_data = input("Enter the earthquake data in CSV format: ")

        earthquake_info = get_earthquake_info_from_csv(csv_data)

        print(f"\nInformation about the current earthquake:")
        print(earthquake_info, "\n")
        earthquake_latitude = earthquake_info['Latitude']
        earthquake_longitude = earthquake_info['Longitude']
        earthquake_origin_date = earthquake_info['Origin Date']

        # User-provided values for range_km
        range_km = float(input("Enter the range in kilometers (E.g. 20km): "))

        b3df = clean_csv(b3csv=b3csvfile_path, earthquake_information=earthquake_info,
                         strawn_formation_info=STRAWN_FORMATION_DATA_FILE_PATH)
        prepared_b3df = calculate_b3_total_bh_pressure(cleaned_b3df=b3df)

        output_file_path = os.path.join(output_dir, "b3_cleaned.csv")
        prepared_b3df.to_csv(output_file_path, index=False)

        b3_data_quality_histogram(prepared_b3df, range_km, output_dir)
        plot_b3_bhp(prepared_b3df, earthquake_info, output_dir, range_km)
        plot_b3_ijv(prepared_b3df, earthquake_info, output_dir, range_km)
        create_indiv_subplot_dirs(base_dir=output_dir)
        gather_well_data(base_path=output_dir)

        quit()

    elif sys.argv[1] == '1':
        """
        CATALOG CODE
        """
        # Prompt user to input the output directory file path
        output_dir = input("Enter the output directory file path: ")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Click on the following link to fetch earthquake data:")
        earthquake_info_url = "http://scdb.beg.utexas.edu/fdsnws/event/1/builder"
        print(earthquake_info_url)

        webbrowser.open(earthquake_info_url)
        csv_data = input("Enter the earthquake data in CSV format: ")

        earthquake_info = get_earthquake_info_from_csv(csv_data)

        print(f"\nInformation about the current earthquake:")
        print(earthquake_info, "\n")
        earthquake_latitude = earthquake_info['Latitude']
        earthquake_longitude = earthquake_info['Longitude']
        earthquake_origin_date = earthquake_info['Origin Date']

        # User-provided values for range_km
        range_km = float(input("Enter the range in kilometers (E.g. 20km): "))
        year_cutoff = int(input("\nEnter the year cutoff you would like to analyze prior to "
                                "the earthquake: (E.g. 5 yrs): "))
        closest_well_data_df = closest_wells_to_earthquake(center_lat=earthquake_latitude,
                                                           center_lon=earthquake_longitude,
                                                           radius_km=range_km)

        strawn_formation_data = pd.read_csv(STRAWN_FORMATION_DATA_FILE_PATH, delimiter=',')
        cleaned_well_data_df = data_preperation(closest_well_data_df, earthquake_latitude, earthquake_longitude,
                                                earthquake_origin_date, strawn_formation_data, year_cutoff)

        output_file = f'{output_dir}/ivrt_cleaned_well_data.csv'
        output_file2 = f'{output_dir}/ivrt_finalized_well_data.csv'

        cleaned_well_data_df.to_csv(output_file, index=False)

        listed_pressure_data, distance_data3 = prepare_listed_pressure_data_from_df(df=cleaned_well_data_df)
        create_pressure_txt(listed_pressure_data, distance_data3, earthquake_info, output_dir, range_km,
                            cleaned_well_data_df=cleaned_well_data_df)

        histograms = create_well_histogram_per_api(cleaned_well_data_df, range_km, output_dir)
        finalized_df = calculate_total_bottomhole_pressure(cleaned_well_data_df=cleaned_well_data_df)

        finalized_df.to_csv(output_file2, index=False)

        total_pressure_data, distance_data = prepare_total_pressure_data_from_df(finalized_df)
        daily_injection_data, distance_data2 = prepare_daily_injection_data_from_df(finalized_df)

        plot_total_pressure(total_pressure_data, distance_data, earthquake_info, output_dir, range_km)
        plot_daily_injection(daily_injection_data, distance_data2, earthquake_info, output_dir, range_km)
        create_indiv_subplot_dirs(base_dir=output_dir)
        gather_well_data(base_path=output_dir, csv_file=output_file, earthquake_info=earthquake_info)

        quit()
    else:
        print("Invalid input. Please enter '0' or '1'.")
else:
    print("Please provide an input argument ('0' or '1').")
