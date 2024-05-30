import os
import sys
import csv
import datetime
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import colorspacious as csp
from math import radians, sin, cos, sqrt, atan2
from well_data_query import closest_wells_to_earthquake
from matplotlib.lines import Line2D
from friction_loss_calc import friction_loss
from collections import defaultdict

# GLOBAL VARIABLES AND FILE PATHS
STRAWN_FORMATION_DATA_FILE_PATH = '/home/skevofilaxc/Documents/earthquake_data/TopStrawn_RD_GCSWGS84.csv'
OUTPUT_DIR = '/home/skevofilaxc/Documents/earthquake_plots'


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
        'Local Magnitude': round(float(magnitude), 2)
    }
    return earthquake_info


def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate differences in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula to calculate distance
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance


def convert_dates(some_earthquake_origin_date):
    """
    Converts some earthquake origin date and one year before said earthquake origin date to datetime objects for calcs
    """
    # Convert some_earthquake_origin_date to a datetime object
    some_earthquake_origin_date = datetime.datetime.strptime(some_earthquake_origin_date, '%Y-%m-%d')

    # Convert some_earthquake_origin_date to a datetime object to calculate 1 year before origin date
    one_year_after_earthquake_date = some_earthquake_origin_date + datetime.timedelta(days=365)

    return some_earthquake_origin_date, one_year_after_earthquake_date


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

    # Convert well position to numpy array for vectorized operations
    well_position = np.array([well_lat, well_lon])

    # Convert dataset positions to numpy array for vectorized operations
    dataset_positions = np.column_stack((dataset_latitudes, dataset_longitudes))

    # Calculate the Euclidean distance between the well's position and each position in the dataset
    distances = np.linalg.norm(dataset_positions - well_position, axis=1)

    # Find the index of the position with the minimum distance
    closest_index = np.argmin(distances)

    # Get Straw Formation Depth
    closest_strawn_depth = strawn_formation_data['Zft_sstvd'].values[closest_index]

    # Classify the well type based on the depth comparison
    if abs(well_depth) + closest_strawn_depth > 0:  # It's deeper
        return 1  # DEEP well type
    else:  # It's above the S.F.
        return 0  # Shallow well type


def is_within_one_year(injection_date, one_year_after_earthquake_date):
    """
    Checks to see if a given well injection date falls within 1 year after to the earthquake occurring.
    We want to do this to see if the injection data is valid in creating a timeline;
    We don't want to get a timeline after the earthquake occurred, we want prior
    If the injection date falls within the 1 year range, great! Return True for further calculations
    If the injection date doesn't fall within the 1 year range, Return False and write to file
    """
    if injection_date > one_year_after_earthquake_date:
        return False

    if injection_date <= one_year_after_earthquake_date:
        return True


def generate_gradient_colors(num_colors, start_color, end_color, gamma):
    # Convert start and end colors to RGB format
    start_rgb = mcolors.to_rgb(start_color)
    end_rgb = mcolors.to_rgb(end_color)

    # Convert RGB to LAB color space
    start_lab = csp.cspace_convert(start_rgb, "sRGB1", "CIELab")
    end_lab = csp.cspace_convert(end_rgb, "sRGB1", "CIELab")

    # Generate positions with non-linear transformation for better color distinction
    if num_colors < 2:
        return [mcolors.to_hex(mcolors.rgb_to_hsv(start_rgb))] * num_colors

    position = np.linspace(0, 1, num_colors)
    position = position ** (1/gamma)  # Apply non-linear transformation

    # Interpolate in the LAB color space
    interpolated_lab = np.array([np.interp(position, [0, 1], [start_lab[i], end_lab[i]]) for i in range(3)]).T

    # Convert LAB back to RGB
    interpolated_rgb = csp.cspace_convert(interpolated_lab, "CIELab", "sRGB1")

    # Ensure RGB values are within the valid range [0, 1]
    interpolated_rgb = np.clip(interpolated_rgb, 0, 1)

    # Convert RGB to hex for better compatibility with plotting libraries
    colors = [mcolors.to_hex(rgb) for rgb in interpolated_rgb]

    return colors


def data_preperation(closest_wells_data_df, earthquake_lat, earthquake_lon, some_earthquake_origin_date,
                     strawn_formation_data):
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

    some_earthquake_origin_date, one_year_after_earthquake_date = convert_dates(some_earthquake_origin_date)

    # Group by 'API Number' and find the earliest 'Date of Injection' for each group
    earliest_injection_dates = closest_wells_data_df.groupby('API Number')['Date of Injection'].min()

    # Find all the wells who don't have a valid injection and remove from dataframe
    for api_number, injection_date in earliest_injection_dates.items():
        if not is_within_one_year(injection_date, one_year_after_earthquake_date):
            print(f"Earliest injection date for well #{api_number}, is not within 1 year of the earthquake date."
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

    for index, row in cleaned_well_data_df.iterrows():
        api_number = row['API Number']
        volume_injected = row['Volume Injected (BBLs)']
        injection_pressure_avg_psig = row['Injection Pressure Average PSIG']
        injection_pressure_avg_psi = injection_pressure_avg_psig + 14.7
        well_total_depth_ft = row['Well Total Depth ft']
        depth_of_tubing_packer = row['Depth of Tubing Packer']  # ft
        injection_date = row['Date of Injection']

        # print(f"Well Depth: {well_total_depth_ft}\nDepth of Packer: {depth_of_tubing_packer}\n
        #print(f"Injection Pressure: {injection_pressure_avg_psi}")
        # if injection_pressure_avg_psi == 14.7:
            # print(f"Injection Pressure is 0")
        #     print(f"\nAPI Number: {api_number}")
        #     print(f"Volume Injected (BBLs): {volume_injected}")
        #     print(f"Injection Pressure Average PSIG: {injection_pressure_avg_psig}")
        #     print(f"Injection Pressure Average PSI: {injection_pressure_avg_psi}")
        #     print(f"Well Total Depth (ft): {well_total_depth_ft}")
        #     print(f"Depth of Tubing Packer (ft): {depth_of_tubing_packer}")
        #     print(f"Date of Injection: {injection_date}\n")
        if volume_injected == 0:
            total_bottomhole_pressure = 0
            # print(f"------------------\nBottomhole Pressure: {total_bottomhole_pressure}\n------------------\n")
        else:
            if pd.isna(depth_of_tubing_packer):
                # print(f"API NUM: {api_number}\nInjected BBL: {volume_injected}\nInjection Date: {injection_date}\nPacker Depth: {well_total_depth_ft}")
                # deltaP = friction_loss(api_number, injection_date, volume_injected, well_total_depth_ft)  # in psi
                hydrostatic_pressure = float(0.465 * well_total_depth_ft)  # 0.465 psi/ft X depth (ft)
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure # - deltaP

            else:
                # deltaP = friction_loss(api_number, injection_date, volume_injected, well_total_depth_ft)  # in psi
                hydrostatic_pressure = float(0.465 * depth_of_tubing_packer)  # 0.465 psi/ft X depth (ft)
                total_bottomhole_pressure = float(injection_pressure_avg_psi) + hydrostatic_pressure # - deltaP

            # print(f"Injection Pressure: {injection_pressure_avg_psi}\n"
            #       f"Hydrostatic Pressure: {hydrostatic_pressure}\nDeltaP: {deltaP}")

        # print(f"\nBottomhole Pressure: {total_bottomhole_pressure}\n------------------\n")
        # Append the total_bottomhole_pressure value to the DataFrame as a new column
        cleaned_well_data_df.loc[index, 'Bottomhole Pressure'] = total_bottomhole_pressure

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


def plot_total_pressure(total_pressure_data, distance_data, earthquake_info, output_directory):
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
    deep_filename = os.path.join(output_directory, f'deep_well_pressure_data_{earthquake_info["Event ID"]}.txt')
    with open(deep_filename, 'w') as f:
        f.write("Date\tAPI Number\tPressure (PSI)\n")
        for date, pressure_points in deep_pressure_data.items():
            for api_number, pressure in pressure_points:
                f.write(f"{date}\t{api_number}\t{pressure}\n")

    # Save shallow well pressure data to a text file
    shallow_filename = os.path.join(output_directory,
                                    f'shallow_well_pressure_data_{earthquake_info["Event ID"]}.txt')
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
    shallow_colors = generate_gradient_colors(len(sorted_all_distances), "#00008B", "#ADD8E6", gamma=3.0)
    deep_colors = generate_gradient_colors(len(sorted_all_distances), "#8B0000", "#FFA07A", gamma=3.0)

    # Create a color map for all API numbers
    color_map_shallow = {api_number: color for (api_number, _), color in zip(sorted_all_distances, shallow_colors)}
    color_map_deep = {api_number: color for (api_number, _), color in zip(sorted_all_distances, deep_colors)}

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), sharex=True)

    # Plot shallow well data
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_pressure = {}  # Dictionary to store median pressure for each API number over a 3-day span

    for date, pressure_points in shallow_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure:
                api_median_pressure[api_number] = []
            api_median_pressure[api_number].append((date, median_pressure))

    for api_number, median_pressure_points in api_median_pressure.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_shallow[api_number])
        dates, pressures = zip(*median_pressure_points)
        ax1.plot(dates, pressures, marker='o', linestyle='', color=color_map_shallow[api_number])

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
                                                                              f'\nLocal Magnitude: {local_magnitude}'))

    ax1.set_title(f'event_{earthquake_info["Event ID"]} Total Pressure Data - Shallow Well')
    ax1.set_ylabel('Total Bottomhole Pressure (PSI)')
    ax1.grid(True)
    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax1.tick_params(axis='x', rotation=45)

    # Plot deep well data
    api_legend_map = {}  # Reset
    api_median_pressure = {}

    for date, pressure_points in deep_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure:
                api_median_pressure[api_number] = []
            api_median_pressure[api_number].append((date, median_pressure))

    for api_number, median_pressure_points in api_median_pressure.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, color_map_deep[api_number])
        dates, pressures = zip(*median_pressure_points)
        ax2.plot(dates, pressures, marker='o', linestyle='', color=color_map_deep[api_number])

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
                                                                              f'\nLocal Magnitude: {local_magnitude}'))

    ax2.set_title(f'event_{earthquake_info["Event ID"]} Total Pressure Data - Deep Well')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Bottomhole Pressure (PSI)')
    ax2.grid(True)
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax2.tick_params(axis='x', rotation=45)

    # Set major locator and formatter to display ticks for each month
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plot_filename = f'event_{earthquake_info["Event ID"]}_well_total_pressure_plot.png'
    plot_filepath = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    print(f"Pressure plots for earthquake: {earthquake_info['Event ID']} were successfully created.")


def create_well_histogram_per_api(cleaned_well_data_df):
    # Define the conditions and categories
    cleaned_well_data_df['Date of Injection'] = pd.to_datetime(cleaned_well_data_df['Date of Injection'],
                                                               errors='coerce')

    # Define the conditions and categories
    conditions = [
        (cleaned_well_data_df['Injection Pressure Average PSIG'].notna() & (
                cleaned_well_data_df['Injection Pressure Average PSIG'] != 0) &
         cleaned_well_data_df['Well Total Depth ft'].notna() & (cleaned_well_data_df['Well Total Depth ft'] != 0)),
        (cleaned_well_data_df['Well Total Depth ft'].notna() & (cleaned_well_data_df['Well Total Depth ft'] != 0) &
         (cleaned_well_data_df['Injection Pressure Average PSIG'].isna() | (
                 cleaned_well_data_df['Injection Pressure Average PSIG'] == 0))),
        ((cleaned_well_data_df['Injection Pressure Average PSIG'].isna() | (
                cleaned_well_data_df['Injection Pressure Average PSIG'] == 0)) &
         (cleaned_well_data_df['Well Total Depth ft'].isna() | (cleaned_well_data_df['Well Total Depth ft'] == 0)))
    ]
    categories = ['Complete', 'Incomplete', 'Missing']

    # Apply the conditions to create a new 'Category' column
    cleaned_well_data_df['Category'] = np.select(conditions, categories, default='Unknown')

    # Extract the month and year from 'Date of Injection'
    cleaned_well_data_df['Month-Year'] = cleaned_well_data_df['Date of Injection'].dt.to_period('M')

    # Sort the DataFrame by 'Date of Injection'
    cleaned_well_data_df.sort_values(by='Date of Injection', inplace=True)

    # Group by 'API Number'
    grouped = cleaned_well_data_df.groupby('API Number')

    for api_number, group in grouped:
        # Ensure chronological order within each 'Month-Year' by sorting by 'Date of Injection'
        group = group.sort_values(by='Date of Injection')

        # incomplete_dates = group[group['Category'] == 'Incomplete']['Date of Injection']
        # print(f"Incomplete dates for API Number {api_number}:")
        # print(incomplete_dates)

        # Count the occurrences in each category for each date within each month-year
        category_counts = group.groupby(['Month-Year', 'Date of Injection', 'Category']).size().unstack(
            fill_value=0).reindex(categories, axis=1, fill_value=0)

        # Reset index to have a flat DataFrame
        category_counts = category_counts.reset_index()

        # Create a MultiIndex with 'Month-Year' and 'Date of Injection'
        category_counts.set_index(['Month-Year', 'Date of Injection'], inplace=True)

        # Ensure that within each month, dates are sorted
        category_counts = category_counts.sort_index(level=['Month-Year', 'Date of Injection'])

        # Accumulate counts within each 'Month-Year'
        category_counts = category_counts.groupby(level=['Month-Year']).cumsum()

        # Aggregate monthly totals for the final histogram
        monthly_totals = category_counts.groupby(level='Month-Year').last()

        # Sort categories chronologically within each month
        monthly_totals = monthly_totals[sorted(categories)]

        # Plot the histogram
        monthly_totals.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title(f'Well Data for API Number {api_number}')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of Records')
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a file
        plot_filename = os.path.join(OUTPUT_DIR, f'well_data_histogram_{api_number}.png')
        plt.savefig(plot_filename)
        plt.close()


if len(sys.argv) > 1 and sys.argv[1] == '0':
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
    print(f"Center Lat: {type(earthquake_latitude)}\nLon: {type(earthquake_longitude)}\nRange: {type(range_km)}")
    closest_well_data_df = closest_wells_to_earthquake(center_lat=earthquake_latitude,
                                                       center_lon=earthquake_longitude,
                                                       radius_km=range_km)

    strawn_formation_data = pd.read_csv(STRAWN_FORMATION_DATA_FILE_PATH, delimiter=',')
    cleaned_well_data_df = data_preperation(closest_well_data_df, earthquake_latitude, earthquake_longitude,
                                            earthquake_origin_date, strawn_formation_data)
    # create_well_histogram_per_api(cleaned_well_data_df)
    finalized_df = calculate_total_bottomhole_pressure(cleaned_well_data_df=cleaned_well_data_df)
    # sample_rows = finalized_df.sample(n=5)  # Sample 5 rows
    total_pressure_data, distance_data = prepare_total_pressure_data_from_df(finalized_df)
    plot_total_pressure(total_pressure_data, distance_data, earthquake_info, OUTPUT_DIR)
    quit()
