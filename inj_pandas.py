import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, datetime
from accessAPIDepthOnline import get_api_depth
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# GLOBAL VARIABLES AND FILE PATHS
injection_data_file_path = '/home/skevofilaxc/Downloads/updated_injectiondata1624.csv'
earthquake_data_file_path = '/home/skevofilaxc/Downloads/texnet_events.csv'
output_dir = '/home/skevofilaxc/Documents/earthquake_plots'
BACKTRACK_EARTHQUAKE_INDEX = 0  # if the get_starting_index is run, we'll set this var to the functions output


def get_starting_index(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Check if there are at least 3 lines in the file
        if len(lines) >= 3:
            # Get the index of the third-to-last line
            return len(lines) - 1
    except FileNotFoundError:
        pass  # If the file doesn't exist, return 0 as the default starting index
    return 0


def fetch_api_depth(api_numbers):
    with ProcessPoolExecutor() as executor:
        # Use executor.map to apply get_api_depth to each API number concurrently
        results = list(executor.map(get_api_depth, api_numbers))

    return dict(zip(api_numbers, results))


# Calculates the pressure at the formation depth using the provided surface pressures
def bottomhole_pressure_calc(surface_pressure, well_depth):
    # Method provided by Jim Moore at RRCT that ignores friction loss in the tubing string
    # Bottomhole pressure = surface pressure + hydrostatic pressure
    # Mud weight: JP Nicot, Jun Ge
    hydrostatic_pressure = float(0.465 * well_depth)  # 0.465 psi/ft X depth (ft)

    return float(surface_pressure) + hydrostatic_pressure


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


def write_earthquake_info_to_file(file_path, earthquake_info, current_earthquake_index):
    if not os.path.exists(file_path):
        # Create the file if it doesn't exist
        with open(file_path, 'w') as file:
            file.write("Event ID, Latitude, Longitude, Origin Date, Origin Time, Local Magnitude, Distance betwen "
                       "previous earthquake (KM), Time Lag (Days)\n")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Check if the earthquake information already exists in the file
    for i, line in enumerate(lines[1:], start=1):  # Skip the header line
        if earthquake_info['Event ID'] in line:
            # Update the existing line
            lines[i] = f"{earthquake_info['Event ID']}, {earthquake_info['Latitude']}, " \
                       f"{earthquake_info['Longitude']}, {earthquake_info['Origin Date']}, " \
                       f"{earthquake_info['Origin Time']}, {earthquake_info['Local Magnitude']}, "

            if current_earthquake_index == -1:
                lines[i] += "N/A, N/A\n"  # No previous earthquake info, so distance and time lag are not applicable
            else:
                previous_earthquake_info = get_next_earthquake_info(extracted_and_sorted_earthquake_data,
                                                                    current_earthquake_index)
                current_earthquake_datetime = datetime.strptime(
                    f"{earthquake_info['Origin Date']} {earthquake_info['Origin Time']}", '%Y-%m-%d %H:%M:%S')
                previous_earthquake_datetime = datetime.strptime(
                    f"{previous_earthquake_info['Origin Date']} {previous_earthquake_info['Origin Time']}",
                    '%Y-%m-%d %H:%M:%S')

                distance = haversine_distance(earthquake_info['Latitude'], earthquake_info['Longitude'],
                                              previous_earthquake_info['Latitude'],
                                              previous_earthquake_info['Longitude'])
                time_lag = abs((current_earthquake_datetime - previous_earthquake_datetime).days)

                lines[i] += f"{distance:.2f}, {time_lag}\n"

            # Write the updated lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
            return

    # If the earthquake information does not exist, append a new line
    with open(file_path, 'a') as file:
        if current_earthquake_index == 0:
            file.write(f"{earthquake_info['Event ID']}, {earthquake_info['Latitude']}, "
                       f"{earthquake_info['Longitude']}, {earthquake_info['Origin Date']}, "
                       f"{earthquake_info['Origin Time']}, {earthquake_info['Local Magnitude']}, N/A, N/A\n")
        else:
            previous_earthquake_info = get_next_earthquake_info(extracted_and_sorted_earthquake_data,
                                                                current_earthquake_index)
            current_earthquake_datetime = datetime.strptime(
                f"{earthquake_info['Origin Date']} {earthquake_info['Origin Time']}", '%Y-%m-%d %H:%M:%S')
            previous_earthquake_datetime = datetime.strptime(
                f"{previous_earthquake_info['Origin Date']} {previous_earthquake_info['Origin Time']}",
                '%Y-%m-%d %H:%M:%S')

            distance = haversine_distance(earthquake_info['Latitude'], earthquake_info['Longitude'],
                                          previous_earthquake_info['Latitude'], previous_earthquake_info['Longitude'])
            time_lag = abs((current_earthquake_datetime - previous_earthquake_datetime).days)

            file.write(f"{earthquake_info['Event ID']}, {earthquake_info['Latitude']}, "
                       f"{earthquake_info['Longitude']}, {earthquake_info['Origin Date']}, "
                       f"{earthquake_info['Origin Time']}, {earthquake_info['Local Magnitude']}, "
                       f"{distance:.2f}, {time_lag}\n")

    # Periodically close and reopen the file to flush the content
    if current_earthquake_index % 1 == 0:
        with open(file_path, 'a'):
            pass


def find_closest_wells(wells_data, earthquake_latitude, earthquake_longitude, N=10, range_km=20):
    """
    Finds the closest N wells to a given earthquake within some range km and returns the UIC num and distance (km) of
    the earthquake from the well
    """
    closest_wells_dict = {}
    for index, well in wells_data.iterrows():
        well_lat = well['Surface Latitude']
        well_lon = well['Surface Longitude']
        distance = haversine_distance(well_lat, well_lon, earthquake_latitude, earthquake_longitude)

        api_number = well['API Number']
        if distance <= range_km:  # Consider only wells within the specified range
            if api_number not in closest_wells_dict or distance < closest_wells_dict[api_number][1]:
                closest_wells_dict[api_number] = (api_number, distance)

    # Sort distances to get the top N closest wells
    closest_wells = sorted(closest_wells_dict.values(), key=lambda x: x[1])[:N]

    return closest_wells


def extract_and_sort_data(csv_file):
    """
    Extracts the earthquake event data from its .csv file and sorts by time of occurrence (first to latest)
    """
    columns_to_extract = [
        'EventID', 'Origin Date', 'Origin Time',
        'Local Magnitude', 'Latitude (WGS84)', 'Longitude (WGS84)'
    ]

    try:
        # Read the CSV file and extract specific columns
        data = pd.read_csv(csv_file, usecols=columns_to_extract)

        # Convert 'Origin Date' and 'Origin Time' to a combined datetime column
        data['DateTime'] = pd.to_datetime(data['Origin Date'] + ' ' + data['Origin Time'])

        # Sort the data by 'DateTime' in descending order
        sorted_data = data.sort_values(by='DateTime', ascending=True)

        # Drop the combined 'DateTime' column after sorting
        sorted_data.drop(columns=['DateTime'], axis=1, inplace=True)

        return sorted_data
    except FileNotFoundError:
        print("File not found.")
        return None


def extract_columns(csv_file):
    """
    Extracts specified columns from the injection well data file
    """
    columns_to_extract = [
        'API Number', 'Surface Longitude', 'Surface Latitude',
        'Injection Date', 'Injection End Date',
        'Injection Pressure Average PSIG'
    ]

    try:
        data = pd.read_csv(csv_file, usecols=columns_to_extract)

        # Convert 'Injection Date' and 'Injection End Date' to datetime format
        data['Injection Date'] = pd.to_datetime(data['Injection Date'])
        data['Injection End Date'] = pd.to_datetime(data['Injection End Date'])
        data['API Number'] = data['API Number'].astype(int)  # Convert API numbers to integers

        return data
    except FileNotFoundError:
        print("File not found.")
        return None


# Function to extract information about the current earthquake and increment the index
def get_next_earthquake_info(data_frame, current_earthquake_index):
    """
    Extracts information about the current earthquake and increments the index.
    """

    try:
        earthquake = data_frame.iloc[current_earthquake_index]  # Extracting the current earthquake
        earthquake_eventid = earthquake['EventID']
        earthquake_latitude = earthquake['Latitude (WGS84)']
        earthquake_longitude = earthquake['Longitude (WGS84)']
        earthquake_origin_date = earthquake['Origin Date']
        earthquake_origin_time = earthquake['Origin Time']
        earthquake_local_magnitude = earthquake['Local Magnitude']

        return {
            'Event ID': earthquake_eventid,
            'Latitude': earthquake_latitude,
            'Longitude': earthquake_longitude,
            'Origin Date': earthquake_origin_date,
            'Origin Time': earthquake_origin_time,
            'Local Magnitude': earthquake_local_magnitude
        }
    except IndexError:
        print(f"No more earthquake data available.")
        return None


def convert_tuple_dates(unconverted_tuple_dates):
    """
    Function turns tuple dates into normal dates for plotting
    :param date_tuple:
    :return:
    """
    # Convert the tuple of dates to a list of datetime objects
    converted_dates = [date_tuple[0] for date_tuple in unconverted_tuple_dates]
    return converted_dates


def convert_dates(some_earthquake_origin_date):
    """
    Converts some earthquake origin date and one year before said earthquake origin date to datetime objects for calcs
    """
    # Convert some_earthquake_origin_date to a datetime object
    some_earthquake_origin_date = datetime.strptime(some_earthquake_origin_date, '%Y-%m-%d')

    # Convert some_earthquake_origin_date to a datetime object to calculate 1 year before origin date
    one_year_after_earthquake_date = some_earthquake_origin_date + timedelta(days=365)

    return some_earthquake_origin_date, one_year_after_earthquake_date


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
        print("Injection date is within the specified range.")
        return True


def process_matching_api_rows(matching_api_rows, one_year_after_earthquake_date, some_earthquake_origin_date):
    # Fetch API depths for all unique API numbers in matching_api_rows
    unique_api_numbers = matching_api_rows['API Number'].astype(str).unique()
    api_depths = fetch_api_depth(unique_api_numbers)

    total_pressure_per_date = defaultdict(float)

    for _, row in matching_api_rows.iterrows():
        injection_date = row['Injection Date'].to_pydatetime()

        if is_within_one_year(injection_date, one_year_after_earthquake_date):
            api_number = str(row['API Number'])

            # Check if the length of api_number is 8 before proceeding
            if len(api_number) == 8:
                average_psig = row['Injection Pressure Average PSIG']

                # Use the fetched API depth directly from the dictionary
                api_depth_ft = api_depths.get(api_number)

                if api_depth_ft is not None:
                    api_depth_ft = float(api_depth_ft)
                    bottomhole_pressure = bottomhole_pressure_calc(average_psig, api_depth_ft)
                    total_pressure_per_date[injection_date] += bottomhole_pressure
                else:
                    print(f"Warning: API Depth for API {api_number} is not available.")
                    # Handle the case where api_depth_ft is not available

            else:
                print(f"Warning: Skipping API {api_number} due to invalid length (not equal to 8).")

    return total_pressure_per_date


def prechecking_injection_pressure(injection_data, topN_closest_wells, some_earthquake_origin_date,
                                   i_th_earthquake_info, current_earthquake_index):
    some_earthquake_origin_date, one_year_after_earthquake_date = convert_dates(some_earthquake_origin_date)

    total_pressure_data = {}

    has_good_injection_data = False

    for api_number, _ in topN_closest_wells:
        matching_api_rows = injection_data[injection_data['API Number'] == api_number]

        if not matching_api_rows.empty:
            earliest_injection_date = matching_api_rows['Injection Date'].min().to_pydatetime()

            if not is_within_one_year(earliest_injection_date, one_year_after_earthquake_date):
                print("Injection date is not within 1 year of the earthquake date, will move onto next earthquake.\n"
                      "------------------------------------")
                write_earthquake_info_to_file('earthquake_info.txt', i_th_earthquake_info, current_earthquake_index - 1)
                break

            total_pressure_data[api_number] = process_matching_api_rows(
                matching_api_rows, one_year_after_earthquake_date, some_earthquake_origin_date)
            has_good_injection_data = True

    if has_good_injection_data:
        plot_total_pressure(total_pressure_data, i_th_earthquake_info, output_dir)
        write_earthquake_info_to_file('earthquake_info.txt', i_th_earthquake_info, current_earthquake_index)
        print("Moving onto next earthquake.\n------------------------------------")
        return  # Exit the function and move on to the next earthquake


def plot_total_pressure(total_pressure_data, earthquake_info, output_directory):
    # Plot the total pressure for each API with legend
    plt.figure(figsize=(20, 8))

    # Create a defaultdict to store the total pressure for each date
    total_pressure_by_date = defaultdict(float)
    all_api_nums = []  # list to store all the api numbers for plot label

    # print(f"Total pressure data: \n{total_pressure_data}")
    # print(f"Earthquake info: {earthquake_info}")

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
            total_pressure_by_date[date] += total_pressure

    dates, total_pressure_values = zip(*total_pressure_by_date.items())

    # Sort the dates and corresponding pressures by date
    sorted_data = sorted(zip(dates, total_pressure_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_total_pressure_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    plt.plot(date_strings, sorted_total_pressure_values, marker='o', label=',\n'.join([f'API {num}' for num in all_api_nums]))
    # plt.xticks(rotation=45, ha='right', fontsize=8)  # Adjust the rotation angle as needed
    plt.xticks(np.arange(0, len(date_strings), step=15), rotation=45, ha='right', fontsize=8)

    # Extract earthquake information
    origin_date = datetime.strptime(earthquake_info['Origin Date'], '%Y-%m-%d')
    # Convert origin_date to a numerical representation
    origin_date_num = mdates.date2num(origin_date)
    # Convert origin_date_num to a string for consistency
    origin_date_str = mdates.num2date(origin_date_num).strftime('%Y-%m-%d')

    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']

    plt.axvline(x=origin_date_str, color='red', linestyle='--', label=f'{earthquake_info["Event ID"]}'
                                                                      f'\nOrigin Time: {origin_time}'
                                                                      f'\nOrigin Date: {origin_date}'
                                                                      f'\nLocal Magnitude: {local_magnitude}')

    plt.xlabel('Injection Date')
    plt.ylabel('Total Bottomhole Pressure')
    plt.title(f'Total Bottomhole Pressure Over Time - Event ID: {earthquake_info["Event ID"]}')
    plt.grid(True)
    plt.legend()

    # Save the plot to a file in the specified output directory
    plot_filename = f'event_{earthquake_info["Event ID"]}_total_pressure_plot.png'
    plot_filepath = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_filepath)

    # Close the plot after saving
    plt.close()
    print(f"Pressure plot for earthquake: {earthquake_info['Event ID']} was successfully created.")


# Extracting and displaying sorted earthquake data
extracted_and_sorted_earthquake_data = extract_and_sort_data(earthquake_data_file_path)
# Extracting and displaying well injection data
wells_data = extract_columns(injection_data_file_path)

if wells_data is not None and extracted_and_sorted_earthquake_data is not None:
    # Initialize current_earthquake_index
    current_earthquake_index = 0
    BACKTRACK_EARTHQUAKE_INDEX = get_starting_index('earthquake_info.txt')

    for i in range(len(extracted_and_sorted_earthquake_data)):
        if BACKTRACK_EARTHQUAKE_INDEX is not None and i < BACKTRACK_EARTHQUAKE_INDEX:
            continue

        i_th_earthquake_info = get_next_earthquake_info(extracted_and_sorted_earthquake_data, i)
        print(f"Information about the current earthquake:")
        print(i_th_earthquake_info, "\n")

        earthquake_latitude = i_th_earthquake_info['Latitude']
        earthquake_longitude = i_th_earthquake_info['Longitude']
        earthquake_origin_date = i_th_earthquake_info['Origin Date']

        top_closest_wells = find_closest_wells(wells_data, earthquake_latitude, earthquake_longitude, N=10, range_km=20)

        prechecking_injection_pressure(wells_data, top_closest_wells, earthquake_origin_date, i_th_earthquake_info, i)
        current_earthquake_index += 1
