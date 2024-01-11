import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, datetime
from accessAPIDepthOnline import get_api_depth

# Paths to datasets
injection_data_file_path = '/home/skevofilaxc/Downloads/injectiondata1624.csv'
earthquake_data_file_path = '/home/skevofilaxc/Downloads/texnet_events.csv'


# Calculates the pressure at the formation depth using the provided surface pressures
def bottomhole_pressure_calc(surface_pressure, well_depth):
    # Method provided by Jim Moore at RRCT that ignores friction loss in the tubing string
    # Bottomhole pressure = surface pressure + hydrostatic pressure
    hydrostatic_pressure = 0.465 * well_depth  # 0.465 psi/ft X depth (ft)
    return surface_pressure + hydrostatic_pressure


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
    with open(file_path, 'a') as file:
        # Write column headers if the file is empty
        if file.tell() == 0:
            file.write("Event ID, Latitude, Longitude, Origin Date, Origin Time, "
                       "Local Magnitude, Distance between Earthquakes (km), Time Lag between Earthquakes\n")

        # Write earthquake information
        file.write(f"{earthquake_info['Event ID']}, {earthquake_info['Latitude']}, "
                   f"{earthquake_info['Longitude']}, {earthquake_info['Origin Date']}, "
                   f"{earthquake_info['Origin Time']}, {earthquake_info['Local Magnitude']}, ")

        if current_earthquake_index == -1:
            file.write("N/A, N/A\n")  # No previous earthquake info, so distance and time lag are not applicable
        else:
            previous_earthquake_info = get_next_earthquake_info(extracted_and_sorted_earthquake_data,
                                                                current_earthquake_index)
            # Convert dates to datetime objects
            current_earthquake_datetime = datetime.strptime(
                f"{earthquake_info['Origin Date']} {earthquake_info['Origin Time']}", '%Y-%m-%d %H:%M:%S')
            previous_earthquake_datetime = datetime.strptime(
                f"{previous_earthquake_info['Origin Date']} {previous_earthquake_info['Origin Time']}", '%Y-%m-%d %H:%M:%S')
            # Calculate distance between earthquakes in km
            distance = haversine_distance(earthquake_info['Latitude'], earthquake_info['Longitude'],
                                          previous_earthquake_info['Latitude'], previous_earthquake_info['Longitude'])

            # Calculate time lag in days
            time_lag = abs((current_earthquake_datetime - previous_earthquake_datetime).days)

            file.write(f"{distance:.2f}, {time_lag}\n")

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


def convert_dates(some_earthquake_origin_date):
    """
    Converts some earthquake origin date and one year before said earthquake origin date to datetime objects for calcs
    """
    # Convert some_earthquake_origin_date to a datetime object
    some_earthquake_origin_date = datetime.strptime(some_earthquake_origin_date, '%Y-%m-%d')

    # Convert some_earthquake_origin_date to a datetime object to calculate 1 year before origin date
    one_year_before_earthquake_date = some_earthquake_origin_date - timedelta(days=365)

    return some_earthquake_origin_date, one_year_before_earthquake_date


def is_within_one_year(earliest_injection_date, one_year_before_earthquake_date, some_earthquake_origin_date):
    """
    Checks to see if the earliest well injection date falls within 1 year prior to the earthquake occurring.
    If the injection date falls within the 1 year range, great! Return True for further calculations
    If the injection date doesn't fall within the 1 year range, Return False and write to file
    """
    if earliest_injection_date >= one_year_before_earthquake_date:
        print("Injection date is not within 1 year prior to the earthquake date, will move onto next earthquake.")
        return False

    if earliest_injection_date <= some_earthquake_origin_date:
        print("Injection date is within the specified range.")
        return True


def process_matching_api_rows(matching_api_rows, one_year_before_earthquake_date, some_earthquake_origin_date,
                              i_th_earthquake_info):
    api_injection_data = {}

    for i in range(len(matching_api_rows)):
        index = matching_api_rows.index[i]  # Get the index of the first matching row
        injection_date = matching_api_rows.loc[index, 'Injection Date'].to_pydatetime()
        print(f"Injection Date: {injection_date}")

        if is_within_one_year(injection_date, one_year_before_earthquake_date, some_earthquake_origin_date):
            print("HERE!!!!")
            api_number = matching_api_rows.loc[index, 'API Number']
            api_injection_data[api_number] = matching_api_rows.loc[index]
            print(f"API Injection Data: {api_injection_data}")


def prechecking_injection_pressure(injection_data, topN_closest_wells, some_earthquake_origin_date,
                                   i_th_earthquake_info, current_earthquake_index):
    some_earthquake_origin_date, one_year_before_earthquake_date = convert_dates(some_earthquake_origin_date)

    for api_number, _ in topN_closest_wells:
        # Gets all the rows with matching api numbers from the well injection data set
        matching_api_rows = injection_data[injection_data['API Number'] == api_number]
        if not matching_api_rows.empty:
            earliest_injection_date_index = matching_api_rows.index[0]  # Get the index of the first matching row
            earliest_injection_date = matching_api_rows.loc[
                earliest_injection_date_index, 'Injection Date'].to_pydatetime()
            print(f"Injection Date: {earliest_injection_date}")
            if not is_within_one_year(earliest_injection_date, one_year_before_earthquake_date,
                                      some_earthquake_origin_date):
                # write the ith earthquake info to .txt file with the columns:
                # Event ID, Lat/Long, Origin Date/Time, Local Magnitude, Distance from 1 to 2,
                # and Total Average Surrounding Pressure
                write_earthquake_info_to_file('earthquake_info.txt', i_th_earthquake_info, current_earthquake_index - 1)
                return  # Exit the function and move on to the next earthquake

    # process_matching_api_rows(matching_api_rows, topN_closest_wells, one_year_before_earthquake_date,
    #                               some_earthquake_origin_date, i_th_earthquake_info)


# Extracting and displaying sorted earthquake data
extracted_and_sorted_earthquake_data = extract_and_sort_data(earthquake_data_file_path)
# Extracting and displaying well injection data
wells_data = extract_columns(injection_data_file_path)

if wells_data is not None and extracted_and_sorted_earthquake_data is not None:
    # Initialize current_earthquake_index
    current_earthquake_index = 0
    # Gets the information about the first earthquake (Event ID, Lat/Long, Origin Date/Time, Local Magnitude)
    for i in range(len(extracted_and_sorted_earthquake_data)):
        i_th_earthquake_info = get_next_earthquake_info(extracted_and_sorted_earthquake_data, i)
        print(f"Information about the current earthquake:")
        print(i_th_earthquake_info, "\n")

        # Extracting earthquake latitude and longitude
        earthquake_latitude = i_th_earthquake_info['Latitude']
        earthquake_longitude = i_th_earthquake_info['Longitude']
        earthquake_origin_date = i_th_earthquake_info['Origin Date']

        # Finding the top N closest wells to the earthquake within a range (20 km)
        top_closest_wells = find_closest_wells(wells_data, earthquake_latitude, earthquake_longitude, N=10, range_km=20)
        print(f"Top closest wells to the earthquake:\n{top_closest_wells}")

        prechecking_injection_pressure(wells_data, top_closest_wells, earthquake_origin_date, i_th_earthquake_info, i)
        current_earthquake_index += 1  # Increment the earthquake index
