import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, datetime
from accessAPIDepthOnline import get_api_depth

# Paths to datasets
injection_data_file_path = '/home/skevofilaxc/Downloads/injectiondata1624.csv'
earthquake_data_file_path = '/home/skevofilaxc/Downloads/texnet_events.csv'


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
    Extracts the earthquake event data from its .csv file and sorts by time of occurrence
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


def first_quake(data_frame):
    """
    Extracts information about the first earthquake in the given DataFrame.
    """
    try:
        first_earthquake = data_frame.iloc[0]  # Extracting the first earthquake
        first_earthquake_eventid = first_earthquake['EventID']
        first_earthquake_latitude = first_earthquake['Latitude (WGS84)']
        first_earthquake_longitude = first_earthquake['Longitude (WGS84)']
        first_earthquake_origin_date = first_earthquake['Origin Date']

        return {
            'Event ID': first_earthquake_eventid,
            'Latitude': first_earthquake_latitude,
            'Longitude': first_earthquake_longitude,
            'Origin Date': first_earthquake_origin_date
        }
    except IndexError:
        print("No earthquake data available.")
        return None


# Function to query and plot average injection pressure over time for specific UIC numbers
def plot_injection_pressure(injection_data, topN_closest_wells, some_earthquake_origin_date):
    # Convert some_earthquake_origin_date to a datetime object to calculate 6 months before origin date
    some_earthquake_origin_date = datetime.strptime(some_earthquake_origin_date, '%Y-%m-%d')
    # print(f"Some earthquake: {some_earthquake_origin_date}, type: {type(some_earthquake_origin_date)}")
    api_injection_data = {}
    for api_number, _ in topN_closest_wells:
        matching_api_rows = injection_data[injection_data['API Number'] == api_number]
        # print(f"Matching Row: {matching_api_rows}")
        if not matching_api_rows.empty:
            for i in range(len(matching_api_rows)):
                index = matching_api_rows.index[i]  # Get the index of the first matching row
                injection_date = matching_api_rows.loc[index, 'Injection Date'].to_pydatetime()
                # print(f"Injection Date: {type(injection_date)}")

                if injection_date <= some_earthquake_origin_date:
                    print("HERE!!!!")
                    api_injection_data[api_number] = matching_api_rows.loc[index]
                    print(f"API Injection Data: {api_injection_data}")


# Extracting and displaying sorted earthquake data
extracted_and_sorted_earthquake_data = extract_and_sort_data(earthquake_data_file_path)
# Extracting and displaying well injection data
wells_data = extract_columns(injection_data_file_path)

if wells_data is not None and extracted_and_sorted_earthquake_data is not None:
    # Using the first_quake function to get information about the first earthquake
    first_quake_info = first_quake(extracted_and_sorted_earthquake_data)
    if first_quake_info is not None:
        print("Information about the first earthquake:")
        print(first_quake_info)

        # Extracting earthquake latitude and longitude
        earthquake_latitude = first_quake_info['Latitude']
        earthquake_longitude = first_quake_info['Longitude']

        # Finding the top N the closest wells to the earthquake within a range (20 km)
        top_closest_wells = find_closest_wells(wells_data, earthquake_latitude, earthquake_longitude, N=10, range_km=20)
        print(f"Top closest wells to the earthquake:\n{top_closest_wells}")

        # Usage:
        earthquake_origin_date = first_quake_info['Origin Date']
        plot_injection_pressure(wells_data, top_closest_wells, earthquake_origin_date)
