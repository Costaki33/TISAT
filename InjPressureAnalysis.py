import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Paths to datasets
injection_data_file_path = '/home/skevofilaxc/Downloads/injectiondata1624.csv'
earthquake_data_file_path = '/home/skevofilaxc/Downloads/texnet_events.csv'


def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculate differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula to calculate distance
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance


def find_closest_wells(wells_data, earthquake_latitude, earthquake_longitude, N=10, range_km=20):
    distances = []
    for index, well in wells_data.iterrows():
        well_lat = well['Surface Latitude']
        well_lon = well['Surface Longitude']
        distance = haversine_distance(well_lat, well_lon, earthquake_latitude, earthquake_longitude)
        distances.append((index, distance))

    # Sort distances to get the top N closest wells
    distances.sort(key=lambda x: x[1])  # Sort based on distance (ascending order)
    closest_wells = distances[:N]  # Extract top N closest wells

    return closest_wells


def extract_and_sort_data(csv_file):
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
    columns_to_extract = [
        'UIC Number', 'Surface Longitude', 'Surface Latitude',
        'Injection Date', 'Injection End Date',
        'Injection Pressure Average PSIG', 'Injection Pressure Max PSIG'
    ]

    try:
        data = pd.read_csv(csv_file, usecols=columns_to_extract)
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

