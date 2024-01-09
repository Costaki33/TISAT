import pandas as pd

# Paths to datasets
injection_data_file_path = '/home/skevofilaxc/Downloads/injectiondata1624.csv'
earthquake_data_file_path = '/home/skevofilaxc/PycharmProjects/earthquake-analysis/texnet_events-Mentone-20231128.csv'


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
        first_earthquake_latitude = first_earthquake['Latitude (WGS84)']
        first_earthquake_longitude = first_earthquake['Longitude (WGS84)']
        first_earthquake_origin_date = first_earthquake['Origin Date']

        return {
            'Latitude': first_earthquake_latitude,
            'Longitude': first_earthquake_longitude,
            'Origin Date': first_earthquake_origin_date
        }
    except IndexError:
        print("No earthquake data available.")
        return None

# Extracting and displaying well injection data
extracted_injection_data = extract_columns(injection_data_file_path)
if extracted_injection_data is not None:
    pd.set_option('display.max_columns', None)
    print(extracted_injection_data.head())

# Extracting and displaying sorted earthquake data
extracted_and_sorted_earthquake_data = extract_and_sort_data(earthquake_data_file_path)
if extracted_and_sorted_earthquake_data is not None:
    print(extracted_and_sorted_earthquake_data.head())

    # Using the first_quake function to get information about the first earthquake
    first_quake_info = first_quake(extracted_and_sorted_earthquake_data)
    if first_quake_info is not None:
        print("Information about the first earthquake:")
        print(first_quake_info)
