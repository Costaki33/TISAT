import pandas as pd

# Paths to datasets
injection_data_file_path = r'C:\Users\costa\OneDrive\Documents\BEG\Wells With Injection Data 1.4.24\UICN000012239.csv'
earthquake_data_file_path = r'C:\Users\costa\PycharmProjects\BEG\take2\texnet_events-Mentone-20231128.csv'


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


# Extracting and displaying well injection data
extracted_injection_data = extract_columns(injection_data_file_path)
if extracted_injection_data is not None:
    pd.set_option('display.max_columns', None)
    print(extracted_injection_data.head())

# Extracting and displaying sorted earthquake data
extracted_and_sorted_earthquake_data = extract_and_sort_data(earthquake_data_file_path)
if extracted_and_sorted_earthquake_data is not None:
    print(extracted_and_sorted_earthquake_data.head())
