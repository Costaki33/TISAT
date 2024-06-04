import pandas as pd
import requests
import csv
import json
import io


def closest_wells_to_earthquake(center_lat, center_lon, radius_km):
    """
    Fetch and return a DataFrame of wells closest to earthquake data.

    Parameters:
    - apiUrl: str, API URL for data export.
    - begin_year: int, start year for data export.
    - begin_month: int, start month for data export.
    - end_year: int, end year for data export.
    - end_month: int, end month for data export.
    - center_lat: float, latitude of center of circular area of interest.
    - center_lon: float, longitude of center of circular area of interest.
    - radius_km: int, radius in kilometers of area of interest.

    Returns:
    - DataFrame containing the closest wells data.
    """

    apiUrl = "https://injection.texnet.beg.utexas.edu/api/Export"
    # Compose the export args dictionary
    exportArgs = {
        "Format": "excel",
        "BeginYear": "2016",
        "BeginMonth": "1",
        "EndYear": "2023",
        "EndMonth": "12",
        "aoiCircleCenterLatDD": center_lat,
        "aoiCircleCenterLonDD": center_lon,
        "aoiCircleRadiusKm": radius_km
    }

    headers = {
        "Content-type": "application/json",
        "Accept": "application/json"
    }

    request = requests.post(apiUrl, data=json.dumps(exportArgs), headers=headers, verify=False)

    if request.status_code == 200:
        responseContent = request.text

        # Use csv module to handle quoted fields properly
        csv_file = io.StringIO(responseContent)
        reader = csv.reader(csv_file)

        # Extract the header
        actual_columns = next(reader)
        # Read the data into a list of lists
        data = list(reader)

        # Create DataFrame
        df = pd.DataFrame(data, columns=actual_columns)

        # Convert columns to the correct data types
        df['API Number'] = pd.to_numeric(df['API Number'], errors='coerce')
        df['Well Total Depth ft'] = pd.to_numeric(df['Well Total Depth ft'], errors='coerce')
        df['Volume Injected (BBLs)'] = pd.to_numeric(df['Volume Injected (BBLs)'], errors='coerce')
        df['Injection Pressure Average PSIG'] = pd.to_numeric(df['Injection Pressure Average PSIG'], errors='coerce')
        df['Injection Pressure Max PSIG'] = pd.to_numeric(df['Injection Pressure Max PSIG'], errors='coerce')
        df['Completed Injection Interval Top'] = pd.to_numeric(df['Completed Injection Interval Top'], errors='coerce')
        df['Completed Injection Interval Bottom'] = pd.to_numeric(df['Completed Injection Interval Bottom'],
                                                                  errors='coerce')
        df['Surface Longitude'] = pd.to_numeric(df['Surface Longitude'], errors='coerce')
        df['Surface Latitude'] = pd.to_numeric(df['Surface Latitude'], errors='coerce')
        df['Annulus Pressure Num Readings'] = pd.to_numeric(df['Annulus Pressure Num Readings'], errors='coerce')
        df['Annulus Pressure Min PSIG'] = pd.to_numeric(df['Annulus Pressure Min PSIG'], errors='coerce')
        df['Annulus Pressure Max PSIG'] = pd.to_numeric(df['Annulus Pressure Max PSIG'], errors='coerce')
        df['Depth of Tubing Packer'] = pd.to_numeric(df['Depth of Tubing Packer'], errors='coerce')

        # Convert date columns to datetime
        df['Date of Injection'] = pd.to_datetime(df['Date of Injection'], errors='coerce')
        df['Injection End Date'] = pd.to_datetime(df['Injection End Date'], errors='coerce')
        df['Date Added'] = pd.to_datetime(df['Date Added'], errors='coerce')
        return df

    else:
        print("Oh no! We got a response code of: " + str(request.status_code))
        quit()


# Example usage:
# df = closest_wells_to_earthquake(center_lat=31.9418, center_lon=-102.3272, radius_km=20)
# print(df.iloc[0])
