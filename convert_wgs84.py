import pandas as pd
from pyproj import Transformer

# Define the transformer from NAD27 to WGS84
transformer = Transformer.from_crs("epsg:4267", "epsg:4326", always_xy=True)

# Read the original CSV file
strawn_formation_data_file_path = '/home/skevofilaxc/Documents/earthquake_data/TopStrawn_RD_GCSNAD27.csv'

df = pd.read_csv(strawn_formation_data_file_path)

# Convert the coordinates
df['lon_wgs84'], df['lat_wgs84'] = transformer.transform(df['lon_nad27'].values, df['lat_nad27'].values)

# Save the dataframe to a new CSV file
output_file = 'TopStrawn_RD_GCSWGS84.csv'
df.to_csv(output_file, index=False)

print(f"Converted coordinates saved to {output_file}")
