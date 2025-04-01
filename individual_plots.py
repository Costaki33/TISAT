import os
import fnmatch
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter


def plot_injection_data(file_path, output_directory, csv_file, earthquake_info, well_type):
    try:
        # Read the data from the text file
        data = pd.read_csv(file_path, sep='\t')
        data.columns = data.columns.str.strip()

        # Read Injection Well Data CSV file
        well_info_df = pd.read_csv(csv_file)

        # Parse the earthquake event information
        origin_date_str = earthquake_info['Origin Date']
        origin_time = earthquake_info['Origin Time']
        local_magnitude = earthquake_info['Local Magnitude']
        origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
        origin_date_num = mdates.date2num(origin_date)

        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Determine the correct column for volume
        volume_column = 'Injection (BBLs)' if 'Injection (BBLs)' in data.columns else 'Monthly Volume (BBL)'

        # Get the unique API numbers
        api_numbers = data['API Number'].unique()
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] Creating Individual Injection Volume Plots for {len(api_numbers)} {well_type} Wells")

        # Create individual scatter plots for each API number
        for api_number in api_numbers:
            api_data = data[data['API Number'] == api_number]

            # Retrieve the distance from earthquake for the current API number
            # Check if 'API Number' or 'APINumber' exists in well_info_df
            if 'API Number' in well_info_df.columns:
                distance_row = well_info_df[well_info_df['API Number'] == api_number]
            elif 'APINumber' in well_info_df.columns:
                distance_row = well_info_df[well_info_df['APINumber'] == api_number]
            else:
                print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] No API column found in well_info_df.")
                distance_row = pd.DataFrame()  # Empty DataFrame if neither column exists
            if not distance_row.empty:
                distance = distance_row['Distance from Earthquake (km)'].values[0]
            else:
                distance = 'Unknown'

            plt.figure(figsize=(18, 10))
            plt.scatter(api_data['Date'], api_data[volume_column])
            plt.xlabel('Date')
            plt.ylabel(volume_column)
            plt.title(
                f'Injection Volumes Over Time for API Number {api_number} (Distance from Earthquake: {distance} km)', fontsize=14, fontweight='bold')
            plt.grid(True)

            # Set the x-axis major locator to MonthLocator
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

            # Rotate date labels for better readability
            plt.xticks(rotation=45)
            # Add a vertical dashed red line for the earthquake origin date
            ax.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)

            # Add the earthquake event to the legend
            legend_handles = [mlines.Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: '
                                                                                         f'{earthquake_info["Event ID"]}\n'
                                                                                         f'Origin Time: {origin_time}\n'
                                                                                         f'Origin Date: {origin_date_str}\n'
                                                                                         f'Local Magnitude: {local_magnitude}'),
                              mlines.Line2D([], [], color='blue', marker='o', linestyle='', label='Volume Injected')]
            # Add scatter plot handle to legend

            plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='medium', ncol=1)

            # Get current tick positions and labels
            # ticks = ax.get_xticks()
            # labels = [item.get_text() for item in ax.get_xticklabels()]

            # Adjust ticks and labels to increase spacing
            # new_ticks = ticks[::2]
            # new_labels = labels[::2]
            #
            # ax.set_xticks(new_ticks)
            # ax.set_xticklabels(new_labels)

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Save the plot as a PNG file in the specified output directory
            plot_file_path = os.path.join(output_directory, f'{api_number}_injection_vol_plot.png')
            plt.savefig(plot_file_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] Successfully Created {len(api_numbers)} {well_type} Injection Volume Well Plots. Plots are stored at {output_directory}")
    except Exception as e:
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] ERROR: An error occurred while processing {file_path}: {e}")


def plot_pressure_data(file_path, output_directory, csv_file, earthquake_info, well_type):
    try:
        # Read the data from the text file
        data = pd.read_csv(file_path, sep='\t')
        data.columns = data.columns.str.strip()

        # Read Injection Well Data CSV file
        well_info_df = pd.read_csv(csv_file)

        # Parse the earthquake event information
        origin_date_str = earthquake_info['Origin Date']
        origin_time = earthquake_info['Origin Time']
        local_magnitude = earthquake_info['Local Magnitude']
        origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
        origin_date_num = mdates.date2num(origin_date)

        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Determine the correct column for volume
        pressure_column = 'Average Pressure (PSIG)' if 'Average Pressure (PSIG)' in data.columns else 'InjectedPSIG'

        # Get the unique API numbers
        api_numbers = data['API Number'].unique()
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] Creating Individual Injection Pressure Plots for {len(api_numbers)} {well_type} Wells")
        # Create individual scatter plots for each API number
        for api_number in api_numbers:
            api_data = data[data['API Number'] == api_number]

            # Retrieve the distance from earthquake for the current API number
            # Check if 'API Number' or 'APINumber' exists in well_info_df
            if 'API Number' in well_info_df.columns:
                distance_row = well_info_df[well_info_df['API Number'] == api_number]
            elif 'APINumber' in well_info_df.columns:
                distance_row = well_info_df[well_info_df['APINumber'] == api_number]
            else:
                print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] No API column found in well_info_df.")
                distance_row = pd.DataFrame()  # Empty DataFrame if neither column exists
            if not distance_row.empty:
                distance = distance_row['Distance from Earthquake (km)'].values[0]
            else:
                distance = 'Unknown'

            plt.figure(figsize=(18, 10))
            plt.scatter(api_data['Date'], api_data[pressure_column])
            plt.xlabel('Date')
            plt.ylabel(pressure_column)
            plt.title(
                f'Average Injection Pressures Over Time for API Number {api_number} (Distance from Earthquake: {distance} km)', fontsize=14, fontweight='bold')
            plt.grid(True)

            # Set the x-axis major locator to MonthLocator
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

            # Rotate date labels for better readability
            plt.xticks(rotation=45)
            # Add a vertical dashed red line for the earthquake origin date
            ax.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)

            # Add the earthquake event to the legend
            legend_handles = [mlines.Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: '
                                                                                         f'{earthquake_info["Event ID"]}\n'
                                                                                         f'Origin Time: {origin_time}\n'
                                                                                         f'Origin Date: {origin_date_str}\n'
                                                                                         f'Local Magnitude: {local_magnitude}'),
                              mlines.Line2D([], [], color='blue', marker='o', linestyle='', label='Pressure Used during Injection Process')]
            # Add scatter plot handle to legend

            plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='medium', ncol=1)

            # Get current tick positions and labels
            # ticks = ax.get_xticks()
            # labels = [item.get_text() for item in ax.get_xticklabels()]

            # Adjust ticks and labels to increase spacing
            # new_ticks = ticks[::2]
            # new_labels = labels[::2]
            #
            # ax.set_xticks(new_ticks)
            # ax.set_xticklabels(new_labels)

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Save the plot as a PNG file in the specified output directory
            plot_file_path = os.path.join(output_directory, f'{api_number}_injection_pressure_plot.png')
            plt.savefig(plot_file_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] Successfully Created {len(api_numbers)} {well_type} Injection Pressure Well Plots. Plots are stored at {output_directory}")
    except Exception as e:
        print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] An error occurred while processing {file_path}: {e}")


def modify_string(abbreviation, range_):
    combined = f"{abbreviation}_{range_}"
    # Check if combined string is long enough
    if len(combined) >= 4:
        modified = combined[:-4] + combined[-2:]
    else:
        # Handle cases where the string is too short
        modified = combined
    return modified


def gather_well_data(base_path: str, csv_file: str, earthquake_info: dict):
    # Iterate through all combinations of abbreviations and ranges
    prefixes = [
        'shallow_well_inj_vol_data',
        'deep_well_inj_vol_data',
        'shallow_well_listed_pressure_data',
        'deep_well_listed_pressure_data'
    ]

    # Initialize a list to store the matching filenames
    matching_files = []

    # Iterate over the files in the specified directory
    for root, _, files in os.walk(base_path):
        for prefix in prefixes:
            for filename in fnmatch.filter(files, f"{prefix}*"):
                # Construct the full file path
                file_path = os.path.join(root, filename)
                matching_files.append(file_path)

    shallow_file_path = matching_files[0]
    deep_file_path = matching_files[1]
    shallow_pressure_fp = matching_files[2]
    deep_pressure_fp = matching_files[3]

    if os.path.isfile(shallow_file_path):
        shallow_output_directory = os.path.join(base_path, 'shallow_individual_plots')
        plot_injection_data(shallow_file_path, shallow_output_directory, csv_file, earthquake_info, "Shallow")
        plot_pressure_data(shallow_pressure_fp, shallow_output_directory, csv_file, earthquake_info, "Shallow")
    if os.path.isfile(deep_file_path):
        deep_output_directory = os.path.join(base_path, 'deep_individual_plots')
        plot_injection_data(deep_file_path, deep_output_directory, csv_file, earthquake_info, "Deep")
        plot_pressure_data(deep_pressure_fp, deep_output_directory, csv_file, earthquake_info, "Deep")
    print(f"[{datetime.datetime.now().replace(microsecond=0, second=0)}] Created individual well subplots")
