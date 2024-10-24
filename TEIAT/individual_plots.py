import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


# Function to plot injection data for each API number and save the plot
def plot_injection_data(file_path, output_directory):
    try:
        # Read the data from the text file
        data = pd.read_csv(file_path, sep='\t')
        data.columns = data.columns.str.strip()

        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Determine the correct column for volume
        volume_column = 'Injection (BBLs)' if 'Injection (BBLs)' in data.columns else 'Monthly Volume (BBL)'

        # Get the unique API numbers
        api_numbers = data['API Number'].unique()

        # Create individual scatter plots for each API number
        for api_number in api_numbers:
            api_data = data[data['API Number'] == api_number]

            plt.figure(figsize=(10, 6))
            plt.scatter(api_data['Date'], api_data[volume_column])
            plt.xlabel('Date')
            plt.ylabel(volume_column)
            plt.title(f'Injection Data for API Number {api_number}')
            plt.grid(True)

            # Set the x-axis major locator to MonthLocator
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

            # Rotate date labels for better readability
            plt.xticks(rotation=45)

            # Get current tick positions and labels
            ticks = ax.get_xticks()
            labels = [item.get_text() for item in ax.get_xticklabels()]

            # Adjust ticks and labels to increase spacing
            new_ticks = ticks[::2]
            new_labels = labels[::2]

            ax.set_xticks(new_ticks)
            ax.set_xticklabels(new_labels)

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Save the plot as a PNG file in the specified output directory
            plot_file_path = os.path.join(output_directory, f'{api_number}_injection_plot.png')
            plt.savefig(plot_file_path)
            plt.close()
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


def modify_string(abbreviation, range_):
    combined = f"{abbreviation}_{range_}"
    # Check if combined string is long enough
    if len(combined) >= 4:
        modified = combined[:-4] + combined[-2:]
    else:
        # Handle cases where the string is too short
        modified = combined
    return modified


def gather_well_data(base_path: str):
    # Iterate through all combinations of abbreviations and ranges
    prefixes = [
        'shallow_well_injection_data_texnet2024',
        'deep_well_injection_data_texnet2024'
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

    if os.path.isfile(shallow_file_path):
        shallow_output_directory = os.path.join(base_path, 'shallow_individual_plots')
        plot_injection_data(shallow_file_path, shallow_output_directory)

    if os.path.isfile(deep_file_path):
        deep_output_directory = os.path.join(base_path, 'deep_individual_plots')
        plot_injection_data(deep_file_path, deep_output_directory)

    print("Created individual well subplots")
