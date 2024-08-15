import os
import csv
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from collections import defaultdict
from friction_loss_calc import friction_loss

def classify_well_type(well_lat, well_lon, well_depth, strawn_formation_data, cache={}):
    """
    Function to classify well type between Shallow or Deep based on the Z-depth of the well
    compared to the closest position of the Strawn Formation.
    """
    # Check if the well has already been classified
    cache_key = (well_lat, well_lon, well_depth)
    if cache_key in cache:
        return cache[cache_key]

    # Extract latitudes, longitudes, and depths from the dataset
    dataset_lats = strawn_formation_data['lat_wgs84'].values
    dataset_lons = strawn_formation_data['lon_wgs84'].values
    dataset_depths = strawn_formation_data['Zft_sstvd'].values

    # Vectorized Haversine calculation
    lat1_rad, lon1_rad = np.radians(well_lat), np.radians(well_lon)
    lat2_rad, lon2_rad = np.radians(dataset_lats), np.radians(dataset_lons)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of Earth in kilometers
    R = 6371.0
    distances = R * c

    # Find the index of the closest formation point
    closest_index = np.argmin(distances)

    # Compare well depth to the closest formation depth
    closest_strawn_depth = dataset_depths[closest_index]
    classification = 1 if abs(well_depth) + closest_strawn_depth > 0 else 0

    # Cache the result
    cache[cache_key] = classification

    return classification


def haversine_distance_series(lat1_series, lon1_series, lat2_series, lon2_series):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1_series)
    lon1_rad = np.radians(lon1_series)
    lat2_rad = np.radians(lat2_series)
    lon2_rad = np.radians(lon2_series)

    # Compute differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of Earth in kilometers
    R = 6371.0
    distance = R * c

    return distance


def clean_csv(b3csv: csv, earthquake_information: dict, strawn_formation_info: str):
    b3df = pd.read_csv(b3csv)
    strawn_formation_data = pd.read_csv(strawn_formation_info, delimiter=',')
    earthquake_lat = earthquake_information['Latitude']
    earthquake_lon = earthquake_information['Longitude']
    well_lat = b3df['SurfaceHoleLatitude']
    well_lon = b3df['SurfaceHoleLongitude']

    # Fix APINumber: Remove all "-" and first two characters to have matching values with IVRT
    b3df['APINumber'] = b3df['APINumber'].str.replace("-", "")
    b3df['APINumber'] = b3df['APINumber'].str[2:]

    distance = round(haversine_distance_series(earthquake_lat, earthquake_lon, well_lat, well_lon), 2)
    b3df['Distance from Earthquake (km)'] = distance

    # Fix Date: Convert provided date information into datetime for plotting
    b3df['StartOfMonthDate'] = pd.to_datetime(b3df['StartOfMonthDate']).dt.normalize()

    well_types_map = {}
    for index, row in b3df.iterrows():
        api_number = row['APINumber']
        well_lat = row['SurfaceHoleLatitude']
        well_lon = row['SurfaceHoleLongitude']
        well_depth = row['MeasuredDepthFt']
        well_types_map[api_number] = classify_well_type(well_lat, well_lon, well_depth, strawn_formation_data)

    b3df['Well Type'] = b3df['APINumber'].map(well_types_map)

    depth_classification_map = {}
    for api_number, group in b3df.groupby('APINumber'):
        classifications = group['PermittedWellDepthClassification'].dropna().unique()
        if len(classifications) == 0:
            # If all entries are blank, use the value from 'Well Type'
            calculated_classification = 'deep' if group['Well Type'].iloc[0] == 1 else 'shallow'
        elif any("Shallow" in classification for classification in classifications) and any(
                "Deep" in classification for classification in classifications):
            calculated_classification = 'both'
        elif any("Shallow" in classification for classification in classifications):
            calculated_classification = 'shallow'
        elif any("Deep" in classification for classification in classifications):
            calculated_classification = 'deep'
        else:
            # Fallback if something unexpected happens
            calculated_classification = 'deep' if group['Well Type'].iloc[0] == 1 else 'shallow'

        depth_classification_map[api_number] = calculated_classification

    b3df['CalculatedPermittedDepthClassification'] = b3df['APINumber'].map(depth_classification_map)

    return b3df


def calculate_b3_total_bh_pressure(cleaned_b3df: pd.DataFrame):
    deltaP = 0
    for index, row in cleaned_b3df.iterrows():
        api_number = row['APINumber']
        volume_injected_month = row['InjectedLiquidBBL']
        inj_pressure_avg_psi = row['InjectedPSIG'] + 14.7
        injection_date = row['StartOfMonthDate']

        """
        Important to note that there is a lack of information regarding the exact value for the length of tubing from
        b3 data; Will utilize the closest depth from the friction_loss_calc()
        """
        depth_of_wellbore_ft = row['MeasuredDepthFt']  # depth of the wellbore, which includes casing (ft)

        hydrostatic_pressure = float(0.465 * depth_of_wellbore_ft)  # 0.465 psi/ft X depth (ft)
        if volume_injected_month == 0:
            total_bottomhole_pressure = float(inj_pressure_avg_psi) + hydrostatic_pressure
        elif volume_injected_month > 0:
            deltaP = friction_loss(api_number, injection_date, volume_injected_month, depth_of_wellbore_ft,
                                   b3=1)  # in psi
            total_bottomhole_pressure = float(inj_pressure_avg_psi) + hydrostatic_pressure - deltaP

        cleaned_b3df.loc[index, 'Bottomhole Pressure'] = total_bottomhole_pressure
        cleaned_b3df.loc[index, 'deltaP'] = deltaP

    return cleaned_b3df


def b3_data_quality_histogram(cleaned_df, range_km, output_directory=None):
    categories = [
        'Both Volume Injected and Pressure Provided',
        'Only Volume Injected Provided',
        'Neither Value Provided'
    ]

    conditions = [
        (cleaned_df['InjectedLiquidBBL'].notna() & (cleaned_df['InjectedLiquidBBL'] != 0) &
         cleaned_df['InjectedPSIG'].notna() & (cleaned_df['InjectedPSIG'] != 0)),

        (cleaned_df['InjectedLiquidBBL'].notna() & (cleaned_df['InjectedLiquidBBL'] != 0) &
         (cleaned_df['InjectedPSIG'].isna() | (cleaned_df['InjectedPSIG'] == 0))),

        ((cleaned_df['InjectedLiquidBBL'].isna() | (cleaned_df['InjectedLiquidBBL'] == 0)) &
         (cleaned_df['InjectedPSIG'].isna() | (cleaned_df['InjectedPSIG'] == 0)))
    ]

    cleaned_df['Category'] = np.select(conditions, categories, default='Unknown')
    cleaned_df.sort_values(by='StartOfMonthDate', inplace=True)
    grouped = cleaned_df.groupby('APINumber')
    histograms = {}

    for api_number, group in grouped:
        group = group.sort_values(by='StartOfMonthDate')
        category_counts = group['Category'].value_counts().reindex(categories, fill_value=0)
        total_counts = category_counts.sum()
        percentages = (category_counts / total_counts) * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(categories, category_counts, color=['blue', 'orange', 'green'])

        for bar, count, percent in zip(bars, category_counts, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{count} ({percent:.2f}%)', ha='center', va='bottom')

        ax.set_title(
            f'Category Distribution for API #{api_number} (Total Records: {total_counts}) ({range_km} KM Range)')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')

        legend_labels = [f'{category}: {category_counts[category]} records, {percentages[category]:.2f}%'
                         for category in categories]

        ax.legend(bars, legend_labels, title='Category', loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()

        histograms[api_number] = fig

        # Save the plot to a file
        if output_directory:
            plot_filename = os.path.join(output_directory,
                                         f'well_data_histogram_{api_number}_range{range_km}km.png')
            plt.savefig(plot_filename)
            plt.close()


def plot_b3_bhp(cleandf, earthquake_information, output_dir, range_km):
    deep_pressure_data = defaultdict(list)
    shallow_pressure_data = defaultdict(list)
    eventID = earthquake_information['Event ID']
    origin_date_str = earthquake_information['Origin Date']
    origin_time = earthquake_information['Origin Time']
    local_magnitude = earthquake_information['Local Magnitude']
    origin_date = pd.to_datetime(origin_date_str)

    # Classify API numbers and prepare pressure data
    api_color_map = {}
    for _, row in cleandf.iterrows():
        api_number = row['APINumber']
        pressure = row['Bottomhole Pressure']
        date = row['StartOfMonthDate']
        depth_class = row['CalculatedPermittedDepthClassification']
        distance_from_eq = row['Distance from Earthquake (km)']

        label_text = f"API {api_number} ({distance_from_eq} km)"

        if depth_class == 'shallow':
            color = 'green'
            shallow_pressure_data[api_number].append((date, pressure, label_text, color))
        elif depth_class == 'deep':
            color = 'blue'
            deep_pressure_data[api_number].append((date, pressure, label_text, color))
        elif depth_class == 'both':
            color = 'purple'
            deep_pressure_data[api_number].append((date, pressure, label_text, color))

        api_color_map[api_number] = color

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)

    # Plot for shallow well data
    shallow_scatter_colors = {}
    for api_number, pressure_points in shallow_pressure_data.items():
        dates, pressures, labels, colors = zip(*pressure_points)
        scatter = ax1.scatter(dates, pressures, label=labels[0], s=12)
        shallow_scatter_colors[api_number] = scatter.get_edgecolor()
    ax1.set_title(f'{eventID} Bottomhole Pressure - Shallow Well ({range_km} KM Range)')
    ax1.set_ylabel('Bottomhole Pressure (PSI)')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=2))

    # Plot for deep well data
    deep_scatter_colors = {}
    for api_number, pressure_points in deep_pressure_data.items():
        dates, pressures, labels, colors = zip(*pressure_points)
        scatter = ax2.scatter(dates, pressures, label=labels[0], s=12)
        deep_scatter_colors[api_number] = scatter.get_edgecolor()
    ax2.set_title(f'{eventID} Bottomhole Pressure - Deep Well ({range_km} KM Range)')
    ax2.set_ylabel('Bottomhole Pressure (PSI)')

    ax1.axvline(x=origin_date, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=origin_date, color='red', linestyle='--', linewidth=2)

    # Combine legends and change text color
    legend_info_shallow = []
    legend_info_deep = []

    for api_number, pressure_points in shallow_pressure_data.items():
        _, _, label, _ = pressure_points[0]
        color = shallow_scatter_colors.get(api_number, 'black')
        distance_from_eq = float(label.split('(')[-1].split()[0])  # Extract distance from label
        legend_info_shallow.append((distance_from_eq, Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                             label=label, markersize=10)))

    for api_number, pressure_points in deep_pressure_data.items():
        _, _, label, _ = pressure_points[0]
        color = deep_scatter_colors.get(api_number, 'black')
        distance_from_eq = float(label.split('(')[-1].split()[0])  # Extract distance from label
        legend_info_deep.append((distance_from_eq, Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                          label=label, markersize=10)))

    # Custom legend for B3 Well Classification
    classification_legend_handles = [
        Line2D([0], [0], color='green', linestyle='-', label=f'B3 Well Classification Key\nShallow Well'),
        Line2D([0], [0], color='blue', linestyle='-', label='Deep Well'),
        Line2D([0], [0], color='purple', linestyle='-', label='Shallow & Deep')
    ]

    # Sort shallow and deep earthquakes by distance from earthquake
    legend_info_shallow.sort(key=lambda x: x[0])
    legend_info_deep.sort(key=lambda x: x[0])

    custom_legend_handles_shallow = [handle for _, handle in legend_info_shallow]
    custom_legend_handles_deep = [handle for _, handle in legend_info_deep]

    # Update legends for both subplots with legend for earthquake
    legend_handles_shallow = [
        Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_information["Event ID"]}'
                                                            f'\nOrigin Time: {origin_time}'
                                                            f'\nOrigin Date: {origin_date_str}'
                                                            f'\nLocal Magnitude: {local_magnitude}'
                                                            f'\nRange: {range_km} km')]

    # Add all 3 custom legends to ax1
    ax1.legend(handles=classification_legend_handles + custom_legend_handles_shallow + legend_handles_shallow,
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

    legend_handles_deep = [Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_information["Event ID"]}'
                                                                               f'\nOrigin Time: {origin_time}'
                                                                               f'\nOrigin Date: {origin_date_str}'
                                                                               f'\nLocal Magnitude: {local_magnitude}'
                                                                               f'\nRange: {range_km} km')]

    # Add all 3 custom legends to ax2
    ax2.legend(handles=custom_legend_handles_deep + legend_handles_deep + classification_legend_handles,
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 1])

    legends_list = [ax1.get_legend(), ax2.get_legend()]
    for legend in legends_list:
        if legend is not None:
            for text in legend.get_texts():
                text_str = text.get_text()
                split_text = text_str.split(' ')

                # Ensure the split_text has at least 2 elements before accessing
                if len(split_text) > 1:
                    api_number = split_text[1]

                    if api_number in api_color_map:
                        text_color = api_color_map[api_number]
                        plt.setp(text, color=text_color)
                else:
                    # Handle cases where the text doesn't contain an API number
                    # For example, if it's a classification label like 'Shallow', 'Deep', 'Both'
                    continue

    # Set x-axis to show ticks for each month
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save the plot to a file
    plot_filename = os.path.join(output_dir,
                                 f'pressure_over_time_{earthquake_information["Event ID"]}_range{range_km}km.png')
    plt.savefig(plot_filename)

    # Save deep well pressure data to a text file
    deep_filename = os.path.join(output_dir,
                                 f'deep_well_pressure_data_{earthquake_information["Event ID"]}_range{range_km}km.txt')
    with open(deep_filename, 'w') as f:
        f.write(f"{'Date':<20}\t{'API Number':<12}\t{'Bottomhole Pressure (PSI)':<25}"
                f"\t{'Depth Classification (B3)':<25}\t{'Well Type (TXNET)':<30}\n")
        for api_number, pressure_points in shallow_pressure_data.items():
            for date, pressure, _, color in pressure_points:
                color_str = str(color) if isinstance(color, np.ndarray) else color
                well_str = {"green": "Shallow", "blue": "Deep", "purple": "Both"}.get(color_str, "")
                f.write(
                    f"{str(date):<20}\t{str(api_number):<12}\t{str(pressure):<25}\t{well_str:<25}"
                    f"\t{'Deep - Strawn Formation':<30}\n")

    # Save shallow well pressure data to a text file
    shallow_filename = os.path.join(output_dir,
                                    f'shallow_well_pressure_data_{earthquake_information["Event ID"]}_range{range_km}km.txt')
    with open(shallow_filename, 'w') as f:
        f.write(f"{'Date':<20}\t{'API Number':<12}\t{'Bottomhole Pressure (PSI)':<25}"
                f"\t{'Depth Classification (B3)':<25}\t{'Well Type (TXNET)':<30}\n")
        for api_number, pressure_points in shallow_pressure_data.items():
            for date, pressure, _, color in pressure_points:
                color_str = str(color) if isinstance(color, np.ndarray) else color
                well_str = {"green": "Shallow", "blue": "Deep", "purple": "Both"}.get(color_str, "")
                f.write(
                    f"{str(date):<20}\t{str(api_number):<12}\t{str(pressure):<25}\t{well_str:<25}"
                    f"\t{'Shallow - Strawn Formation':<30}\n")


def plot_b3_ijv(cleandf, earthquake_information, output_dir, range_km):
    deep_injection_data = defaultdict(list)
    shallow_injection_data = defaultdict(list)
    eventID = earthquake_information['Event ID']
    origin_date_str = earthquake_information['Origin Date']
    origin_time = earthquake_information['Origin Time']
    local_magnitude = earthquake_information['Local Magnitude']
    origin_date = pd.to_datetime(origin_date_str)

    # Classify API numbers and prepare injection data
    api_color_map = {}
    for _, row in cleandf.iterrows():
        api_number = row['APINumber']
        injected_volume_bbl = row['InjectedLiquidBBL']
        date = row['StartOfMonthDate']
        depth_class = row['CalculatedPermittedDepthClassification']
        distance_from_eq = row['Distance from Earthquake (km)']

        label_text = f"API {api_number} ({distance_from_eq} km)"

        if depth_class == 'shallow':
            color = 'green'
            shallow_injection_data[api_number].append((date, injected_volume_bbl, label_text, color))
        elif depth_class == 'deep':
            color = 'blue'
            deep_injection_data[api_number].append((date, injected_volume_bbl, label_text, color))
        elif depth_class == 'both':
            color = 'purple'
            deep_injection_data[api_number].append((date, injected_volume_bbl, label_text, color))

        api_color_map[api_number] = color

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)

    # Plot for shallow well data
    shallow_scatter_colors = {}
    for api_number, injection_points in shallow_injection_data.items():
        dates, injections, labels, colors = zip(*injection_points)
        scatter = ax1.scatter(dates, injections, label=labels[0], s=12)
        shallow_scatter_colors[api_number] = scatter.get_edgecolor()
    ax1.set_title(f'{eventID} Monthly Injection Volumes - Shallow Well ({range_km} KM Range)')
    ax1.set_ylabel('Daily Injection (BBLs)')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=2))

    # Plot for deep well data
    deep_scatter_colors = {}
    for api_number, injection_points in deep_injection_data.items():
        dates, injections, labels, colors = zip(*injection_points)
        scatter = ax2.scatter(dates, injections, label=labels[0], s=12)
        deep_scatter_colors[api_number] = scatter.get_edgecolor()
    ax2.set_title(f'{eventID} Monthly Injection Volumes - Deep Well ({range_km} KM Range)')
    ax2.set_ylabel('Daily Injection (BBLs)')

    ax1.axvline(x=origin_date, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=origin_date, color='red', linestyle='--', linewidth=2)

    # Combine legends and change text color
    legend_info_shallow = []
    legend_info_deep = []

    for api_number, injection_points in shallow_injection_data.items():
        _, _, label, _ = injection_points[0]
        color = shallow_scatter_colors.get(api_number, 'black')
        distance_from_eq = float(label.split('(')[-1].split()[0])  # Extract distance from label
        legend_info_shallow.append((distance_from_eq, Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                             label=label, markersize=10)))

    for api_number, injection_points in deep_injection_data.items():
        _, _, label, _ = injection_points[0]
        color = deep_scatter_colors.get(api_number, 'black')
        distance_from_eq = float(label.split('(')[-1].split()[0])  # Extract distance from label
        legend_info_deep.append((distance_from_eq, Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                          label=label, markersize=10)))

    # Custom legend for B3 Well Classification
    classification_legend_handles = [
        Line2D([0], [0], color='green', linestyle='-', label=f'B3 Well Classification Key\nShallow Well'),
        Line2D([0], [0], color='blue', linestyle='-', label='Deep Well'),
        Line2D([0], [0], color='purple', linestyle='-', label='Shallow & Deep')
    ]

    # Sort shallow and deep earthquakes by distance from earthquake
    legend_info_shallow.sort(key=lambda x: x[0])
    legend_info_deep.sort(key=lambda x: x[0])

    custom_legend_handles_shallow = [handle for _, handle in legend_info_shallow]
    custom_legend_handles_deep = [handle for _, handle in legend_info_deep]

    # Update legends for both subplots with legend for earthquake
    legend_handles_shallow = [
        Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_information["Event ID"]}'
                                                            f'\nOrigin Time: {origin_time}'
                                                            f'\nOrigin Date: {origin_date_str}'
                                                            f'\nLocal Magnitude: {local_magnitude}'
                                                            f'\nRange: {range_km} km')]

    # Add all 3 custom legends to ax1
    ax1.legend(handles=classification_legend_handles + custom_legend_handles_shallow + legend_handles_shallow,
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

    legend_handles_deep = [Line2D([0], [0], color='red', linestyle='--', label=f'{earthquake_information["Event ID"]}'
                                                                               f'\nOrigin Time: {origin_time}'
                                                                               f'\nOrigin Date: {origin_date_str}'
                                                                               f'\nLocal Magnitude: {local_magnitude}'
                                                                               f'\nRange: {range_km} km')]

    # Add all 3 custom legends to ax2
    ax2.legend(handles=custom_legend_handles_deep + legend_handles_deep + classification_legend_handles,
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 1])

    legends_list = [ax1.get_legend(), ax2.get_legend()]
    for legend in legends_list:
        if legend is not None:
            for text in legend.get_texts():
                text_str = text.get_text()
                split_text = text_str.split(' ')

                # Ensure the split_text has at least 2 elements before accessing
                if len(split_text) > 1:
                    api_number = split_text[1]

                    if api_number in api_color_map:
                        text_color = api_color_map[api_number]
                        plt.setp(text, color=text_color)
                else:
                    # Handle cases where the text doesn't contain an API number
                    # For example, if it's a classification label like 'Shallow', 'Deep', 'Both'
                    continue

    # Set x-axis to show ticks for each month
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save the plot to a file
    plot_filename = os.path.join(output_dir,
                                 f'injection_over_time_{earthquake_information["Event ID"]}_range{range_km}km.png')
    plt.savefig(plot_filename)

    deep_filename = os.path.join(output_dir,
                                 f'deep_well_injection_data_{earthquake_information["Event ID"]}_range{range_km}km.txt')
    with open(deep_filename, 'w') as f:
        f.write(f"{'Date':<20}\t{'API Number':<12}\t{'Monthly Volume (BBL)':<25}"
                f"\t{'Depth Classification (B3)':<30}\t{'Well Type (TXNET)':<30}\n")
        for api_number, injection_points in deep_injection_data.items():
            for date, injection, _, color in injection_points:
                color_str = str(color) if isinstance(color, np.ndarray) else color
                well_str = {"green": "Shallow", "blue": "Deep", "purple": "Both"}.get(color_str, "")
                f.write(
                    f"{str(date):<20}\t{str(api_number):<12}\t{str(injection):<25}\t{well_str:<25}"
                    f"\t{'Deep - Strawn Formation':<30}\n")

    # Save shallow well injection data to a text file
    shallow_filename = os.path.join(output_dir,
                                    f'shallow_well_injection_data_{earthquake_information["Event ID"]}_range{range_km}km.txt')
    with open(shallow_filename, 'w') as f:
        f.write(f"{'Date':<20}\t{'API Number':<12}\t{'Monthly Volume (BBL)':<25}"
                f"\t{'Depth Classification (B3)':<30}\t{'Well Type (TXNET)':<30}\n")
        for api_number, injection_points in shallow_injection_data.items():
            for date, injection, _, color in injection_points:
                color_str = str(color) if isinstance(color, np.ndarray) else color
                well_str = {"green": "Shallow", "blue": "Deep", "purple": "Both"}.get(color_str, "")
                f.write(
                    f"{str(date):<20}\t{str(api_number):<12}\t{str(injection):<25}\t{well_str:<25}"
                    f"\t{'Shallow - Strawn Formation':<30}\n")
