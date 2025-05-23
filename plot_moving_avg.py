import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict
import os
from scipy.signal import savgol_filter


# Function to prepare injection data
def prepare_daily_injection_data_from_df(finalized_df):
    daily_injection_data = defaultdict(dict)
    distance_data = {}
    for _, row in finalized_df.iterrows():
        api_number = row['API Number']
        date_of_injection = row['Date of Injection']
        volume_injected = row['Volume Injected (BBLs)']
        well_type = row['Well Type']
        distance = row['Distance from Earthquake (km)']
        if isinstance(date_of_injection, str):
            date_of_injection = datetime.datetime.strptime(date_of_injection, '%Y-%m-%d')
        if api_number not in daily_injection_data:
            daily_injection_data[api_number] = {'TYPE': well_type}
            distance_data[api_number] = distance
        if date_of_injection not in daily_injection_data[api_number]:
            daily_injection_data[api_number][date_of_injection] = 0
        daily_injection_data[api_number][date_of_injection] += volume_injected
    return daily_injection_data, distance_data


def prepare_total_pressure_data_from_df(finalized_df):
    total_pressure_data = defaultdict(dict)
    distance_data = {}

    for index, row in finalized_df.iterrows():
        api_number = row['API Number']
        date_of_injection = row['Date of Injection']
        bottomhole_pressure = row['Bottomhole Pressure']
        well_type = row['Well Type']
        distance = row['Distance from Earthquake (km)']

        # Convert date_of_injection to a datetime object
        if isinstance(date_of_injection, str):
            date_of_injection = datetime.datetime.strptime(date_of_injection, '%Y-%m-%d')

        if api_number not in total_pressure_data:
            total_pressure_data[api_number] = {'TYPE': well_type}
            distance_data[api_number] = distance  # Store the distance for each API number

        total_pressure_data[api_number][date_of_injection] = bottomhole_pressure

    return total_pressure_data, distance_data


def append_if_unique(item, target_list):
    if item not in target_list:
        target_list.append(item)


def plot_daily_injection_moving_avg(daily_injection_data, distance_data, earthquake_info, output_directory, range_km, shallow_colormap, deep_colormap):
    print(f"[{datetime.datetime.now().replace(microsecond=0)}] Generating Daily Injection Volume Plot with Moving Average")
    # Create a defaultdict to store the daily injection for each date
    daily_injection_by_date = defaultdict(float)
    deep_injection_data = defaultdict(list)
    shallow_injection_data = defaultdict(list)
    all_api_nums = []  # list to store all the API numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)
    shallow_apis = []
    deep_apis = []

    if not daily_injection_data:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No data to plot.")
        return

    # Check if daily_injection_data is a dictionary
    if not isinstance(daily_injection_data, dict):
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in daily_injection_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, injections = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, daily_injection in zip(unconverted_tuple_dates, injections):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            daily_injection_by_date[date] += daily_injection

    dates, daily_injection_values = zip(*daily_injection_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding injections by date
    sorted_data = sorted(zip(dates, daily_injection_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_daily_injection_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in daily_injection_data.items():
        if data['TYPE'] == 1:
            for date, injection in data.items():
                if date != 'TYPE':
                    deep_injection_data[date].append((api_number, injection))  # Include API number with injection
                    append_if_unique(api_number, deep_apis)
        elif data['TYPE'] == 0:
            for date, injection in data.items():
                if date != 'TYPE':
                    shallow_injection_data[date].append((api_number, injection))  # Include API number with injection
                    append_if_unique(api_number, shallow_apis)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(28, 20))

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_injection = {}  # Dictionary to store median injection for each API number over a 3-day span

    for date, injection_points in shallow_injection_data.items():
        api_injection_values = {}
        for api_number, injection in injection_points:
            if api_number not in api_injection_values:
                api_injection_values[api_number] = []
            api_injection_values[api_number].append(injection)

        for api_number, injection_values in api_injection_values.items():
            median_injection = np.median(injection_values)
            if api_number not in api_median_injection:
                api_median_injection[api_number] = []
            api_median_injection[api_number].append((date, median_injection))

    all_shallow_median_injections = []

    for api_number, median_injection_points in api_median_injection.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, shallow_colormap[api_number])
        dates, injections = zip(*median_injection_points)

        # Plot raw data points for each API number
        ax1.plot(dates, injections, marker='o', linestyle='', color=shallow_colormap[api_number], markersize=2,
                 alpha=0.3)

        # Calculate and plot moving average
        moving_average = pd.Series(injections).rolling(window=10).mean()
        # Find gaps greater than 7 days and break the line at those points
        last_date = dates[0]
        segment_dates = [last_date]
        segment_injections = [moving_average[0]]

        for i in range(1, len(dates)):
            if (dates[i] - last_date).days > 7:
                # Plot the current segment if the gap is too large
                ax1.plot(segment_dates, segment_injections, color=shallow_colormap[api_number], linewidth=2,
                         linestyle='-')
                # Start a new segment
                segment_dates = []
                segment_injections = []

            # Append the current date and injection to the segment
            segment_dates.append(dates[i])
            segment_injections.append(moving_average[i])
            last_date = dates[i]

        # Plot the last segment
        if segment_dates:
            ax1.plot(segment_dates, segment_injections, color=shallow_colormap[api_number], linewidth=2,
                     linestyle='-')

        all_shallow_median_injections.extend(injections)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax1.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(
        f'Reported Daily Injected Volumes with Moving Avg for Shallow Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Reported Injected Volumes (BBLs)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    legend = ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Shallow Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend.set_title("Shallow Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', length=10, width=2, rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_injections:
        shallow_min, shallow_max = np.percentile(all_shallow_median_injections, [5, 95])
        ax1.set_ylim(shallow_min, shallow_max)

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}

    api_median_injection = {}  # Dictionary to store median injection for each API number over a 3-day span

    for date, injection_points in deep_injection_data.items():
        api_injection_values = {}
        for api_number, injection in injection_points:
            if api_number not in api_injection_values:
                api_injection_values[api_number] = []
            api_injection_values[api_number].append(injection)

        for api_number, injection_values in api_injection_values.items():
            median_injection = np.median(injection_values)
            if api_number not in api_median_injection:
                api_median_injection[api_number] = []
            api_median_injection[api_number].append((date, median_injection))

    all_deep_median_injections = []

    for api_number, median_injection_points in api_median_injection.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, deep_colormap[api_number])
        dates, injections = zip(*median_injection_points)

        # Plot raw data points for each API number
        ax2.plot(dates, injections, marker='o', linestyle='', color=deep_colormap[api_number], markersize=2, alpha=0.3)

        cleaned_dates = dates
        cleaned_injections = injections

        moving_average = pd.Series(cleaned_injections).rolling(window=10, min_periods=1).mean()

        # Initialize lists to store segment dates and injections
        segment_dates = []
        segment_injections = []

        # Step 2: Loop through cleaned_dates and cleaned_injections to create segments for consecutive dates only
        for i in range(len(cleaned_dates)):
            # If it's the first element, start a new segment
            if i == 0:
                segment_dates = [cleaned_dates[i]]
                segment_injections = [moving_average[i]]
            else:
                # Check if the current date is exactly the next day after the previous date
                if (cleaned_dates[i] - cleaned_dates[i - 1]).days in [1, 2, 3]:
                    segment_dates.append(cleaned_dates[i])
                    segment_injections.append(moving_average[i])
                else:
                    # Step 3: Process the current segment using Savitzky-Golay and plot it
                    if len(segment_injections) >= 7:  # Ensure enough points for the Savitzky-Golay filter
                        smoothed_segment = savgol_filter(segment_injections, window_length=7, polyorder=2)
                        ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2,
                                 linestyle='-')
                    else:
                        ax2.plot(segment_dates, segment_injections, color=deep_colormap[api_number],
                                 linewidth=2, linestyle='-')

                    # Start a new segment
                    segment_dates = [cleaned_dates[i]]
                    segment_injections = [moving_average[i]]

        # Plot the last segment, if any
        if segment_dates:
            if len(segment_injections) >= 7:
                smoothed_segment = savgol_filter(segment_injections, window_length=7, polyorder=3)
                ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2,
                         linestyle='-')
            else:
                ax2.plot(segment_dates, segment_injections, color=deep_colormap[api_number], linewidth=2,
                         linestyle='-')

        all_deep_median_injections.extend(injections)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax2.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(
        f'Reported Daily Injected Volumes with Moving Avg for Deep Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Reported Daily Injected Volumes (BBLs)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    legend2 = ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Deep Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend2.set_title("Deep Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', length=10, width=2, rotation=45)

    # Calculate y-axis limits for deep wells using the 5th and 95th percentiles
    if all_deep_median_injections:
        deep_min, deep_max = np.percentile(all_deep_median_injections, [5, 95])
        ax2.set_ylim(deep_min, deep_max)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)

    # Save the plot to a file
    output_file_path = os.path.join(output_directory,
                                    f"daily_injection_moving_avg_plot_{earthquake_info['Event ID']}_range{range_km}km.png")
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(
        f"[{datetime.datetime.now().replace(microsecond=0)}] Daily injection with Moving Avg plots Successfully Created and Saved.")


def plot_daily_pressure_moving_avg(listed_pressure_data, distance_data, earthquake_info, output_directory, range_km, shallow_colormap, deep_colormap):
    print(f"[{datetime.datetime.now().replace(microsecond=0)}] Generating Daily Reported Pressure Plot with Moving Average.")
    # Create a defaultdict to store the total pressure for each date
    total_pressure_by_date = defaultdict(float)
    deep_pressure_data = defaultdict(list)
    shallow_pressure_data = defaultdict(list)
    all_api_nums = []  # list to store all the api numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)
    shallow_apis = []
    deep_apis = []

    if not listed_pressure_data:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No data to plot.")
        return

    # Check if total_pressure_data is a dictionary
    if not isinstance(listed_pressure_data, dict):
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in listed_pressure_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, pressures = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, total_pressure in zip(unconverted_tuple_dates, pressures):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            total_pressure_by_date[date] += total_pressure

    dates, total_pressure_values = zip(*total_pressure_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding pressures by date
    sorted_data = sorted(zip(dates, total_pressure_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_total_pressure_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in listed_pressure_data.items():
        if data['TYPE'] == 1:
            for date, pressure in data.items():
                if date != 'TYPE':
                    deep_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
                    append_if_unique(api_number, deep_apis)
        elif data['TYPE'] == 0:
            for date, pressure in data.items():
                if date != 'TYPE':
                    shallow_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
                    append_if_unique(api_number, shallow_apis)

    fig, axes = plt.subplots(2, 1, figsize=(28, 20))

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_pressure_shallow = {}  # Dictionary to store median pressure for each API number over a 3-day span

    for date, pressure_points in shallow_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_shallow:
                api_median_pressure_shallow[api_number] = []
            api_median_pressure_shallow[api_number].append((date, median_pressure))

    all_shallow_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_shallow.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, shallow_colormap[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the shallow well data points
        ax1.plot(dates, pressures, marker='o', linestyle='', color=shallow_colormap[api_number], markersize=2,
                 alpha=0.3)

        # Calculate and plot moving average with segmentation
        moving_average = pd.Series(pressures).rolling(window=10, min_periods=1).mean()
        segment_dates, segment_pressures = [], []
        for i in range(len(dates)):
            if i == 0 or (dates[i] - dates[i - 1]).days <= 3:
                segment_dates.append(dates[i])
                segment_pressures.append(moving_average[i])
            else:
                if len(segment_pressures) >= 7:
                    smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                    ax1.plot(segment_dates, smoothed_segment, color=shallow_colormap[api_number], linewidth=2)
                segment_dates, segment_pressures = [dates[i]], [moving_average[i]]
        if segment_dates:
            if len(segment_pressures) >= 7:
                smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                ax1.plot(segment_dates, smoothed_segment, color=shallow_colormap[api_number], linewidth=2)
            else:
                ax1.plot(segment_dates, segment_pressures, color=shallow_colormap[api_number], linewidth=2)

        all_shallow_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax1.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(
        f'Reported Daily Avg Pressures Used with Moving Avg for Shallow Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Reported Average Pressure (PSIG)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.grid(True)
    legend = ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Shallow Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend.set_title("Shallow Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold

    ax1.tick_params(axis='x', length=10, width=2, rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_bps:
        # print(f"shallow median bps: {all_shallow_median_bps}")

        # Ensure all values in the list are finite numbers
        valid_bps = [bp for bp in all_shallow_median_bps if np.isfinite(bp)]

        if not valid_bps:
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] No valid data points found in all_shallow_median_bps.")
        else:
            # Calculate percentiles only with valid data points
            shallow_min, shallow_max = np.percentile(valid_bps, [5, 95])
            # print(f"shallow_min: {shallow_min}, shallow_max: {shallow_max}")

            # Validate the calculated percentiles
            if not np.isfinite(shallow_min) or not np.isfinite(shallow_max):
                print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid axis limits: shallow_min={shallow_min}, shallow_max={shallow_max}")
            else:
                ax1.set_ylim(shallow_min, shallow_max)
    else:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No data points available to calculate shallow well pressure limits.")

    # Set major locator and formatter to display ticks for each month
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}  # Reset
    api_median_pressure_deep = {}

    for date, pressure_points in deep_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_deep:
                api_median_pressure_deep[api_number] = []
            api_median_pressure_deep[api_number].append((date, median_pressure))

    all_deep_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_deep.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, deep_colormap[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the deep well data points
        ax2.plot(dates, pressures, marker='o', linestyle='', color=deep_colormap[api_number], markersize=2, alpha=0.3)
        # Calculate and plot moving average with segmentation
        moving_average = pd.Series(pressures).rolling(window=10, min_periods=1).mean()
        segment_dates, segment_pressures = [], []
        for i in range(len(dates)):
            if i == 0 or (dates[i] - dates[i - 1]).days == 1:
                segment_dates.append(dates[i])
                segment_pressures.append(moving_average[i])
            else:
                if len(segment_pressures) >= 7:
                    smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                    ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2)
                segment_dates, segment_pressures = [dates[i]], [moving_average[i]]
        if segment_dates:
            if len(segment_pressures) >= 7:
                smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2)
            else:
                ax2.plot(segment_dates, segment_pressures, color=deep_colormap[api_number], linewidth=2)

        all_deep_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax2.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(f'Reported Daily Avg Pressures Used with Moving Avg for Deep Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reported Average Pressure (PSIG)', fontsize=12, fontweight='bold')
    ax2.grid(True)
    legend2 = ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Deep Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend2.set_title("Deep Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold
    ax2.tick_params(axis='x', length=10, width=2, rotation=45)

    # Calculate y-axis limits for deep wells using the 5th and 95th percentiles
    if all_deep_median_bps:
        # Convert to a numpy array if it's not already
        all_deep_median_bps = np.array(all_deep_median_bps)

        # Remove NaN values
        filtered_data = all_deep_median_bps[~np.isnan(all_deep_median_bps)]

        # Check if there is any data left after filtering
        if filtered_data.size > 0:
            # Calculate the 5th and 95th percentiles
            deep_min, deep_max = np.percentile(filtered_data, [5, 95])
            ax2.set_ylim(deep_min, deep_max)
        else:
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] WARNING: No valid data available after removing NaNs. Cannot set axis limits.")
    else:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] ERROR: No deep median bps data available.")

    # Set major locator and formatter to display ticks for each month
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save the plot as an image file
    output_filename = os.path.join(output_directory,
                                   f'event_{earthquake_info["Event ID"]}_reported_avg_pressure_moving_avg_range{range_km}km.png')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"[{datetime.datetime.now().replace(microsecond=0)}] Daily Avg PSIG Pressure plots with Moving Avg Successfully Created and Saved.")


def plot_calculated_bottomhole_pressure_moving_avg(calculated_bottomhole_pressure_data, distance_data, earthquake_info, output_directory, range_km, cleaned_well_data_df, shallow_colormap, deep_colormap):
    print(
        f"[{datetime.datetime.now().replace(microsecond=0)}] Generating Calcualted Daily Bottomhole Pressure (BHP) with Moving Average")
    # Create a defaultdict to store the total pressure for each date
    total_pressure_by_date = defaultdict(float)
    deep_pressure_data = defaultdict(list)
    shallow_pressure_data = defaultdict(list)
    all_api_nums = []  # list to store all the api numbers for plot label
    origin_date_str = earthquake_info['Origin Date']  # Use earthquake origin date directly
    origin_time = earthquake_info['Origin Time']
    local_magnitude = earthquake_info['Local Magnitude']
    origin_date = datetime.datetime.strptime(origin_date_str, '%Y-%m-%d')
    origin_date_num = mdates.date2num(origin_date)
    shallow_apis = []
    deep_apis = []

    if not calculated_bottomhole_pressure_data:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No data to plot.")
        return

    # Check if calculated_bottomhole_pressure_data is a dictionary
    if not isinstance(calculated_bottomhole_pressure_data, dict):
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format. Expected a dictionary.")
        return

    for api_number, api_data in calculated_bottomhole_pressure_data.items():
        # Flatten the dictionary keys into separate lists
        try:
            unconverted_tuple_dates, pressures = zip(*api_data.items())
            all_api_nums.append(api_number)
        except (TypeError, ValueError):
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid data format for API {api_number}. Expected dictionary keys to be datetime tuples.")
            continue

        # Use unconverted_tuple_dates directly since it's already a tuple
        for date, total_pressure in zip(unconverted_tuple_dates, pressures):
            if date == 'TYPE':  # Skip 'TYPE' entries
                continue
            total_pressure_by_date[date] += total_pressure

    dates, total_pressure_values = zip(*total_pressure_by_date.items())

    # Convert all date strings to datetime objects
    dates = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') if isinstance(date_str, str) else date_str for
             date_str in dates]

    # Sort the dates and corresponding pressures by date
    sorted_data = sorted(zip(dates, total_pressure_values), key=lambda x: x[0])

    # Unpack the sorted data
    sorted_dates, sorted_total_pressure_values = zip(*sorted_data)

    # Convert datetime objects to strings
    date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]

    for api_number, data in calculated_bottomhole_pressure_data.items():
        if data['TYPE'] == 1:
            for date, pressure in data.items():
                if date != 'TYPE':
                    deep_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
                    append_if_unique(api_number, deep_apis)
        elif data['TYPE'] == 0:
            for date, pressure in data.items():
                if date != 'TYPE':
                    shallow_pressure_data[date].append((api_number, pressure))  # Include API number with pressure
                    append_if_unique(api_number, shallow_apis)

    fig, axes = plt.subplots(2, 1, figsize=(28, 20))  # Create a 2x1 grid for shallow and deep plots

    # Plot shallow well data
    ax1 = axes[0]
    api_legend_map = {}  # Dictionary to map API numbers to legend labels
    api_median_pressure_shallow = {}  # Dictionary to store median pressure for each API number over a 3-day span

    for date, pressure_points in shallow_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_shallow:
                api_median_pressure_shallow[api_number] = []
            api_median_pressure_shallow[api_number].append((date, median_pressure))

    all_shallow_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_shallow.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, shallow_colormap[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the shallow well data points
        ax1.plot(dates, pressures, marker='o', linestyle='', color=shallow_colormap[api_number], markersize=2,
                 alpha=0.3)

        # Separate the data points for the category 'Only Volume Injected Provided'
        category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
                                             (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        category_pressures = category_data['Bottomhole Pressure']

        # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        ax1.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
                 markeredgecolor='black', markeredgewidth=0.5, markersize=2.5, alpha=0.3)

        # Calculate and plot moving average with segmentation
        moving_average = pd.Series(pressures).rolling(window=10, min_periods=1).mean()
        segment_dates, segment_pressures = [], []

        for i in range(len(dates)):
            if i == 0 or (dates[i] - dates[i - 1]).days <= 3:
                segment_dates.append(dates[i])
                segment_pressures.append(moving_average[i])
            else:
                # Smooth and plot the current segment if it meets length criteria
                if len(segment_pressures) >= 7:
                    smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                    ax1.plot(segment_dates, smoothed_segment, color=shallow_colormap[api_number], linewidth=2)
                # Start a new segment
                segment_dates, segment_pressures = [dates[i]], [moving_average[i]]

        # Plot the last segment, if any
        if segment_dates:
            if len(segment_pressures) >= 7:
                smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                ax1.plot(segment_dates, smoothed_segment, color=shallow_colormap[api_number], linewidth=2)
            else:
                ax1.plot(segment_dates, segment_pressures, color=shallow_colormap[api_number], linewidth=2)

        # Extend the list of all shallow median pressures for further processing
        all_shallow_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax1.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax1.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax1.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax1.set_title(f'Calculated Bottomhole Pressure with Moving Avg for Shallow Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Bottomhole Pressure (PSI)', fontsize=12, fontweight='bold')  # Increase font size and bold
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')  # Increase font size and bold

    ax1.grid(True)
    legend = ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Shallow Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend.set_title("Shallow Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold
    ax1.tick_params(axis='x', length=10, width=2, rotation=45)

    # Calculate y-axis limits for shallow wells using the 5th and 95th percentiles
    if all_shallow_median_bps:
        # print(f"shallow median bps: {all_shallow_median_bps}")

        # Ensure all values in the list are finite numbers
        valid_bps = [bp for bp in all_shallow_median_bps if np.isfinite(bp)]

        if not valid_bps:
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] No valid data points found in all_shallow_median_bps.")
        else:
            # Calculate percentiles only with valid data points
            shallow_min, shallow_max = np.percentile(valid_bps, [5, 95])
            # print(f"shallow_min: {shallow_min}, shallow_max: {shallow_max}")

            # Validate the calculated percentiles
            if not np.isfinite(shallow_min) or not np.isfinite(shallow_max):
                print(f"[{datetime.datetime.now().replace(microsecond=0)}] Invalid axis limits: shallow_min={shallow_min}, shallow_max={shallow_max}")
            else:
                ax1.set_ylim(shallow_min, shallow_max)
    else:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No data points available to calculate shallow well pressure limits.")

    # Set major locator and formatter to display ticks for each month
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot deep well data
    ax2 = axes[1]
    api_legend_map = {}  # Reset
    api_median_pressure_deep = {}

    for date, pressure_points in deep_pressure_data.items():
        api_pressure_values = {}
        for api_number, pressure in pressure_points:
            if api_number not in api_pressure_values:
                api_pressure_values[api_number] = []
            api_pressure_values[api_number].append(pressure)

        for api_number, pressure_values in api_pressure_values.items():
            median_pressure = np.median(pressure_values)
            if api_number not in api_median_pressure_deep:
                api_median_pressure_deep[api_number] = []
            api_median_pressure_deep[api_number].append((date, median_pressure))

    all_deep_median_bps = []

    for api_number, median_pressure_points in api_median_pressure_deep.items():
        if api_number not in api_legend_map:
            distance = distance_data.get(api_number, 'N/A')
            api_legend_map[api_number] = (f'{api_number} ({distance} km)', distance, deep_colormap[api_number])
        dates, pressures = zip(*median_pressure_points)

        # Plot the deep well data points
        ax2.plot(dates, pressures, marker='o', linestyle='', color=deep_colormap[api_number], markersize=2, alpha=0.3)

        # Separate the data points for the category 'Only Volume Injected Provided'
        category_data = cleaned_well_data_df[(cleaned_well_data_df['API Number'] == api_number) &
                                             (cleaned_well_data_df['Category'] == 'Only Volume Injected Provided')]
        category_dates = pd.to_datetime(category_data['Date of Injection'], errors='coerce')
        category_pressures = category_data['Bottomhole Pressure']

        # Plot the data points for the category 'Only Volume Injected Provided' with an outline
        ax2.plot(category_dates, category_pressures, marker='o', linestyle='', color='none',
                 markeredgecolor='black', markeredgewidth=0.5, markersize=2.5, alpha=0.3)

        # Calculate and plot moving average with segmentation
        moving_average = pd.Series(pressures).rolling(window=10, min_periods=1).mean()
        segment_dates, segment_pressures = [], []

        for i in range(len(dates)):
            # Collect consecutive dates or start a new segment if there's a gap
            if i == 0 or (dates[i] - dates[i - 1]).days == 1:
                segment_dates.append(dates[i])
                segment_pressures.append(moving_average[i])
            else:
                # Plot the smoothed segment if long enough
                if len(segment_pressures) >= 7:
                    smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                    ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2)
                # Start a new segment
                segment_dates, segment_pressures = [dates[i]], [moving_average[i]]

        # Plot the last segment, if any
        if segment_dates:
            if len(segment_pressures) >= 7:
                smoothed_segment = savgol_filter(segment_pressures, window_length=7, polyorder=2)
                ax2.plot(segment_dates, smoothed_segment, color=deep_colormap[api_number], linewidth=2)
            else:
                ax2.plot(segment_dates, segment_pressures, color=deep_colormap[api_number], linewidth=2)

        # Extend the list of all deep median pressures for further processing
        all_deep_median_bps.extend(pressures)

    legend_handles = []
    sorted_legend_items = sorted(api_legend_map.values(), key=lambda x: x[1])
    for legend_label, _, color in sorted_legend_items:
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=color, label=legend_label))

    x_min, x_max = ax2.get_xlim()
    if not (x_min <= origin_date_num <= x_max):
        x_min = min(x_min, origin_date_num)
        x_max = max(x_max, origin_date_num)

        # Add a margin (e.g., 2% of the total range) to the right side if the origin date does not fall within the bounds
        # May not fall within bounds if the reported data does not include plots past the earthquake time
        # May be due to under-reporting
        range_x = x_max - x_min
        margin = range_x * 0.02
        ax2.set_xlim(x_min, x_max + margin)
    # Always draw the vertical line
    ax2.axvline(x=origin_date_num, color='red', linestyle='--', zorder=2)
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label=f'Earthquake Event: {earthquake_info["Event ID"]}'
                                                                              f'\nOrigin Time: {origin_time}'
                                                                              f'\nOrigin Date: {origin_date_str}'
                                                                              f'\nLocal Magnitude: {local_magnitude}'
                                                                              f'\nRange: {range_km} km'))

    ax2.set_title(f'Calculated Bottomhole Pressure with Moving Avg for Deep Wells near event_{earthquake_info["Event ID"]} in a {range_km} KM Range', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')  # Increase font size and bold
    ax2.set_ylabel('Total Bottomhole Pressure (PSI)', fontsize=12, fontweight='bold')
    ax2.grid(True)
    legend2 = ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=2,
                        title="Deep Well Information and Earthquake Details",
                        prop={'size': 10})  # Increase handle font size

    # Bold the title and adjust its font size
    legend2.set_title("Deep Well Information and Earthquake Details",
                     prop={'size': 12, 'weight': 'bold'})  # Title font size and bold
    ax2.tick_params(axis='x', rotation=45)

    # Calculate y-axis limits for deep wells using the 5th and 95th percentiles
    if all_deep_median_bps:
        # Convert to a numpy array if it's not already
        all_deep_median_bps = np.array(all_deep_median_bps)

        # Remove NaN values
        filtered_data = all_deep_median_bps[~np.isnan(all_deep_median_bps)]

        # Check if there is any data left after filtering
        if filtered_data.size > 0:
            # Calculate the 5th and 95th percentiles
            deep_min, deep_max = np.percentile(filtered_data, [5, 95])
            ax2.set_ylim(deep_min, deep_max)
        else:
            print(f"[{datetime.datetime.now().replace(microsecond=0)}] WARNING: No valid data available after removing NaNs. Cannot set axis limits.")
    else:
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] No deep median bps data available.")

    # Set major locator and formatter to display ticks for each month
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save the plot as an image file
    output_filename = os.path.join(output_directory,
                                   f'event_{earthquake_info["Event ID"]}_calc_bottomhole_pressure_moving_avg_range{range_km}km.png')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"[{datetime.datetime.now().replace(microsecond=0)}] Daily Calculated BHP Plots Successfully Created and Saved.")