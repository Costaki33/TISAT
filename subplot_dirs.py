import os
import datetime


def create_indiv_subplot_dirs(base_dir: str):
    # Expand user tilde to full path
    base_directory = os.path.expanduser(base_dir)
    # Define the subdirectories to be created
    subfolders = ['deep_individual_plots', 'shallow_individual_plots']

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_directory, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"[{datetime.datetime.now().replace(microsecond=0)}] Created: {subfolder_path}")
