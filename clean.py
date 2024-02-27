import csv

def filter_rows(input_csv, output_csv, api_numbers_to_keep, column_index):
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header to the new CSV file
        header = next(reader)
        writer.writerow(header)

        # Iterate through the rows and write matching rows to the new CSV file
        for row in reader:
            if row and len(row) > column_index and row[column_index].strip() in api_numbers_to_keep:
                writer.writerow(row)

# Example usage
input_csv_file = '/home/skevofilaxc/Downloads/injectiondata1624.csv'
output_csv_file = '/home/skevofilaxc/Documents/earthquake_plots/aajo/good_wells.csv'
api_numbers_to_keep = ['31742092', '31741765', '31741183', '31740035', '31732701',
                       '31741662', '31738378', '31740243', '31742329']
column_index_to_search = 1  # Adjust this to the index of the column with API numbers (0-based)

filter_rows(input_csv_file, output_csv_file, api_numbers_to_keep, column_index_to_search)
