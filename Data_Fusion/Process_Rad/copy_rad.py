import os
import pandas as pd
import shutil

# Define the input CSV file path
input_csv_file = 'Files/patients_with_labels_MR.csv'

# Create the output directory 'CT' if it doesn't exist
output_dir = 'MR'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_file)

# Create a dictionary to keep track of the count for each case_id
case_id_counts = {}

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    case_id = row['case_id']
    file_location = row['File Location']

    # If the case_id is not in the dictionary, initialize it with count 1
    if case_id not in case_id_counts:
        case_id_counts[case_id] = 1

    # Get the count for the current case_id
    count = case_id_counts[case_id]

    # Construct the source path and destination path with unique names
    source_path = os.path.join(file_location, 'GAP_feature_map.npz')
    dest_path = os.path.join(output_dir, f'{case_id}-{count}.npz')

    # Check if the source file exists
    if os.path.exists(source_path):
        # Copy the file to the 'CT' folder with the unique name
        shutil.copy(source_path, dest_path)
        print(f'Copied and renamed {source_path} to {dest_path}')
        
        # Increment the count for the case_id
        case_id_counts[case_id] += 1
    else:
        print(f'File not found: {source_path}')

