import pandas as pd
import os

# Define the file paths
input_csv = "../Files/process_list_edited.csv"
output_csv = "../Files/slide_location.csv"
h5_files_path = "/home/csantiago/Datasets/Pathology-CPTAC/feature_maps"

# Read the original CSV file
df = pd.read_csv(input_csv)

# Function to extract the case_id from slide_id
def extract_case_id(slide_id):
    return "-".join(slide_id.split("-")[:-1])

# Create new columns slide_location and case_id
df["slide_location"] = h5_files_path + '/' + df["slide_id"].str.replace(".svs", ".npz")
df["case_id"] = df["slide_id"].apply(extract_case_id)


# Keep only the relevant columns
df = df[["case_id", "slide_location"]]

# Write the new CSV file
df.to_csv(output_csv, index=False)

print(f"New CSV '{output_csv}' has been created successfully.")

# Directory containing .npz files
npz_directory = '/home/csantiago/Datasets/Pathology-TCGA/features'

# Path to the CSV file
csv_file = '../Files/slide_location.csv'

# Get a list of all .npz files in the directory
npz_files = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]

# Initialize an empty DataFrame to store the data
data = {'case_id': [], 'slide_location': []}

# Extract case_id from the .npz filenames and build the DataFrame
for npz_file in npz_files:
    # Split the filename by '-'
    parts = npz_file.split('-')
    
    # Check if there are enough parts to extract case_id
    if len(parts) >= 3:
        case_id = '-'.join(parts[:3])
        slide_location = os.path.join(npz_directory, npz_file)
        data['case_id'].append(case_id)
        data['slide_location'].append(slide_location)
    else: 
        print("ERROR", npz_file)

# Create a DataFrame from the data
new_data_df = pd.DataFrame(data)

# Load the existing CSV file (if it exists)
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
else:
    existing_df = pd.DataFrame(columns=['case_id', 'slide_location'])

# Concatenate the existing DataFrame with the new data
final_df = pd.concat([existing_df, new_data_df], ignore_index=True)

# Sort the DataFrame by the 'case_id' column
final_df.sort_values(by='case_id', inplace=True)

# Save the updated DataFrame to the CSV file
final_df.to_csv(csv_file, index=False)

# Check for duplicate slide_location values
duplicate_slide_locations = final_df[final_df.duplicated('slide_location')]

# Print duplicate slide_location values to standard output
if not duplicate_slide_locations.empty:
    print("Duplicate slide_location values:")
    print(duplicate_slide_locations)
else:
    print("No duplicate slide_location values found.")

print("Rows added to the CSV file and CSV file sorted by 'case_id'.")
