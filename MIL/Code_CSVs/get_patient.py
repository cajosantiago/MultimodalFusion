import pandas as pd

# Define the paths to the input CSV files
combined_csv_path = "../Files/simple_combined_three.csv"
cptac_csv_path = "../Files/CPTAC/Ranking_CT_short.csv"
tcga_csv_path = "../Files/TCGA/Ranking_CT_short.csv"

# Define the path to the output CSV file
output_csv_path = "../Files/patients_with_labels_CT.csv"

# Read the input CSV files
combined_df = pd.read_csv(combined_csv_path)
cptac_df = pd.read_csv(cptac_csv_path)
tcga_df = pd.read_csv(tcga_csv_path)

# Extract unique case IDs from the combined CSV
case_ids = combined_df['case_id'].unique()

# Create an empty list to store the data
output_data = []

# Iterate over the case IDs
for case_id in case_ids:
    # Check if the case ID exists in CPTAC
    cptac_matches = cptac_df[cptac_df["Subject ID"] == case_id]
    if not cptac_matches.empty:
        for _, row in cptac_matches.iterrows():
            output_data.append({"case_id": case_id, "File Location": row["File Location"]})

    # Check if the case ID exists in TCGA
    tcga_matches = tcga_df[tcga_df["Subject ID"] == case_id]
    if not tcga_matches.empty:
        for _, row in tcga_matches.iterrows():
            output_data.append({"case_id": case_id, "File Location": row["File Location"]})

# Create a DataFrame from the output data
output_df = pd.DataFrame(output_data, columns=["case_id", "File Location"])

#GET LINUX TYPE OF DIRECTORIES
# Remove ".\" from the beginning of every record in the "File Location" column
output_df["File Location"] = output_df["File Location"].str.lstrip(".\\")

# Change "\" to "/" in the "File Location" column
output_df["File Location"] = output_df["File Location"].str.replace(r"\\", "/", regex=True)

# Modify "File Location" column based on directory values
output_df.loc[output_df["File Location"].str.contains("TCGA"), "File Location"] = "/home/csantiago/Datasets/Radiology-TCGA/" + output_df["File Location"]
output_df.loc[output_df["File Location"].str.contains("CPTAC"), "File Location"] = "/home/csantiago/Datasets/Radiology-CPTAC/" + output_df["File Location"]

# Find the highest number of occurrences of the same "case_id"
max_occurrences = output_df['case_id'].value_counts().max()

# Find the case ID(s) with the highest number of occurrences
case_ids_with_max_occurrences = output_df['case_id'].value_counts()[output_df['case_id'].value_counts() == max_occurrences].index

# Print the result
print("The highest number of occurrences of the same 'case_id' is:", max_occurrences)
print("The case ID(s) with the highest number of occurrences:")
for case_id in case_ids_with_max_occurrences:
    print(case_id)

#GET THE LABELs"

# Create a new column "vital_status_12" in the patients DataFrame
output_df["vital_status_12"] = output_df["case_id"].map(combined_df.set_index("case_id")["vital_status_12"])

# Read the text file into a list
with open('../Files/CT_failed_shapes.txt', 'r') as f:
    directories = f.read().splitlines()

# Extract the file locations from the DataFrame
file_locations = output_df['File Location'].tolist()

# Check if any directory matches a file location and filter the DataFrame accordingly
output_df = output_df[~output_df['File Location'].isin(directories)]

# Save the updated DataFrame back to the CSV file
output_df.to_csv(output_csv_path, index=False)