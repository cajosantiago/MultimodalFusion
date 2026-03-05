import pandas as pd

# Load the General_dataset.csv and Path_position_FINAL.csv
general_df = pd.read_csv("../Files/General_dataset.csv")
path_position_df = pd.read_csv("../Files/Path_position_FINAL.csv")

# Merge the two dataframes on "Subject ID" and "case_id" columns
merged_df = general_df.merge(path_position_df, left_on="Subject ID", right_on="case_id", how="left")

# Create the "pathology" column based on your conditions
merged_df["pathology"] = merged_df["case_id"].apply(lambda x: 1 if pd.notnull(x) else 0)

# Reorder the columns to place "pathology" between "MR" and "genetic"
merged_df = merged_df[['Subject ID', 'CT', 'MR', 'pathology', 'genetic', 'clinical', 'vital_status_12', 'vital_status_24']]

# Save the updated dataframe to a new CSV file
merged_df.to_csv("../Files/General_dataset_with_pathology.csv", index=False)

# Count the number of '0's in the "pathology" column
count_0 = (merged_df["pathology"] == 0).sum()

# Print the count
print("Number of '0's in the Pathology column:", count_0)