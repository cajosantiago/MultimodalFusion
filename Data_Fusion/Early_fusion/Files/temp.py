import pandas as pd

# Read the CSV files into dataframes
general_df = pd.read_csv('General_dataset_fixed.csv')
path_df = pd.read_csv('Path_position_FINAL.csv')

# Get a list of unique case_ids from the path_df
unique_case_ids = path_df['case_id'].unique()

# Get a list of unique Subject IDs from the general_df
unique_subject_ids = general_df['Subject ID'].unique()

# Find case_ids that are not in Subject IDs
case_ids_not_in_subjects = [case_id for case_id in unique_case_ids if case_id not in unique_subject_ids]

# Count the number of case_ids not in Subject IDs
num_case_ids_not_in_subjects = len(case_ids_not_in_subjects)

# Print the count
print("Number of case_ids not in Subject IDs:", num_case_ids_not_in_subjects)


# Print case_ids not in Subject IDs
for case_id in case_ids_not_in_subjects:
    print(case_id)

    