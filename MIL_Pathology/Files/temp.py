
import pandas as pd

# Load the slide_location.csv into a DataFrame
slide_location_df = pd.read_csv('patient_list.csv')

# Load the train.txt and test.txt into sets
with open('train.txt', 'r') as train_file, open('test.txt', 'r') as test_file:
    train_ids = set(line.strip() for line in train_file)
    test_ids = set(line.strip() for line in test_file)

# Get the unique case_ids from slide_location.csv
slide_location_ids = set(slide_location_df['case_id'])

# Find case_ids that are not in either train or test
missing_ids = slide_location_ids - (train_ids | test_ids)

# Count the missing case_ids
num_missing_ids = len(missing_ids)

# Print the count of missing case_ids and the missing case_ids themselves
if num_missing_ids > 0:
    print(f"Number of case IDs in slide_location.csv not found in train.txt or test.txt: {num_missing_ids}")
    for case_id in missing_ids:
        print(case_id)
else:
    print("All case IDs in slide_location.csv are present in either train.txt or test.txt.")
"""
    
import pandas as pd

# Load the slide_location.csv into a DataFrame
slide_location_df = pd.read_csv('slide_location.csv')

# Load the General_dataset_fixed.csv into a DataFrame
general_dataset_df = pd.read_csv('simple_combined_three.csv')

# Get the unique case_ids from slide_location.csv
slide_location_ids = set(slide_location_df['case_id'])

# Get the unique "Subject ID" values from General_dataset_fixed.csv
general_dataset_ids = set(general_dataset_df['case_id'])

# Find case_ids that are not in General_dataset_fixed.csv
missing_ids = slide_location_ids - general_dataset_ids

# Count the missing case_ids
num_missing_ids = len(missing_ids)

# Print the count of missing case_ids and the missing case_ids themselves
if num_missing_ids > 0:
    print(f"Number of case IDs in slide_location.csv not found in General_dataset_fixed.csv: {num_missing_ids}")
    for case_id in missing_ids:
        print(case_id)
else:
    print("All case IDs in slide_location.csv are present in General_dataset_fixed.csv.")
"""