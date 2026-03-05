import pandas as pd

# Step 1: Read the case IDs from train.txt and test.txt
with open("..Files/train.txt", "r") as train_file:
    train_case_ids = [line.strip() for line in train_file]

with open("..Files/test.txt", "r") as test_file:
    test_case_ids = [line.strip() for line in test_file]

# Step 2: Read the CSV file
df = pd.read_csv("..Files/General_dataset_fixed.csv")

# Step 3: Filter rows based on 'vital_status_12'
filtered_df = df[df['vital_status_12'] != -1]

# Step 4: Filter rows based on 'Subject ID' and create train and test DataFrames
train_df = filtered_df[filtered_df['Subject ID'].isin(train_case_ids)]
test_df = filtered_df[filtered_df['Subject ID'].isin(test_case_ids)]

# Step 5: Get the 'Subject ID' that are not in train.txt or test.txt and 'vital_status_12' is not -1
missing_subject_ids = df[~df['Subject ID'].isin(train_case_ids) & ~df['Subject ID'].isin(test_case_ids) & (df['vital_status_12'] != -1)]['Subject ID']

# Step 6: Append the missing Subject IDs to train_df
train_df = pd.concat([train_df, df[df['Subject ID'].isin(missing_subject_ids)]])

# Step 7: Save the training and testing datasets as CSV
train_df.to_csv("..Files/train.csv", index=False)
test_df.to_csv("..Files/test.csv", index=False)

print("Training and testing datasets created and saved as train.csv and test.csv.")



import pandas as pd

# Load train.csv and sort by 'Subject ID'
train_df = pd.read_csv("..Files/train.csv")
train_df = train_df.sort_values(by='Subject ID')

# Save the sorted train.csv
train_df.to_csv("..Files/train.csv", index=False)

# Load test.csv and sort by 'Subject ID'
test_df = pd.read_csv("..Files/test.csv")
test_df = test_df.sort_values(by='Subject ID')

# Save the sorted test.csv
test_df.to_csv("..Files/test.csv", index=False)

print("Sorted train.csv and test.csv files saved as train_sorted.csv and test_sorted.csv.")
