"""import pandas as pd

# Step 1: Read the case IDs from train.txt and test.txt
with open("../Files/train.txt", "r") as train_file:
    train_case_ids = [line.strip() for line in train_file]

with open("../Files/test.txt", "r") as test_file:
    test_case_ids = [line.strip() for line in test_file]

# Step 2: Read the CSV file and filter rows based on case IDs
df = pd.read_csv("../Files/General_dataset_with_pathology.csv")

filtered_df = df[df['vital_status_12'] != -1]


train_df = filtered_df[filtered_df['Subject ID'].isin(train_case_ids)]
test_df = filtered_df[filtered_df['Subject ID'].isin(test_case_ids)]

# Step 4: Save the training and testing datasets as CSV
train_df.to_csv("../Files/train.csv", index=False)
test_df.to_csv("../Files/test.csv", index=False)

print("Training and testing datasets created and saved as train.csv and test.csv.")
"""