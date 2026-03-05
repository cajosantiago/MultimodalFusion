"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
df = pd.read_csv('../Files/patient_list_CT.csv')

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['vital_status_12'])

train_df = train_df.sort_values(by='case_id')
test_df = test_df.sort_values(by='case_id')

# Save the train and test sets to new CSV files
train_df.to_csv('../Files/train_dataset.csv', index=False)
test_df.to_csv('../Files/val_dataset.csv', index=False)
"""



import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
df = pd.read_csv('../Files/patient_list_CT.csv')

# Load the original splits from train.txt and test.txt
with open('../Files/train.txt', 'r') as f_train, open('../Files/test.txt', 'r') as f_test:
    train_cases = set(f_train.read().splitlines())
    test_cases = set(f_test.read().splitlines())

# Filter cases that are in patient_list_CT.csv
train_cases = train_cases.intersection(df['case_id'])
test_cases = test_cases.intersection(df['case_id'])

# Create train and test DataFrames
train_df = df[df['case_id'].isin(train_cases)].sort_values(by='case_id')
test_df = df[df['case_id'].isin(test_cases)].sort_values(by='case_id')

# Save the train and test sets to new CSV files
train_df.to_csv('../Files/train_dataset.csv', index=False)
test_df.to_csv('../Files/val_dataset.csv', index=False)
