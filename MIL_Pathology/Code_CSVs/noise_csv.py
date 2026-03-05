import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('../Files/train_dataset.csv')

# Prompt the user to enter the number of times to duplicate the rows
num_duplicates = int(input("Enter the number of times to duplicate the rows: "))

# Duplicate rows where vital_status_12 is '0'
df_duplicates = df[df['vital_status_12'] == 0].copy()

# Concatenate the original DataFrame with the duplicated DataFrame
df_updated = pd.concat([df] + [df_duplicates] * num_duplicates, ignore_index=True)

# Save the updated DataFrame to a new CSV file
df_updated.to_csv('../Files/train_noise_CT.csv', index=False)

# Print the count of records for each label
label_counts = df_updated['vital_status_12'].value_counts()
print("Record counts for each label:")
print(label_counts)
