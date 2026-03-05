import pandas as pd

input_file = '../Files/patients_with_labels_CT.csv'  
output_file = '../Files/patient_list_CT.csv'

# Read the input CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Drop duplicate rows based on 'case_id' column, keeping only the first occurrence
df_unique = df.drop_duplicates(subset='case_id', keep='first')

# Extract the desired columns
extracted_columns = ['case_id', 'vital_status_12']
extracted_df = df_unique[extracted_columns].copy()  # Create a copy of the DataFrame

# Add a new column named "original" with the value 1 for all rows - to know if they have noise or not
extracted_df['original'] = 1

# Sort the extracted data by 'case_id' in ascending order
extracted_df.sort_values(by='case_id', inplace=True)



# Write the extracted data to the output CSV file
extracted_df.to_csv(output_file, index=False)

print(f"Output CSV file '{output_file}' created successfully.")
