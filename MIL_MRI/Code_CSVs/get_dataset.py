import pandas as pd

# Read the input CSV file
input_file = '../Files/patients_with_labels_MR.csv'
df = pd.read_csv(input_file)

# Create a new DataFrame to store the modified records
new_df = pd.DataFrame()

# Iterate over each unique case_id
for case_id in df['case_id'].unique():
    # Count the occurrences of the case_id
    count = df[df['case_id'] == case_id].shape[0]
    
    # Check if the count is less than bag size 47
    if count < 47:
        # Calculate the number of rows to be added
        rows_to_add = 47 - count
        
        # Get the original vital_status_12 value for the case_id
        original_status = df[df['case_id'] == case_id]['vital_status_12'].iloc[0]
        
        # Create a new DataFrame for the additional rows
        additional_rows = pd.DataFrame({
            'case_id': [case_id] * rows_to_add,
            'File Location': ['Not applicable'] * rows_to_add,
            'vital_status_12': [original_status] * rows_to_add
        })
        
        # Append the additional rows to the new DataFrame
        new_df = pd.concat([new_df, additional_rows])

# Concatenate the original DataFrame with the new DataFrame
result_df = pd.concat([df, new_df])
result_df = result_df.sort_values(['case_id', 'File Location'])

output_file = '../Files/MR_scans_dataset.csv'
# Save the result to a new CSV file
result_df.to_csv(output_file, index=False)



