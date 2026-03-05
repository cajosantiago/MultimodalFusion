import pandas as pd

def main(input_csv, output_csv, vitals_csv):
    # Read only the 'case_id' column from the input CSV file
    df = pd.read_csv(input_csv, usecols=['case_id'])

    # Count occurrences of each case_id before dropping duplicates
    value_counts = df['case_id'].value_counts()
    max_occurrences = value_counts.max()
    max_occurrence_case_id = value_counts.idxmax()
    print(f"Max occurrences of the same case_id before deleting duplicates: {max_occurrences}")
    print(f"Case_id with max occurrences: {max_occurrence_case_id}")

    # Read the vitals CSV file
    df_vitals = pd.read_csv(vitals_csv)

    # Merge the data based on the 'case_id' column
    df_output = df.merge(df_vitals[['case_id', 'vital_status_12']], on='case_id', how='left')

    # Remove rows where vital_status_12 is empty
    df_output.dropna(subset=['vital_status_12'], inplace=True)

    # Convert 'vital_status_12' column to integers
    df_output['vital_status_12'] = df_output['vital_status_12'].astype(int)

    # Drop duplicate rows to ensure every case_id is unique
    df_output.drop_duplicates(subset='case_id', inplace=True)

    # Sort the DataFrame by the 'case_id' column
    df_output.sort_values(by='case_id', inplace=True)

    # Save the merged and sorted DataFrame to the output CSV file
    df_output.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv_file = "../Files/slide_location.csv"
    output_csv_file = "../Files/patient_list.csv"
    vitals_csv_file = "../Files/simple_combined_three.csv"
    main(input_csv_file, output_csv_file, vitals_csv_file)
