import pandas as pd

# Read the two CSV files into pandas DataFrames
df1 = pd.read_csv('../Files/train.csv')
df2 = pd.read_csv('../Files/test.csv')

# Concatenate the DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv('../Files/concatenated.csv', index=False)
