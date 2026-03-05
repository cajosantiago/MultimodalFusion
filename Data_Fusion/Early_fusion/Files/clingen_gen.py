import pandas as pd

# Load the encoded_three.csv file into a DataFrame
encoded_three_df = pd.read_csv('encoded_three.csv')

# Load the TCGA_gene.csv and CPTAC_gene.csv files into DataFrames
tcga_gene_df = pd.read_csv('TCGA_gene.csv')
cptac_gene_df = pd.read_csv('CPTAC_gene.csv')

# Create new columns in encoded_three_df for VHL_mutation, PBMR1_mutation, and TTN_mutation
encoded_three_df['VHL_mutation'] = -1
encoded_three_df['PBMR1_mutation'] = -1
encoded_three_df['TTN_mutation'] = -1

# Update the values in encoded_three_df using values from TCGA_gene_df and CPTAC_gene_df
for index, row in encoded_three_df.iterrows():
    case_id = row['case_id']
    
    if case_id in tcga_gene_df['case_id'].values:
        tcga_row = tcga_gene_df[tcga_gene_df['case_id'] == case_id]
        encoded_three_df.at[index, 'VHL_mutation'] = tcga_row['VHL_mutation'].values[0]
        encoded_three_df.at[index, 'PBMR1_mutation'] = tcga_row['PBMR1_mutation'].values[0]
        encoded_three_df.at[index, 'TTN_mutation'] = tcga_row['TTN_mutation'].values[0]
    
    elif case_id in cptac_gene_df['case_id'].values:
        cptac_row = cptac_gene_df[cptac_gene_df['case_id'] == case_id]
        encoded_three_df.at[index, 'VHL_mutation'] = cptac_row['VHL_mutation'].values[0]
        encoded_three_df.at[index, 'PBMR1_mutation'] = cptac_row['PBMR1_mutation'].values[0]
        encoded_three_df.at[index, 'TTN_mutation'] = cptac_row['TTN_mutation'].values[0]

# Save the updated DataFrame to a new CSV file
encoded_three_df.to_csv('encoded_three_updated.csv', index=False)
