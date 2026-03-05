import pandas as pd
import itertools

# Load your CSV file
df = pd.read_csv('../Files/train.csv')

# Select only the columns related to modalities (CT, MR, pathology, genetic, clinical)
modalities = ['CT', 'MR', 'pathology', 'genetic', 'clinical']
df = df[modalities]

# Get the unique values in each modality column
unique_values = df.apply(lambda col: col.unique(), axis=0)

# Generate all possible combinations of modalities with 0 and 1
combinations = []
for combo in itertools.product(*unique_values):
    combinations.append(combo)

# Print the possible combinations
for i, combo in enumerate(combinations, 1):
    print(f"Combination {i}: {combo}")
