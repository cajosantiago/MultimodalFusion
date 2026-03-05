import numpy as np
import pandas as pd

train_df = pd.read_csv('../Files/train_dataset.csv')
#train_df['vital_status_12'] = train_df['vital_status_12'].map({0: 1, 1: 0})
labels_train = train_df['vital_status_12']
labels_train= np.array(labels_train)
                       
num_labels_0 = 0
num_labels_1 = 0

for label in labels_train:
    if label == 0:
        num_labels_0 += 1
    elif label == 1:
        num_labels_1 += 1

print("There are {} labels 0 and {} labels 1.".format(num_labels_0, num_labels_1))
print(np.std(labels_train))