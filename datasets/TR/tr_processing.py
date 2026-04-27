import os
import pandas as pd

dataset_path = "/path/TR/RespiratoryDatabase@TR"
labels_file = "/path/Desktop/TR/Labels.xlsx"

labels_df = pd.read_excel(labels_file)

labels_df['Patient ID'] = labels_df['Patient ID'].astype(str)

metadata = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".wav"):

        patient_id = filename.split('_')[0]

        diagnosis = labels_df.loc[labels_df['Patient ID'] == patient_id, 'Diagnosis'].values
        
        if len(diagnosis) > 0:
            diagnosis = diagnosis[0]
        else:
            diagnosis = 'Unknown' 

        metadata.append([patient_id, filename, diagnosis])


metadata_df = pd.DataFrame(metadata, columns=['Patient_ID', 'File_Name', 'Diagnosis'])


metadata_df.to_csv('path/to/my/file', index=False)

print(f"Total number of rows created: {len(metadata_df)}")
