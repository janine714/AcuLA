import os
import pandas as pd


dataset_path = "/path/to//ICBHI_final_database"
demographic_file = "/path/to/ICBHI_Challenge_demographic_information.txt"
diagnosis_file = "/path/to/ICBHI_Challenge_diagnosis.txt"

demographic_df = pd.read_csv(demographic_file, sep='\t', header=None, names=['Patient_ID', 'Age', 'Sex', 'Height', 'Weight', 'BMI'])
diagnosis_df = pd.read_csv(diagnosis_file, sep='\t', header=None, names=['Patient_ID', 'Diagnosis'])

demographic_df = demographic_df.drop(columns=['Height', 'Weight', 'BMI'])

demographic_df['Patient_ID'] = demographic_df['Patient_ID'].astype(str)
diagnosis_df['Patient_ID'] = diagnosis_df['Patient_ID'].astype(str)

chest_location_mapping = {
    'Tc': 'Trachea', 'Al': 'Anterior left', 'Ar': 'Anterior right',
    'Pl': 'Posterior left', 'Pr': 'Posterior right', 'Ll': 'Lateral left', 'Lr': 'Lateral right'
}

acquisition_mode_mapping = {
    'sc': 'Sequential/single channel', 'mc': 'Simultaneous/multichannel'
}

equipment_mapping = {
    'AKGC417L': 'AKG C417L Microphone',
    'LittC2SE': '3M Littmann Classic II SE Stethoscope',
    'Litt3200': '3M Littmann 3200 Electronic Stethoscope',
    'Meditron': 'WelchAllyn Meditron Master Elite Electronic Stethoscope'
}

# Replace abbreviations with full words
def replace_abbreviations(value, mapping):
    return mapping.get(value, value)

# Empty list to store the metadata
metadata = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".wav"):
        patient_id, recording_idx, chest_location, acq_mode, equipment = filename[:-4].split('_')

        # Replace abbreviations with full forms
        chest_location_full = replace_abbreviations(chest_location, chest_location_mapping)
        acq_mode_full = replace_abbreviations(acq_mode, acquisition_mode_mapping)
        equipment_full = replace_abbreviations(equipment, equipment_mapping)

        # Corresponding annotation file
        annotation_file = os.path.join(dataset_path, filename.replace('.wav', '.txt'))

        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                for line in f:
                    cycle_start, cycle_end, crackles, wheezes = line.strip().split()

                    metadata.append([patient_id, recording_idx, chest_location_full, acq_mode_full, equipment_full,
                                     cycle_start, cycle_end, crackles, wheezes])

metadata_df = pd.DataFrame(metadata, columns=['Patient_ID', 'Recording_Index', 'Chest_Location', 'Acquisition_Mode',
                                              'Equipment', 'Cycle_Start', 'Cycle_End', 'Crackles', 'Wheezes'])

metadata_df['Patient_ID'] = metadata_df['Patient_ID'].astype(str)

merged_df = pd.merge(metadata_df, demographic_df, on='Patient_ID', how='left')
merged_df = pd.merge(merged_df, diagnosis_df, on='Patient_ID', how='left')

merged_df.to_csv('icbhi_metadata_with_demographics_diagnosis.csv', index=False)

print(f"Total number of rows created: {len(merged_df)}")
print("Metadata CSV file with demographics and diagnosis information created successfully.")
