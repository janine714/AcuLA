import os
import pandas as pd

diagnosis_mapping = {
    'NORMAL': 'Normal',
    'ASD': 'Atrial septal defect',
    'PDA': 'Patent ductus arteriosus',
    'PFO': 'Patent foramen ovale',
    'VSD': 'Ventricular septal defect'
}

base_dir = '/Users/wangzaining/Desktop/ZCHSound'
clean_details_path = os.path.join(base_dir, 'Clean Heartsound Data Details.csv')
noise_details_path = os.path.join(base_dir, 'Noise Heartsound Data Details.csv')

clean_df = pd.read_csv(clean_details_path, sep=';', usecols=['fileName', 'diagnosis'])
noise_df = pd.read_csv(noise_details_path, sep=';', usecols=['fileName', 'diagnosis'])

clean_df['diagnosis'] = clean_df['diagnosis'].map(diagnosis_mapping)
noise_df['diagnosis'] = noise_df['diagnosis'].map(diagnosis_mapping)

clean_df['Data_Type'] = 'Clean'
noise_df['Data_Type'] = 'Noise'

combined_df = pd.concat([clean_df, noise_df], ignore_index=True)

output_path = os.path.join(base_dir, 'zch_metadata.csv')
combined_df.to_csv(output_path, index=False)
