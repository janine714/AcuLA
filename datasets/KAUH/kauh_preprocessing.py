import os
import pandas as pd

audio_files_dir = 'datasets/KAUH/AudioFiles/'

audio_files = os.listdir(audio_files_dir)

metadata_list = []

location_mapping = {
    'P': 'Posterior', 'L': 'Lower', 'R': 'Right', 'U': 'Upper', 'A': 'Anterior', 'M': 'Middle'
}
sound_type_mapping = {
    'I': 'Inspiratory', 'E': 'Expiratory', 'W': 'Wheezes', 'C': 'Crackles', 'N': 'Normal', 'Crep': 'Crepitations', 'Bronchial': 'Bronchial'
}

# Replace abbreviations with full words
def replace_abbreviations(abbreviation, mapping):
    parts = abbreviation.split(' ')
    full_form = [mapping.get(part, part) for part in parts]
    return ' '.join(full_form)

for file in audio_files:
    if file.endswith('.wav'):
        file_name = file[:-4] 
        print(f"Processing file: {file_name}")  
        
        # Split by the first underscore to separate the ID from the rest
        try:
            ID, rest = file_name.split('_', 1)

            parts = rest.split(',')

            if len(parts) >= 5:
                Diagnosis = parts[0].strip()

                Sound_Type = replace_abbreviations(parts[1].strip(), sound_type_mapping)

                Location = replace_abbreviations(parts[2].strip(), location_mapping)
                
                Age = parts[3].strip()
                Gender = parts[4].strip()

                metadata_list.append({
                    'ID': ID,
                    'Diagnosis': Diagnosis,
                    'Sound_Type': Sound_Type,
                    'Location': Location,
                    'Age': Age,
                    'Gender': Gender,
                    'File_Name': file
                })
            else:
                print(f"Skipping file due to unexpected format: {file_name}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

df = pd.DataFrame(metadata_list)

output_dir = 'datasets/KAUH/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(os.path.join(output_dir, 'metadata_kaudh.csv'), index=False)

print(f"Metadata CSV created with {len(df)} rows")

