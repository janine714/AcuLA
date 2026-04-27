import os
import pandas as pd
import json
from collections import Counter

audio_files_dir = '/Users/wangzaining/Desktop/Classification/train_classification_wav'
json_files_dir = '/Users/wangzaining/Desktop/Classification/train_classification_json'
patient_summary_path = '/Users/wangzaining/Desktop/SPRSound_patient_summary.csv'
output_dir = '/Users/wangzaining/Desktop/Classification/'
output_path = os.path.join(output_dir, 'metadata_train_classification.csv')

patient_summary_df = pd.read_csv(patient_summary_path)

metadata_list = []

for json_file in os.listdir(json_files_dir):
    if json_file.endswith('.json'):
        patient_num = json_file.split('_')[0]
        audio_file = f"{json_file[:-5]}.wav"
        audio_path = os.path.join(audio_files_dir, audio_file)
        
        if os.path.exists(audio_path):
            json_path = os.path.join(json_files_dir, json_file)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            record_annotation = data.get("record_annotation", "Unknown")
            if record_annotation == "Poor Quality":
                continue
            
            event_annotations = data.get("event_annotation", [])
            if event_annotations:
                event_types = [event.get("type", "Unknown") for event in event_annotations]
                most_common_event_type = Counter(event_types).most_common(1)[0][0]
            else:
                most_common_event_type = None
            
            metadata_list.append({
                'Patient_Num': patient_num,
                'Record_Annotation': record_annotation,
                'Event_Type': most_common_event_type
            })
        else:
            print(f"Audio file not found for {json_file}")

metadata_df = pd.DataFrame(metadata_list)

metadata_df['Patient_Num'] = metadata_df['Patient_Num'].astype(str)
patient_summary_df['patient_num'] = patient_summary_df['patient_num'].astype(str)

merged_df = metadata_df.merge(patient_summary_df, left_on='Patient_Num', right_on='patient_num', how='left')
merged_df = merged_df.drop(columns=['patient_num', 'age', 'gender'])

merged_df.to_csv(output_path, index=False)
print(f"Metadata saved to {output_path}")
