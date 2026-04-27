import os
import csv
import glob

def parse_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract patient ID and recording locations
    patient_id = lines[0].split()[0]
    recording_locations = '+'.join([line.split()[0] for line in lines[1:] if not line.startswith('#')])

    # Parse the metadata
    metadata = {}
    for line in lines:
        if line.startswith('#'):
            key, value = line.strip('#').split(':', 1)
            metadata[key.strip()] = value.strip()

    return patient_id, recording_locations, metadata

def main():
    input_dir = './test_data'  # Replace with your input directory path
    output_file = 'test_data.csv'

    # Define the header for the CSV file
    header = [
        'Patient ID', 'Recording locations:', 'Age', 'Sex', 'Height', 'Weight',
        'Pregnancy status', 'Murmur', 'Murmur locations', 'Most audible location',
        'Systolic murmur timing', 'Systolic murmur shape', 'Systolic murmur grading',
        'Systolic murmur pitch', 'Systolic murmur quality', 'Diastolic murmur timing',
        'Diastolic murmur shape', 'Diastolic murmur grading', 'Diastolic murmur pitch',
        'Diastolic murmur quality', 'Outcome', 'Campaign', 'Additional ID'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        # Iterate over all .txt files in the input directory
        for txt_file in glob.glob(os.path.join(input_dir, '*.txt')):
            patient_id, recording_locations, metadata = parse_txt_file(txt_file)

            # Prepare the row data
            row = [
                patient_id,
                recording_locations,
                metadata.get('Age', ''),
                metadata.get('Sex', ''),
                metadata.get('Height', ''),
                metadata.get('Weight', ''),
                metadata.get('Pregnancy status', ''),
                metadata.get('Murmur', ''),
                metadata.get('Murmur locations', ''),
                metadata.get('Most audible location', ''),
                metadata.get('Systolic murmur timing', ''),
                metadata.get('Systolic murmur shape', ''),
                metadata.get('Systolic murmur grading', ''),
                metadata.get('Systolic murmur pitch', ''),
                metadata.get('Systolic murmur quality', ''),
                metadata.get('Diastolic murmur timing', ''),
                metadata.get('Diastolic murmur shape', ''),
                metadata.get('Diastolic murmur grading', ''),
                metadata.get('Diastolic murmur pitch', ''),
                metadata.get('Diastolic murmur quality', ''),
                metadata.get('Outcome', ''),
                metadata.get('Campaign', ''),
                metadata.get('Additional ID', '')
            ]

            writer.writerow(row)

    print(f"CSV file '{output_file}' has been created successfully.")