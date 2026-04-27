import pandas as pd

# Read the CSV file
df = pd.read_csv('training_data.csv')

def generate_meta_report(row):
    report = f"Murmur: {row['Murmur']}, "
    
    if pd.notna(row['Murmur locations']):
        report += f"Murmur locations: {row['Murmur locations']}, "
    
    if pd.notna(row['Most audible location']):
        report += f"Most audible location: {row['Most audible location']}, "
    
    if pd.notna(row['Systolic murmur timing']):
        report += f"Systolic murmur timing: {row['Systolic murmur timing']}, "
        report += f"Shape: {row['Systolic murmur shape']}, "
        report += f"Grading: {row['Systolic murmur grading']}, "
        report += f"Pitch: {row['Systolic murmur pitch']}, "
        report += f"Quality: {row['Systolic murmur quality']}, "
    
    if pd.notna(row['Diastolic murmur timing']):
        report += f"Diastolic murmur timing: {row['Diastolic murmur timing']}, "
        report += f"Shape: {row['Diastolic murmur shape']}, "
        report += f"Grading: {row['Diastolic murmur grading']}, "
        report += f"Pitch: {row['Diastolic murmur pitch']}, "
        report += f"Quality: {row['Diastolic murmur quality']}, "
    
    report += f"Outcome: {row['Outcome']}"
    
    return report

# Create a new DataFrame with Patient_ID and Meta_Report columns
new_df = pd.DataFrame(columns=['Patient_ID', 'Meta_Report'])

# Generate Meta_Report for each row and add to the new DataFrame
for index, row in df.iterrows():
    meta_report = generate_meta_report(row)
    new_df = new_df.append({'Patient_ID': row['Patient ID'], 'Meta_Report': meta_report}, ignore_index=True)

# Save the new DataFrame to a CSV file
new_df.to_csv('meta_report.csv', index=False)

print("CSV file with Patient_ID and Meta_Report has been saved as 'meta_report.csv'")