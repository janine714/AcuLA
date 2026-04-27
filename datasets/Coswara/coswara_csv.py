import pandas as pd


input_csv_path = '/Users/wangzaining/Desktop/combined_data.csv' 
output_smoker_csv_path = '/Users/wangzaining/Desktop/smoker_classification.csv' 
output_gender_csv_path = '/Users/wangzaining/Desktop/gender_classification.csv' 

metadata = pd.read_csv(input_csv_path)

smoker_metadata = metadata[['id', 'smoker']].dropna()  
gender_metadata = metadata[['id', 'g']].dropna()  

smoker_metadata.to_csv(output_smoker_csv_path, index=False)
gender_metadata.to_csv(output_gender_csv_path, index=False)

print(f"Smoker classification metadata saved to: {output_smoker_csv_path}")
print(f"Gender classification metadata saved to: {output_gender_csv_path}")
