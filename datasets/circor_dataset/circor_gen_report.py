import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(api_key="---")

def get_openai_response(meta_report):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a cardiologist tasked with interpreting cardiac auscultation findings. Based on the given conditions, write 2-3 lines report covering all clinically relevant information. Only use the information given to write about conditions. Please do NOT mention anything about further evaluation or characterization. Your output should be JSON of the following format: {\"report\": ...}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": meta_report,
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    return response.choices[0].message.content

# Read the input CSV file
input_file = 'test_data_meta_report.csv'
df = pd.read_csv(input_file)

# Create a new column for the generated report
df['Gen_Report'] = ''

# Process each row and call the OpenAI API
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    meta_report = row['Meta_Report']
    try:
        api_response = get_openai_response(meta_report)
        gen_report = json.loads(api_response)['report']
        df.at[index, 'Gen_Report'] = gen_report
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df.at[index, 'Gen_Report'] = f"Error: {str(e)}"

# Save the results to a new CSV file
output_file = 'test_gen_report.csv'
df.to_csv(output_file, index=False)

print(f"Processing complete. Results saved to {output_file}")