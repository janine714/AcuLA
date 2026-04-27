import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def get_openai_response(report, num_qas=3):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a cardiologist tasked with interpreting cardiac auscultation findings. Based on the given report your job is to generate atleast {num_qas} question answers (QAs) without mentioning report. Note that questions should be answered using the report. Your output should be a JSON list of the following structure: {{'QAs': [{{'question': ..., 'answer': ...}}, {{'question': ..., 'answer': ...}}, ...]}} with keys being only question and answer."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": report,
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

def determine_num_qas(meta_report):
    murmur_status = meta_report.lower().split(',')[0].split(':')[1].strip()
    return 3 if murmur_status in ['absent', 'unknown'] else 5

# Read the input CSV file
input_file = 'validation_gen_report.csv'
df = pd.read_csv(input_file)

# Create a list to store all QA pairs
qa_data = []

# Process each row and call the OpenAI API
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    patient_id = row['Patient_ID']
    meta_report = row['Meta_Report']
    gen_report = row['Gen_Report']
    num_qas = determine_num_qas(meta_report)
    try:
        api_response = get_openai_response(gen_report, num_qas)
        qas = json.loads(api_response)
        qas = qas["QAs"]

        for item in qas:
            qa_data.append({
                'Patient_ID': patient_id,
                'Question': item['question'],
                'Answer': item['answer']
            })
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        qa_data.append({
            'Patient_ID': patient_id,
            'Question': f"Error: {str(e)}",
            'Answer': ''
        })

# Create a new DataFrame from the QA data
qa_df = pd.DataFrame(qa_data)
# Save the results to a new CSV file
output_file = 'validation_data_qas.csv'
qa_df.to_csv(output_file, index=False)

print(f"Processing complete. Results saved to {output_file}")