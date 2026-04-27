import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="---")

def get_openai_response(report, num_qas=2):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a clinician tasked with interpreting respiratory auscultation findings. Based on the given diagnosis from a doctor based on auscultation, your job is to generate at least {num_qas} question answers (QAs) without mentioning report. The question and answers should focus on diagnosis, such that what condition is present in the audio as these will be used to train an AI model for audio based health diagnostics. Note that questions should be answered using only the condition report and nothing else. Your output should be a JSON list of the following structure: {{'QAs': [{{'question': ..., 'answer': ...}}, {{'question': ..., 'answer': ...}}, ...]}} with keys being only question and answer. Example: Did the auscultation findings at the trachea include wheezes?, No, wheezes were not present."
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

# Read the input CSV file
input_file = './metadata_train_classification_updated.csv'
df = pd.read_csv(input_file)

# Create a list to store all QA pairs
qa_data = []

# Process each row and call the OpenAI API
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    patient_id = row['Patient_Num']
    diagnosis = row["Event_Type"]
    report = f"Diagnosis: {diagnosis}"
    num_qas = 2
    try:
        api_response = get_openai_response(report, num_qas)
        qas = json.loads(api_response)
        qas = qas["QAs"]
        for item in qas:
            qa_data.append({
                'Patient_Num': patient_id,
                'Question': item['question'],
                'Answer': item['answer']
            })
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        qa_data.append({
            'Patient_Num': patient_id,
            'Question': f"Error: {str(e)}",
            'Answer': ''
        })

# Create a new DataFrame from the QA data
qa_df = pd.DataFrame(qa_data)
# Save the results to a new CSV file
output_file = 'sprsound_train_qas.csv'
qa_df.to_csv(output_file, index=False)
print(f"Processing complete. Results saved to {output_file}")