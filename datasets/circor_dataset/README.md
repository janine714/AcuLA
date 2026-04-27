Download training set from this link: https://physionet.org/content/circor-heart-sound/1.0.3/
https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip

Validation and test sets are available here: https://drive.google.com/drive/folders/1PFXayY2e1HfrnQ6zpub4y0_UOfbrmG2N?usp=sharing

1) Files with postfix of `_data.csv` contain raw data.
2) Files with postfix of `_data_report.csv` contain meta report generated from raw data, patient ids to map to audio data, and gen_report that will be used for training/evaluation LLMs.
3) File with postfix of `_data_qas.csv` contain question answers.