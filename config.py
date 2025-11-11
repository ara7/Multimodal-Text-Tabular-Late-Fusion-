# Author: Ara, Lena
# Description: Configuration file for the multimodal classifier project.
# All hyperparameters, file paths, and feature names are stored here.

MODEL_NAME = "dmis-lab/biobert-v1.1"
TRAIN_DATA = "patients_data_train.csv"
TEST_DATA = "patients_data_test.csv"
MODEL_SAVE_PATH = "./models/biobert_demo_dx/"
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5

# This is crucial for fixing your brittle preprocessing
TABULAR_FEATURES_1 = [
            'Patient_age', 'patient_gender', 'patient_race', 'patient_ethnicity','confusin_flag','SID']
TABULAR_FEATURES_2 = ['0_Congestive Heart Failure',
           '1_Cardiac Arrhythmias', '2_Valvular Disease',
           '3_Pulmonary Circulation Disorders', '4_Peripheral Vascular Disorders',
           '5_Hypertension, Uncomplicated', '6_Hypertension, Complicated',
           '7_Paralysis', '8_Other Neurological Disorders',
           '9_Chronic Pulmonary Disease', '10_Diabetes, Uncomplicated',
           '11_Diabetes, Complicated', '12_Hypothyroidism', '13_Renal Failure',
           '14_Liver Disease', '15_Peptic Ulcer Disease Excluding Bleeding',
           '16_AIDS/HIV', '17_Lymphoma', '18_Metastatic Cancer',
           '19_Solid Tumor Without Metastasis',
           '20_Rheumatoid Arthritis/Collagen Vascular Diseases','22_Obesity', '21_Coagulopathy',
           '23_Weight Loss', '24_Fluid And Electrolyte Disorders',
           '25_Blood Loss Anemia', '26_Deficiency Anemia', '27_Alcohol Abuse',
           '28_Drug Abuse', '29_Psychoses', '30_Depression',
           'charlson_score', 'emergency', 'prev_incident_delirium', 'ASA'
]
TEXT_FEATURE = "processed_extracted"
LABEL_COLUMN = "new_label"
