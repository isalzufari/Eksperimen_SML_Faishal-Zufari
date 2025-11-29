import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_file):
  # Load the dataset
  df = pd.read_csv(input_file)

  df_preprocessed = df.drop('StudentID', axis=1)
  
  le = LabelEncoder()
  binary_columns = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
  
  for col in binary_columns:
    df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
    
  return df_preprocessed

if __name__ == "__main__":
  INPUT_FILE = '../placementdata_raw.csv'
  OUTPUT_DIR = './placementdata_preprocessing'
  OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'placement_preprocessed.csv')
  
  print("Memulai preprocessing data...")
  
  processed_df = preprocess_data(INPUT_FILE)
  
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  processed_df.to_csv(OUTPUT_FILE, index=False)
  
  print(f"Preprocessing selesai. Data tersimpan di {OUTPUT_FILE}")