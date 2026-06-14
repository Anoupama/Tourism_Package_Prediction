# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Anoupama/Tourism_Package_Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Cleaning and preparing data
# Drop the unnamed index column if it exists
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# Drop CustomerID as it's a unique identifier (not useful for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Handling data in MaritalStatus column
if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].str.strip().replace("Unmarried", "Single")

# Encode categorical columns
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                        'MaritalStatus', 'Designation']

for col in categorical_features:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("\nDatasets saved locally.")

# Upload to Hugging Face
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Anoupama/Tourism_Package_Prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face")

print("\nData preparation completed successfully!")
