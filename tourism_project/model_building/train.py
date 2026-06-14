# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-package-prediction-training-experiment")

api = HfApi()

# Load the preprocessed data from Hugging Face
Xtrain_path = "hf://datasets/Anoupama/Tourism_Package_Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Anoupama/Tourism_Package_Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Anoupama/Tourism_Package_Prediction/ytrain.csv"
ytest_path = "hf://datasets/Anoupama/Tourism_Package_Prediction/ytest.csv"

print("Loading preprocessed data...")
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

# Identify numeric features (all features after encoding)
numeric_features = Xtrain.columns.tolist()

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Preprocessor - StandardScaler for all numeric features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("\nStarting MLflow experiment...")
# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    print(f"\nEvaluated {len(results['params'])} parameter combinations")

    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    print(f"\nBest parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    # Predictions
    print("\nMaking predictions...")
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    train_roc_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })

    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Train Accuracy: {train_report['accuracy']:.4f} | Test Accuracy: {test_report['accuracy']:.4f}")
    print(f"Train Precision: {train_report['1']['precision']:.4f} | Test Precision: {test_report['1']['precision']:.4f}")
    print(f"Train Recall: {train_report['1']['recall']:.4f} | Test Recall: {test_report['1']['recall']:.4f}")
    print(f"Train F1-Score: {train_report['1']['f1-score']:.4f} | Test F1-Score: {test_report['1']['f1-score']:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    print("="*50)

    print("\nTest Set Classification Report:")
    print(classification_report(ytest, y_pred_test, target_names=['No Purchase', 'Purchase']))

    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

    # Save the model locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved locally as: {model_path}")

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model logged to MLflow")

    # Upload to Hugging Face
    repo_id = "Anoupama/Tourism_Package_Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_prediction_model_v1.joblib",
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
