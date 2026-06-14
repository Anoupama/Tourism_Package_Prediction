# Tourism_Package_Prediction - MLOps Pipeline Assignment

## 📋 Project Objective

The primary objective of this project is to design and deploy an end-to-end MLOps pipeline that predicts whether a customer is likely to purchase the Wellness Tourism Package before being contacted by the marketing team.

As an MLOps Engineer, the responsibilities include:
    * Automating data cleaning, preprocessing, and feature transformation
    * Building, training, and evaluating a predictive machine learning model
    * Implementing CI/CD pipelines using GitHub Actions
    * Ensuring model scalability, reproducibility, and continuous improvement
    
This solution enables the business to make data-driven marketing decisions, optimize customer outreach, and improve customer acquisition while reducing operational overhead.

## 🚀 Live Demo

- **Hugging Face Spaces**: [View Live](https://huggingface.co/spaces/Anoupama/Tourism_Package_Prediction)

## ⚡ Quick Start
```bash
git clone https://github.com/Anoupama/Tourism_Package_Prediction.git
cd Tourism_Project
```

## 🗂️ Project Structure

```
tourism_project/
├── .github/
│   └── workflows/
│       └── pipeline.yml                 # GitHub Actions CI/CD workflow
├── data/
│   └── tourism.csv                      # Original dataset
├── deployment/
│   ├── app.py                           # Streamlit web application
│   ├── Dockerfile                       # Docker configuration
│   └── requirements.txt                 # Deployment dependencies
├── hosting/
│   └── hosting.py                       # Script to push to Hugging Face Spaces
├── model_building/
│   ├── data_register.py                 # Dataset registration to Hugging Face
│   ├── prep.py                          # Data preprocessing script
│   └── train.py                         # Model training with MLflow tracking
└── requirements.txt                     # Workflow dependencies
```

## 🎯 Key Features

### 1. **Data Registration & Preparation**
- Automated dataset upload to Hugging Face Hub
- Comprehensive data cleaning and preprocessing
- Handling of missing values and data quality issues
- Label encoding of categorical variables
- Stratified train-test split (80-20)

### 2. **Model Training**
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Experiment Tracking**: MLflow integration
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 3. **Deployment**
- **Web Application**: Interactive Streamlit app
- **Containerization**: Docker support
- **Hosting**: Hugging Face Spaces

### 4. **CI/CD Pipeline**
- Automated workflow with GitHub Actions
- Four main jobs:
  1. Dataset Registration
  2. Data Preparation
  3. Model Training
  4. Deployment to Hugging Face

## Data Description

The dataset consists of customer demographics and interaction attributes used to predict the likelihood of purchasing the Wellness Tourism Package.

Target Variable
ProdTaken: Indicates whether the customer purchased the package

0 – No
1 – Yes

## Business Impact
This predictive MLOps solution enables:

  - Accurate identification of high-potential customers, improving targeting precision
  - Reduced manual effort and operational inefficiencies through automation
  - Enhanced marketing campaign performance with data-driven insights
  - Scalable and reproducible model deployment across environments
  - Faster adaptation to evolving customer behavior via continuous retraining and monitoring

## Conclusion
By implementing an automated MLOps pipeline, “Visit with Us” can enhance marketing effectiveness, improve customer targeting accuracy, and support sustainable business growth. This project demonstrates how MLOps principles - such as continuous integration, model monitoring, and automated retraining - can be applied to solve real-world challenges in the tourism industry, enabling data-driven decision-making and more personalized customer engagement.
