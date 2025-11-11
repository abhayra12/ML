#!/usr/bin/env python
"""
Model Training Script for Churn Prediction

This script:
1. Loads the telecom customer churn dataset
2. Preprocesses and prepares features
3. Trains a logistic regression model
4. Saves the trained model to disk

Usage:
    python train.py
    
Output:
    model.bin - Serialized sklearn Pipeline (DictVectorizer + LogisticRegression)
"""

import pickle  # For serializing the trained model
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import sklearn  # Version checking

# Import machine learning components
from sklearn.feature_extraction import DictVectorizer  # Convert dicts to numeric vectors
from sklearn.linear_model import LogisticRegression   # Classification model
from sklearn.pipeline import make_pipeline            # Combine preprocessing + model


# Print version information for debugging and reproducibility
print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')




def load_data():
    """
    Load and preprocess the telecom churn dataset.
    
    This function:
    1. Downloads data from URL (or uses local CSV if available)
    2. Standardizes column names (lowercase, no spaces)
    3. Encodes categorical values (lowercase, no spaces)
    4. Handles missing values in totalcharges
    5. Converts churn target to binary (0/1)
    
    Returns:
        pd.DataFrame: Preprocessed churn dataset ready for modeling
    """
    # Load data from GitHub (same dataset as ML Zoomcamp)
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(data_url)

    # Standardize column names: convert to lowercase and replace spaces with underscores
    # This ensures feature names are consistent when creating pipelines
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Identify categorical columns (object dtype)
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    # Standardize categorical values: lowercase and replace spaces
    # Ensures consistent encoding during vectorization
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # Handle totalcharges: convert to numeric, fill invalid values with 0
    # Some records might have missing/invalid charges
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    # Convert churn target to binary: 'yes' → 1 (churned), 'no' → 0 (stayed)
    df.churn = (df.churn == 'yes').astype(int)
    
    return df




def train_model(df):
    """
    Train a churn prediction model using a scikit-learn pipeline.
    
    Pipeline combines:
    1. DictVectorizer - Converts feature dicts to numeric matrix
    2. LogisticRegression - Binary classification model
    
    Args:
        df (pd.DataFrame): Preprocessed churn dataset with all features
        
    Returns:
        sklearn.pipeline.Pipeline: Trained model ready for predictions
    """
    # Define numerical features (continuous values)
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    # Define categorical features (text/discrete values)
    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]

    # Extract target variable (what we're predicting)
    y_train = df.churn
    
    # Convert DataFrame to list of dictionaries for pipeline
    # Each dict represents one customer's features
    train_dict = df[categorical + numerical].to_dict(orient='records')

    # Create pipeline: DictVectorizer → LogisticRegression
    # Benefits of pipeline:
    # - Prevents data leakage (fit_transform only on training data)
    # - Single object for deployment (easier serialization)
    # - Ensures same preprocessing in production
    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    # Train the pipeline on data
    # DictVectorizer learns feature vocabulary, LogisticRegression learns weights
    pipeline.fit(train_dict, y_train)

    return pipeline


def save_model(pipeline, output_file):
    """
    Serialize and save trained model to disk.
    
    Args:
        pipeline (sklearn.pipeline.Pipeline): Trained model to save
        output_file (str): Path where model will be saved (e.g., 'model.bin')
        
    Returns:
        None
        
    Saves:
        Binary pickle file containing the complete pipeline
        Can be loaded later with: pickle.load(open(output_file, 'rb'))
    """
    # Use context manager for safe file handling
    # Automatically closes file even if error occurs
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)



# Main execution
if __name__ == '__main__':
    # Step 1: Load and preprocess data
    df = load_data()
    
    # Step 2: Train the model
    pipeline = train_model(df)
    
    # Step 3: Save to disk for production deployment
    save_model(pipeline, 'model.bin')
    
    # Confirm successful training
    print('Model saved to model.bin')