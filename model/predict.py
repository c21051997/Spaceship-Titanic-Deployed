# Code to load the model and make predictions

import pandas as pd
import joblib

#  Load artifacts from the 'model' directory
model = joblib.load('model/titanic_model.joblib')
scaler = joblib.load('model/scaler.joblib')
training_columns = joblib.load('model/training_columns.joblib')

# Import the preprocessor
from model.preprocessing import preprocess_data

def make_prediction(input_data):
    """
    Takes new passenger data as a list of dictionaries,
    and returns a prediction.
    """
    df = pd.DataFrame(input_data)
    
    # Preprocess the new data using the saved scaler
    processed_df = preprocess_data(df, scaler=scaler, is_train=False)
    
    # Ensure columns match the training data, adding missing columns
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
            
    # Reorder columns to match the training order
    processed_df = processed_df[training_columns]

    # Make prediction
    prediction = model.predict(processed_df)
    
    return prediction