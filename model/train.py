# Code to train and save the model
import pandas as pd
import joblib
from xgboost import XGBClassifier
from preprocessing import preprocess_data

def train_pipeline():
    """Trains the model and saves the artifacts."""
    
    print("Starting training pipeline...")

    # Load raw data
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv') # We concat to ensure consistent encoding
    combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    
    # Preprocess the data using the dedicated function
    processed_df, scaler = preprocess_data(combined_raw.copy(), is_train=True)
    
    # Split the data back into train and test
    train_df = processed_df[processed_df['Transported'].notnull()]
    
    # Define features (X) and target (y)
    X = train_df.drop(['Transported', 'PassengerId'], axis=1)
    y = train_df['Transported'].astype(int)

    # Initialize and train the final model with best parameters
    print("Training final XGBoost model...")
    final_model = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=200, # Increased slightly from your notebook
        subsample=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    final_model.fit(X, y)
    
    # Save the model, the scaler, and the column list
    joblib.dump(final_model, 'model/titanic_model.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')
    joblib.dump(list(X.columns), 'model/training_columns.joblib')
    
    print("Model training complete and artifacts saved.")

if __name__ == '__main__':
    train_pipeline()