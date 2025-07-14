# All data cleaning & feature engineering logic

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, scaler=None, is_train=False):
    """
    This function takes a raw dataframe and returns a clean,
    model-ready dataframe, ready for training or prediction.
    """
    # Basic Imputation & Feature Creation
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['VIP'] = df['VIP'].fillna(False)
    
    # Impute categorical columns with the mode
    for col in ['HomePlanet', 'Destination']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Impute Age with the median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Smarter Spending Imputation
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[df['CryoSleep'] == True, spending_cols] = df.loc[df['CryoSleep'] == True, spending_cols].fillna(0)
    for col in spending_cols:
        median_spend = df.loc[df['CryoSleep'] == False, col].median()
        df[col].fillna(median_spend, inplace=True)

    # Advanced Feature Engineering
    # Total Spend
    df['TotalSpend'] = df[spending_cols].sum(axis=1)
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)

    # Cabin features
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # Group features
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    
    # Age Bins
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

    # Final Data Cleaning
    df = df.drop(['Cabin', 'Name', 'Age', 'Group'], axis=1) # Drop original columns

    # One-hot encode all categorical features
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert Data Types
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].astype(int)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce').fillna(0).astype(int)

    # Feature Scaling
    # Identify only the columns that actually need scaling
    cols_to_scale = ['TotalSpend', 'CabinNum', 'GroupSize']
    
    if is_train:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        return df, scaler
    else:
        # On prediction data, use the scaler that was already fitted
        if scaler:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        return df