from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def clean_data(df):
    df = df.dropna(subset=['Product_ID', 'Date'])
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def create_income_group(df):
    bins = [10000, 50000, 90000, float('inf')]
    labels = ['Low', 'Medium', 'High']
    df['IncomeGroup'] = pd.cut(df['Annual_Income'], bins=bins, labels=labels)
    return df

def prepare_data_for_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test