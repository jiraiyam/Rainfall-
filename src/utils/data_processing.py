import pandas as pd
import numpy as np

def load_weather_data(file_path):
    """
    Load weather data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Processed weather data
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess weather data
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    if df is None:
        return None
    
    # Add additional preprocessing steps
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    return df 