import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads data from JSON or CSV and returns a pandas DataFrame."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Please use JSON or CSV.")
        
        # Ensure we have 'transcript' and 'summary' columns
        if 'transcript' not in df.columns or 'summary' not in df.columns:
            raise ValueError("Data must contain 'transcript' and 'summary' columns.")
        
        return df

    def split_data(self, df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
        """Splits data into train (70%), val (15%), and test (15%)."""
        # First split off the test set
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Calculate the adjusted validation size based on the remaining data
        val_ratio = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=random_state)
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    # Test script
    loader = DataLoader("../data/mock_dataset.json")
    try:
        df = loader.load_data()
        print(f"Loaded {len(df)} records.")
        # If dataset is very small, train_test_split might result in empty splits, but logic remains correct.
        train, val, test = loader.split_data(df)
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    except Exception as e:
        print(f"Error: {e}")
