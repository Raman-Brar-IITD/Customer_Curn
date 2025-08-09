import os
import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(config_path):
    """
    Reads the raw dataset, processes it, and splits it into train and test sets.
    """
    # Load parameters from the YAML config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get parameters for data processing
    params = config['data_processing']
    test_split_ratio = params['test_split_ratio']
    random_state = params['random_state']
    target_column = params['target_column']

    # Define file paths
    raw_data_path = os.path.join("data", "raw", "Dataset.csv")
    processed_data_dir = os.path.join("data", "processed")
    train_data_path = os.path.join(processed_data_dir, "train.csv")
    test_data_path = os.path.join(processed_data_dir, "test.csv")

    # Create the processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load the raw dataset
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    print(f"Raw data loaded with shape: {df.shape}")

    # --- Data Cleaning ---
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing 'TotalCharges' with the median
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    print("Handled missing values in 'TotalCharges'.")

    # --- Data Splitting ---
    print("Splitting data into training and testing sets...")
    train, test = train_test_split(
        df,
        test_size=test_split_ratio,
        random_state=random_state,
        stratify=df[target_column] # Stratify to maintain churn proportion
    )
    print(f"Training set shape: {train.shape}")
    print(f"Testing set shape: {test.shape}")

    # Save the processed data
    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    print(f"Training data saved to: {train_data_path}")
    print(f"Testing data saved to: {test_data_path}")


if __name__ == "__main__":
    # Set up argument parser to get the config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to the config file")
    args = parser.parse_args()

    # Run the data processing
    make_dataset(config_path=args.config)

