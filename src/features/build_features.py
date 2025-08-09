import os
import argparse
import yaml
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import issparse

def build_features(config_path):
    """
    Builds features for the model using a preprocessing pipeline.
    """
    # Load parameters from the YAML config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get parameters for features and data processing
    feature_params = config['features']
    data_params = config['data_processing']
    
    numerical_features = feature_params['numerical_features']
    categorical_features = feature_params['categorical_features']
    target_column = data_params['target_column']

    # Define file paths
    processed_data_dir = os.path.join("data", "processed")
    train_data_path = os.path.join(processed_data_dir, "train.csv")
    test_data_path = os.path.join(processed_data_dir, "test.csv")
    
    feature_dir = os.path.join("data", "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    train_features_path = os.path.join(feature_dir, "train_features.csv")
    test_features_path = os.path.join(feature_dir, "test_features.csv")
    train_target_path = os.path.join(feature_dir, "train_target.csv")
    test_target_path = os.path.join(feature_dir, "test_target.csv")
    
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)


    # Load the processed datasets
    print("Loading processed data...")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print("Processed data loaded.")

    # Separate features and target
    # We drop 'customerID' here as it's an identifier, not a feature.
    X_train = train_df.drop(columns=[target_column, 'customerID'])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column, 'customerID'])
    y_test = test_df[target_column]
    
    # Binarize the target column ('Yes'/'No' to 1/0)
    y_train = y_train.apply(lambda x: 1 if x == 'Yes' else 0)
    y_test = y_test.apply(lambda x: 1 if x == 'Yes' else 0)


    # Create the preprocessing pipeline
    print("Building feature engineering pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep other columns, if any
    )

    # Fit the pipeline on the training data and transform both sets
    print("Applying feature transformations...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print("Feature transformation complete.")
    
    # Get all feature names from the preprocessor
    # This is the modern and correct way to get names from a ColumnTransformer
    all_feature_names = preprocessor.get_feature_names_out()


    # Convert transformed arrays back to DataFrames
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)


    # Save the feature-engineered data
    X_train_transformed_df.to_csv(train_features_path, index=False)
    X_test_transformed_df.to_csv(test_features_path, index=False)
    y_train.to_csv(train_target_path, index=False, header=True)
    y_test.to_csv(test_target_path, index=False, header=True)
    print(f"Feature-engineered data saved to: {feature_dir}")

    # Save the fitted preprocessor pipeline
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to the config file")
    args = parser.parse_args()

    # Run the feature building process
    build_features(config_path=args.config)
