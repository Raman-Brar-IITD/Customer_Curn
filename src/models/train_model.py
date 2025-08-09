import os
import argparse
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(config_path):
    """
    Trains the model, logs metrics and artifacts to MLflow, and saves the model.
    """
    # Load parameters from the YAML config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get parameters for training
    training_params = config['training']
    n_estimators = training_params['n_estimators']
    max_depth = training_params['max_depth']
    random_state = training_params['random_state']

    # Define file paths
    feature_dir = os.path.join("data", "features")
    train_features_path = os.path.join(feature_dir, "train_features.csv")
    test_features_path = os.path.join(feature_dir, "test_features.csv")
    train_target_path = os.path.join(feature_dir, "train_target.csv")
    test_target_path = os.path.join(feature_dir, "test_target.csv")
    
    model_path = os.path.join("models", "model.joblib")
    metrics_path = os.path.join("reports", "metrics.json")
    plots_path = os.path.join("reports", "plots")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # Load the feature-engineered data
    print("Loading feature-engineered data...")
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    y_train = pd.read_csv(train_target_path).squeeze()
    y_test = pd.read_csv(test_target_path).squeeze()
    print("Data loaded.")

    # Start an MLflow run
    with mlflow.start_run():
        print("Starting MLflow run...")
        mlflow.log_params(training_params)

        # Initialize and train the model
        print("Training RandomForestClassifier model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        print(f"Metrics: {metrics}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print("Logged metrics to MLflow.")

        # Create and save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        confusion_matrix_path = os.path.join(plots_path, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Log confusion matrix plot to MLflow
        mlflow.log_artifact(confusion_matrix_path, "plots")
        print("Logged confusion matrix to MLflow.")

        # Save the trained model
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Logged model to MLflow.")

        # Save metrics to a JSON file for DVC
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to the config file")
    args = parser.parse_args()
    train_model(config_path=args.config)
