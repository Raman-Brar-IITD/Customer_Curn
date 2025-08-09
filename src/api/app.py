import os
import joblib
import pandas as pd
from flask import Flask, request, render_template
import yaml

# Initialize the Flask app
app = Flask(__name__)

# --- Load Model and Preprocessor ---
# It's better to load these once when the app starts.
try:
    model_path = os.path.join("models", "model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

# --- Load Feature Names from params.yaml ---
try:
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    numerical_features = config['features']['numerical_features']
    categorical_features = config['features']['categorical_features']
    all_feature_names_ordered = numerical_features + categorical_features
except Exception as e:
    print(f"Error loading feature names from params.yaml: {e}")
    all_feature_names_ordered = []


@app.route('/', methods=['GET'])
def home():
    """
    Renders the main page with the prediction form.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, preprocesses it, makes a prediction, and returns the result.
    """
    if not model or not preprocessor:
        return "Model or preprocessor not loaded. Please check the logs."

    try:
        # --- Get Data from Form ---
        # Create a dictionary from the form data
        form_data = request.form.to_dict()
        
        # Convert numerical fields from string to float
        form_data['tenure'] = float(form_data['tenure'])
        form_data['MonthlyCharges'] = float(form_data['MonthlyCharges'])
        form_data['TotalCharges'] = float(form_data['TotalCharges'])

        # --- Create DataFrame ---
        # Create a pandas DataFrame from the dictionary, ensuring the column order
        # matches the one used during training.
        input_df = pd.DataFrame([form_data], columns=all_feature_names_ordered)

        # --- Preprocess and Predict ---
        # Transform the input data using the loaded preprocessor
        print(f"Input DataFrame for preprocessing:\n{input_df}")
        transformed_data = preprocessor.transform(input_df)
        
        # Make a prediction
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)

        # --- Format Output ---
        churn_probability = prediction_proba[0][1]
        if prediction[0] == 1:
            result = f"This customer is likely to churn (Probability: {churn_probability:.2f})."
        else:
            result = f"This customer is likely to stay (Churn Probability: {churn_probability:.2f})."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return render_template('index.html', prediction_text=f"Error: {e}")


if __name__ == "__main__":
    # The app runs on port 5000 and is accessible from any IP address.
    # Use 0.0.0.0 to make it accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)
