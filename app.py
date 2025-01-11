from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
voting_model = joblib.load('voting_model.pkl')
scaler = joblib.load('scaler.pkl')
sfm = joblib.load('sfm.pkl')  # Assuming you've already saved the selected feature model.

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected features from the form (this could be dynamically populated)
    selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = {}

    # Collect form data for each selected feature
    for feature in selected_features:
        input_data[feature] = request.form.get(feature)
    
    # If any feature is missing from the form, return an error message
    if None in input_data.values():
        return "Error: Missing input data for one or more features", 400
    
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Apply the feature selection model
    input_selected = sfm.transform(input_scaled)
    
    # Make prediction
    prediction = voting_model.predict(input_selected)
    
    # Interpret prediction
    prediction_text = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    
    # Return prediction to the user
    return render_template('index.html', prediction_text=prediction_text, selected_features=selected_features)

if __name__ == '__main__':
    app.run(debug=True)
