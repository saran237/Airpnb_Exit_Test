from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained Random Forest model
with open("random_forest_airbnb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example: columns expected by your model (update as necessary)
MODEL_FEATURES = ['Host Id', 'Host Since', 'Name', 'Neighbourhood ', 'Property Type',
       'Review Scores Rating (bin)', 'Room Type', 'Zipcode', 'Beds',
       'Number of Records', 'Number Of Reviews', 'Price',
       'Review Scores Rating']

@app.route('/predict', methods=['POST'])
@app.route('/')
def home():
    return "Welcome to the Airbnb Price Prediction API!"
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert incoming data to a DataFrame
    input_df = pd.DataFrame([data])
    
    # TODO: Apply any preprocessing necessary (one-hot encoding, feature engineering, etc.)
    # For example, create the interaction feature if needed
    if 'Neighbourhood ' in input_df.columns and 'Room Type' in input_df.columns:
        input_df['Neighbourhood_RoomType'] = input_df['Neighbourhood '] + "_" + input_df['Room Type']
    
    # One-hot encoding (must match training)
    input_df = pd.get_dummies(input_df)
    # Reindex to match the model's expected features
    input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Return prediction as JSON
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)