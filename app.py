from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all fields safely
        form_values = request.form.to_dict()
        print("Received Form Data:", form_values)  # Debug print

        # Convert values to float
        features = [float(form_values[key]) for key in [
            'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','sqft_above','sqft_basement',
            'yr_built','yr_renovated'
        ]]

        final_input = np.array([features])
        prediction = model.predict(final_input)
        output = round(prediction[0],2)

        return render_template('index.html',
            prediction_text=f"Predicted House Price : ₹ {output:,.2f}")

    except Exception as e:
        print("ERROR:", e)
        return render_template('index.html',
            prediction_text="⚠️ Please fill all fields with valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
