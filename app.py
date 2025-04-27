import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Load the model dictionary containing 'model' and 'accuracy'
model_data = joblib.load('sales_prediction_model (2).pkl')
model = model_data['model']

@app.route('/')
def index():
    return render_template('index.html', status_message="Please fill the details.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch form data
        form_data = {
            'Item_Identifier': request.form['Item_Identifier'],
            'Item_Weight': float(request.form['Item_Weight']),
            'Item_Fat_Content': request.form['Item_Fat_Content'],
            'Item_Visibility': float(request.form['Item_Visibility']),
            'Item_Type': request.form['Item_Type'],
            'Item_MRP': float(request.form['Item_MRP']),
            'Outlet_Identifier': request.form['Outlet_Identifier'],
            'Outlet_Establishment_Year': int(request.form['Outlet_Establishment_Year']),
            'Outlet_Size': request.form['Outlet_Size'],
            'Outlet_Location_Type': request.form['Outlet_Location_Type'],
            'Outlet_Type': request.form['Outlet_Type']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # --- Create Linear Regression style graph ---
        x = np.array([0, 1])
        y = prediction * x

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, color='blue', label='Predicted Sales Line')
        plt.scatter(1, prediction, color='red', label=f'Predicted Value ₹{round(prediction, 2)}')
        plt.title('Linear Regression - Sales Prediction')
        plt.xlabel('Normalized Input')
        plt.ylabel('Predicted Sales')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return render_template('index.html', prediction=round(prediction, 2), image_url=image_base64, status_message="Prediction Successful!")

    except Exception as e:
        return render_template('index.html', status_message=f"Error during prediction: {str(e)}")

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)


