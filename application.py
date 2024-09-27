from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Corrected the form fields mapping
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            
            # Convert the data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input Data as DataFrame:", pred_df)
            
            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()
            
            # Perform the prediction
            results = predict_pipeline.predict(pred_df)
            print("Prediction Results:", results)

            # Return results to the same home.html template
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render_template('home.html', error="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    # Set debug to True for better error messages
    app.run(host="0.0.0.0",port=8080)
