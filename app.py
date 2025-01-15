from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

models = joblib.load(open('air_quality.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('frontend 2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['Temperature'], data['Humidity'], data['PM25'], data['PM10'], data['SO2'], data['NO2'], data['CO']])
    prediction = models.predict([features])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
