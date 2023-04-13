from joblib import load
from flask import Flask, render_template, request, jsonify
import numpy as np

# Load the saved model from file
model = load("naive_bayes_model.joblib")

# Define the Flask app
app = Flask(__name__, template_folder='templates')

# Define the Flask route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a Flask route to handle incoming AJAX requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the AJAX request
    data = request.get_json()
    X = np.array([
        data['hair'],
        data['feathers'],
        data['eggs'],
        data['milk'],
        data['airborne'],
        data['aquatic'],
        data['predator'],
        data['toothed'],
        data['backbone'],
        data['breathes'],
        data['venomous'],
        data['fins'],
        data['legs'],
        data['tail'],
        data['domestic'],
        data['catsize'],
        data['Names']
    ])
    # Use the loaded model to make predictions
    y_pred = model.predict(X.reshape(1, -1))
    if y_pred==1:
        return jsonify({"output": "Mammal"})
    elif y_pred == 2:
        return jsonify({"output": "Bird"})
    elif y_pred == 3:
        return jsonify({"output": "Reptile"})
    elif y_pred==4:
        return jsonify({"output": "Fish"})
    elif y_pred==5:
        return jsonify({"output": "Amp"})
    elif y_pred==6:
        return jsonify({"output": "bug"})
    elif y_pred==7:
        return jsonify({"output": "Invertebrate"})
    # Return the predictions as a JSON response
    response = {'output': y_pred.tolist()}
    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
