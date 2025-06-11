
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('boston_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}K')
    except:
        return render_template('index.html', prediction_text="Invalid input, please try again.")

if __name__ == '__main__':
    app.run(debug=True)
