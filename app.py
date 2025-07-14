from flask import Flask, request, jsonify
from model.predict import make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        prediction = make_prediction(data)
        
        # The output of our model is a numpy array, so we convert to a boolean
        return jsonify({'transported': bool(prediction[0])})
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Runs the app on port 8000, accessible from other containers
    app.run(host='0.0.0.0', port=8000)