from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os.path
from helpers.model_helpers import clean_input as ci


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

my_path = os.path.abspath(os.path.dirname(__file__))

model = pickle.load(open(os.path.join(my_path, 'models/rf_model.pkl'),'rb'))
vect = pickle.load(open(os.path.join(my_path, 'resources/tfidf_vect.pkl'),'rb'))

@app.route('/predict')
def predict():
    # get input text
    #data = request.get_json(silent=True)
    #text = data.get('text', "")
    text = request.args.get('text', "")
    text_cleaned = ci(text)

    # predict
    text_vect = vect.transform([text_cleaned])
    cls = model.predict(text_vect.toarray())
    prob = model.predict_proba(text_vect.toarray())

    # prepare response
    response = {}
    response['text'] = text
    response['text_cleaned'] = text_cleaned
    response['bin'] = cls.tolist()[0]
    response['prob'] = prob.tolist()[0]

    # print to command line
    print(response)

    # log estimations
    estimations_file = os.path.join(my_path, 'logs/estimations.txt')
    with open(estimations_file, "a") as myfile:
         myfile.write(str(response) + "\n")

    # return answer
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')