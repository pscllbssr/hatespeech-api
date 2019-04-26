from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os.path

app = Flask(__name__)
CORS(app)

my_path = os.path.abspath(os.path.dirname(__file__))

@app.route('/predict/', methods=['POST'])
def predict():
    # get input text
    #data = request.get_json(silent=True)
    #text = data.get('text', "")
    model_version = request.values.get("model", "v2_rf")
    text = request.values.get('text', "")

    if model_version == "v2_mnb":
        # NAIVE BAYES, iteration 2
        
        # clean
        from helpers.model_helpers import clean_all as ci
        text_cleaned = ci(text)

        # import trained resources
        model = pickle.load(open(os.path.join(my_path, 'models/v2/mnb_model.pkl'),'rb'))
        vect = pickle.load(open(os.path.join(my_path, 'resources/count_wb_vectorizer.pkl'),'rb'))
    
    elif model_version == "v2_rf":
        # RANDOM FOREST, iteration 2
        
        # clean
        from helpers.model_helpers import clean_all as ci
        text_cleaned = ci(text)

        # import trained resources
        model = pickle.load(open(os.path.join(my_path, 'models/v2/rf_model.pkl'),'rb'))
        vect = pickle.load(open(os.path.join(my_path, 'resources/count_wb_vectorizer.pkl'),'rb'))
    
    elif model_version == "v2_svm":
        # SUPPORT VECTOR MACHINE, iteration 2

        # clean
        from helpers.model_helpers import clean_all as ci
        text_cleaned = ci(text)

        # import trained resources
        model = pickle.load(open(os.path.join(my_path, 'models/v2/svm_model.pkl'),'rb'))
        vect = pickle.load(open(os.path.join(my_path, 'resources/count_wb_vectorizer.pkl'),'rb'))
        
    elif model_version == 'v1_rf':
        # oldest version, first try

        # clean
        from helpers.model_helpers import clean_input as ci
        text_cleaned = ci(text)

        # import trained resources
        model = pickle.load(open(os.path.join(my_path, 'models/v1/rf_model.pkl'),'rb'))
        vect = pickle.load(open(os.path.join(my_path, 'resources/tfidf_vect.pkl'),'rb'))

    # predict
    text_vect = vect.transform([text_cleaned])
    cls = model.predict(text_vect.toarray())
    prob = model.predict_proba(text_vect.toarray())    

    # prepare response
    response = {}
    response['text'] = text
    response['text_cleaned'] = text_cleaned
    response['clf'] = model_version
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