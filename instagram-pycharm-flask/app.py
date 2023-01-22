from flask import Flask, request, jsonify
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    pp = request.form.get('pp')
    numperlenu = request.form.get('numperlen')
    fullname = request.form.get('fullname')
    numperlenf = request.form.get('numperlenf')
    nameequsername = request.form.get('nameequsername')
    deslen = request.form.get('deslen')
    exturl = request.form.get('exturl')
    privorpub = request.form.get('privorpub')
    postnum = request.form.get('postnum')
    followers = request.form.get('followers')
    follows = request.form.get('follows')

    input_query = np.array([[pp, numperlenu, fullname, numperlenf, nameequsername, deslen, exturl, privorpub, postnum,
                             followers, follows]])
    result = model.predict(input_query)[0]

    return jsonify({'realorfake':str(result)})


if __name__ == '__main__':
    app.run(debug=True)
