__version__ = '1.0'
__author__ = 'Fen'
__email__ = 'felonerroks@gmail.com'

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from keras import backend

from face import extract_face
from face import get_embeddings
from face import is_match
from face import MatchingResult

# configuration
DEBUG = True
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/face/verification', methods=['POST'])
def verify():
    backend.clear_session()
    threshold = 0.2
    uploaded_files = request.files
    filenames = list(uploaded_files)
    file_items = list(uploaded_files.values())

    if len(uploaded_files) < 2:
        return jsonify(["don't upload less than 2 pictures"])
    # extract faces
    faces = [extract_face(f) for f in file_items]
    if any(elem is None for elem in faces):
        return jsonify(["don't upload pictures without faces"])

    embeddings = get_embeddings(faces)
    results = []
    for x in range(1, len(embeddings)):
        result_cosine = is_match(embeddings[0], embeddings[x])
        r = MatchingResult(x, result_cosine, bool(result_cosine <= threshold))
        results.append(r)
    return jsonify([ob.__dict__ for ob in results])
    # return json.dumps([ob.__dict__ for ob in results])


if __name__ == '__main__':
    app.run(host='0.0.0.0')
