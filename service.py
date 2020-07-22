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

# threshold for similarity
threshold = 0.2


# sanity check route
@app.route('/face/verification', methods=['POST'])
def verify():
    backend.clear_session()

    uploaded_files = request.files
    filenames = list(uploaded_files)
    # need to cast to list type
    file_items = list(uploaded_files.values())

    # only one face on each image can be detected
    # thus it requires at least two images
    if len(uploaded_files) < 2:
        return jsonify(["don't upload less than 2 pictures"])
    # extract faces
    faces = [extract_face(f) for f in file_items]

    # user may upload images without a face
    if any(elem is None for elem in faces):
        return jsonify(["don't upload pictures without faces"])

    embeddings = get_embeddings(faces)
    results = []
    for x in range(1, len(embeddings)):
        result_cosine = is_match(embeddings[0], embeddings[x])
        # need to cast the compareration result to bool type
        # because threshold is a numpy bool which will cause
        # serialization problem when jsonify
        r = MatchingResult(x, result_cosine, bool(result_cosine <= threshold))
        results.append(r)
    return jsonify([ob.__dict__ for ob in results])


if __name__ == '__main__':
    # default is 0.0.0.1
    app.run(host='0.0.0.0', port=11351)
