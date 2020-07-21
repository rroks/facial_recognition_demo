from PIL import Image
from mtcnn import MTCNN
from numpy import asarray

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosine
from keras import backend


class MatchingResult:
    def __init__(self, seq, cosine, verified):
        self.seq = seq
        self.cosine = cosine
        self.verified = verified


# extract a single face from a given photograph
# size is specified to suit keras-vggface util
def extract_face(filename, required_size=(224, 224)):
    print("++++++" + str(type(filename)))
    # load image from file
    image = Image.open(filename.stream)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights and settings
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(faces):
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    backend.clear_session()
    # create a vggface model
    model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    model._make_predict_function()
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
# default is 0.5, set to fit requirements
def is_match(known_embedding, candidate_embedding, thresh=0.2):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return score
