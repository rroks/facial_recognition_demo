from PIL import Image
from mtcnn import MTCNN
from numpy import asarray

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
from keras import backend


# response object for face verification
# seq is the sequence number in the request image queue
# cosine is the similarity value for two faces, threshold is 0.2
# verified is the result determine by this service, if cosine value
# is smaller than the threshold, the two faces are considered to
# be very likely from the same person
class MatchingResult:
    def __init__(self, seq, cosine, verified):
        self.seq = seq
        self.cosine = cosine
        self.verified = verified


# extract a single face from a given image
# size is specified to suit keras-vggface util
# which is 224 x 224 here
def extract_face(filename, required_size=(224, 224)):
    # load image from file by name
    image = Image.open(filename.stream)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights and settings
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # directly return None if no face were detected
    if len(results) == 0:
        return
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
# embedding is the feature of a face
def get_embeddings(faces):
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # clear session before load model
    # without this request will cause errors in a second call to this function
    backend.clear_session()
    # create a vggface model
    model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    model._make_predict_function()
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
# default is 0.5, set to fit requirements, which is 0.2 here
# to get more precise result, better to train a model use specific
# dataset rather than vggface provided by oxford
def is_match(known_embedding, candidate_embedding, thresh=0.2):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return score
