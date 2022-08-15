from image_converter import convert_formats
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from scipy.spatial.distance import cosine


# This function is used to extract the face from a given photograph;
# Basically size is set for 224 up to 224;
def extract_face(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    print(results[0]['box'])
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    # noinspection PyTypeChecker
    face_array = asarray(image)
    return face_array


# This function is used for calling face extraction function and
# getting samples;
def prepare_sample(file_names):
    faces = [extract_face(f) for f in file_names]
    samples = asarray(faces, 'float32')
    return preprocess_input(samples, version=2)


# This function is used for calculation of prediction with face embeddings;
# Standard model is resnet50;
def prepare_prediction(samples):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    prediction = model.predict(samples)
    return prediction


# This function is used for comparison of distance between embeddings
# to check up percent of matches;
def is_match(known_embedding, candidate_embedding):
    thresh = 0.5
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


def main():
    filenames = ['1.jpg',
                 '2.jpg',
                 '3.jpg']
    predictions = prepare_prediction(prepare_sample(filenames))
    print('Tests:')
    is_match(predictions[0], predictions[1])
    is_match(predictions[0], predictions[2])


if __name__ == "__main__":
    main()
