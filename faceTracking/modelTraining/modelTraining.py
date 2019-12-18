# imports

import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class TrainModel:

    def execute(self):
        data = pickle.loads(open("static/Embeddings/faceEmbeddings.pickle", "rb").read())

        labelEncoder = LabelEncoder()
        labels = labelEncoder.fit_transform(data["names"])

        recognizer = SVC(kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        f = open("static/Recogniser/faceRecognitionModel.pickle", "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        f = open("static/LabelEncoder/labelEncoder.pickle", "wb")
        f.write(pickle.dumps(labelEncoder))
        f.close()
