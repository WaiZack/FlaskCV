# imports
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os


class ExtractEmbedding:

    def extraction(self):
        baseConfidence = 0.5

        # Face Detector
        protoPath = "static/PretrainedModels/deploy.prototxt.txt"
        modelPath = "static/PretrainedModels/res10_300x300_ssd_iter_140000.caffemodel"
        faceDetector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # Face Embedder
        embedder = cv2.dnn.readNetFromTorch("static/PretrainedModels/nn4.small2.v1.t7")

        # Getting Training Images
        imageLocation = "static/TrainingFaces"
        imagePaths = list(paths.list_images(imageLocation))

        # print(imagePaths)

        knownFaces = []
        knownNames = []

        for i, imagePath in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                              (104.0, 177.0, 123.0), swapRB=False, crop=False)
            faceDetector.setInput(imageBlob)
            detections = faceDetector.forward()

            mostLikelyToHaveFace = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, mostLikelyToHaveFace, 2]

            if confidence > baseConfidence:
                box = detections[0, 0, mostLikelyToHaveFace, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fH < 20 or fW < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownFaces.append(vec.flatten())

                data = {"embeddings": knownFaces, "names": knownNames}
                f = open("static/Embeddings/faceEmbeddings.pickle", "wb")
                f.write(pickle.dumps(data))
                f.close()
