# imports
import numpy as np
import imutils
import pickle
import cv2
import cvlib as cvl
import matplotlib
from cvlib.object_detection import draw_bbox

class RecogniseFaces:
    def __init__(self):
        self.baseConfidence = 0.5
        self.nameInFrame = []
        # Load face detector
        protoPath = "static/PretrainedModels/face/deploy.prototxt.txt"
        modelPath = "static/PretrainedModels/face/res10_300x300_ssd_iter_140000.caffemodel"
        self.faceDetector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # Load embedding extractor
        self.embedder = cv2.dnn.readNetFromTorch("static/PretrainedModels/face/nn4.small2.v1.t7")

        # Load recogniser & label encoder
        self.recogniser = pickle.loads(open("static/Recogniser/faceRecognitionModel.pickle", "rb").read())
        self.labelEncoder = pickle.loads(open("static/LabelEncoder/labelEncoder.pickle", "rb").read())

    def recog_face(self, frame):
        del self.nameInFrame[:]
        # using cvlib to detect common objects
        bbox, label, conf = cvl.detect_common_objects(frame, confidence=0.4, model='yolov3-tiny')
        if 'person' not in label:
            frame = draw_bbox(frame, bbox, label, conf, write_conf=True)

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.faceDetector.setInput(imageBlob)
        detections = self.faceDetector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.baseConfidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                predictions = self.recogniser.predict_proba(vec)[0]
                j = np.argmax(predictions)
                probability = predictions[j]
                name = self.labelEncoder.classes_[j]
                if name not in self.nameInFrame:
                    self.nameInFrame.append(name)

                text = "{}: {:.2f}%".format(name, probability * 100)
                labelPos = startY - 10 if startY - 10 > 10 else startY + 10
                frame = cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (startX, labelPos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return frame, self.nameInFrame
