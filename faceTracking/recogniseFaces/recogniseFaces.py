# imports
import numpy as np
import imutils
import pickle
import cv2
import os

# Load face detector
protoPath = "deploy.prototxt.txt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
faceDetector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load embedding extractor
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# Load recogniser & label encoder
recogniser = pickle.loads(open("../../static/Recogniser/faceRecognitionModel.pickle", "rb").read())
labelEncoder = pickle.loads(open("../../static/LabelEncoder/labelEncoder.pickle", "rb").read())

video = VideoStream(src=0).start()
