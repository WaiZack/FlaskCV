from flask import Flask
from faceTracking.findAndExtractFaces import extractEmbedding
from faceTracking.recogniseFaces import recogniseFaces
from faceTracking.modelTraining import modelTraining
from flask import Response
from flask import Flask
from flask import render_template
import cv2
import threading
from flask_socketio import SocketIO
import argparse

# getting input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input test video")
args = vars(ap.parse_args())

# initialising variables
outputFrame = None
lock = threading.Lock()
nameFound = []
appearanceDict = {}

app = Flask(__name__)
socketio = SocketIO(app)

ee = extractEmbedding.ExtractEmbedding()
mt = modelTraining.TrainModel()

print("INFO - LOADING VIDEO")

video = cv2.VideoCapture(args["input"])


#video = cv2.VideoCapture("static/TestInput/ff.mp4")

@app.route('/')
def index():
    return render_template("index.html")


def detectFace():
    global video, outputFrame, lock, nameFound
    rf = recogniseFaces.RecogniseFaces()
    while True:
        grabbed, grabbed_frame = video.read()
        # terminate when video ends
        if not grabbed:
            print("INFO - VIDEO ENDED")
            break

        faceFound, peopleInFrame = rf.recog_face(grabbed_frame)
        if faceFound is not None:
            with lock:
                outputFrame = faceFound.copy()
                displayInfo(peopleInFrame)


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (encoded, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not encoded:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


# Using websockets to update table on front end
def displayInfo(names):
    global socketio
    webstump = '<table class = "table table-dark"><thead><tr><th>Name</th>' \
               '<th>Number of Times Detected</th></tr></thead>'
    for name in names:
        appearanceDict.update({name: appearanceDict.get(name, 0) + 1})
    dictKeys = appearanceDict.keys()
    for row in dictKeys:
        webstump = webstump + (
                    '<tr>' + '<th>' + row + '</th>' + '<th>' + str(appearanceDict.get(row)) + '</th>' + '</tr>')
    webstump = webstump + '</table>'
    socketio.emit('newInfo', webstump)


@app.route("/render")
def render():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    ee.extraction()
    mt.execute()

    t = threading.Thread(target=detectFace)
    t.daemon = True
    t.start()

    socketio.run(app)

video.release()
cv2.destroyAllWindows()
