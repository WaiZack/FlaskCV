from flask import Flask
from faceTracking.findAndExtractFaces import extractEmbedding
from faceTracking.recogniseFaces import recogniseFaces
from faceTracking.modelTraining import modelTraining
from flask import Response
from flask import Flask
from flask import render_template
import cv2
import threading
import wikipedia as wp
from flask_socketio import SocketIO

outputFrame = None
lock = threading.Lock()
nameFound = []
wikiResults = []
appearanceDict = {}

app = Flask(__name__)
socketio = SocketIO(app)

ee = extractEmbedding.ExtractEmbedding()

mt = modelTraining.TrainModel()

print("INFO - LOADING VIDEO")
video = cv2.VideoCapture("static/TestInput/ff.mp4")
#try using cvlib to get all the frames the iterate over the list


@app.route('/')
def index():
    return render_template("index.html")

def detectFace():
    global video, outputFrame, lock, nameFound
    rf = recogniseFaces.RecogniseFaces()
    while True:
        grabbed, grabbed_frame = video.read()
        #terminate when video ends
        if not grabbed:
            print("INFO - VIDEO ENDED")
            break

        faceFound, peopleInFrame = rf.recog_face(grabbed_frame)
        if faceFound is not None:
            with lock:
                outputFrame = faceFound.copy()
                displayInfo(peopleInFrame)
                #nameFound = peopleInFrame.copy()
                # for person in peopleInFrame:
                #   temp = nameDict.get(person)
                #  if temp is None:
                #     nameDict.update({person: 1})
                # else:
                #   nameDict[person] = nameDict.get(person) + 1

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

def displayInfo(names):
    global wikiResults, socketio
    del wikiResults[:]
    webstump = '<table class = "table table-dark"><thead><tr><th>Name</th><th>Wiki Link</th></tr></thead>'
    for name in names:
        appearanceDict.update({name: appearanceDict.get(name, 0) + 1})
    dictKeys = appearanceDict.keys()
    for row in dictKeys:
        webstump = webstump + ('<tr>'+ '<th>'+ row + '<th>' + '</tr>')
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
