from flask import Flask
from faceTracking.findAndExtractFaces import extractEmbedding
from faceTracking.recogniseFaces import recogniseFaces
from faceTracking.modelTraining import modelTraining
from flask import Response
from flask import Flask
from flask import render_template
import cv2
import threading

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

ee = extractEmbedding.ExtractEmbedding()

mt = modelTraining.TrainModel()

video = cv2.VideoCapture("static/TestInput/ff.mp4")


@app.route('/')
def index():
    return render_template("index.html")


def detectFace():
    global video, outputFrame, lock
    rf = recogniseFaces.RecogniseFaces()
    while True:
        grabbed, grabbed_frame = video.read()
        if not grabbed:
            continue

        face_found = rf.recog_face(grabbed_frame)
        if face_found is not None:
            with lock:
                outputFrame = face_found.copy()
            #text, (startX, startY), (endX, endY), labelPos = face_found
            #cv2.rectangle(grabbed_frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            #cv2.putText(grabbed_frame, text, (startX, labelPos),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #with lock:
            #outputFrame = grabbed_frame.copy()


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


@app.route("/render")
def render():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    ee.extraction()
    mt.execute()

    t = threading.Thread(target=detectFace)
    t.daemon = True
    t.start()

    app.run()

video.release()
cv2.destroyAllWindows()
