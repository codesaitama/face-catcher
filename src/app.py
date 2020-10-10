import os.path
import numpy as np
import cv2
import json
from flask import Flask, request, Response
import uuid
from pathlib import Path

folder_path = str(Path().absolute()).replace('\\src\\', '') + '\src\\'

test_path = folder_path + 'face_detect_cascade.xml'


def detect_face(img):

    face_cascade = cv2.CascadeClassifier(folder_path + 'face_detect_cascade.xml')
    iMORPH_GRADIENT = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))

    path_file = (folder_path + 'static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file, img)
    
    return json.dumps(path_file)


#API
app = Flask(__name__)
@app.route('/api/upload', methods=['POST'])

def upload():
    print(test_path)
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img_processed = detect_face(img)
    return Response(response = img_processed, status = 200, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)