from flask import Flask, request, render_template, redirect, url_for, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the path for the uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def deskew_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    corrected_filename = 'corrected.png'
    corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], corrected_filename)
    cv2.imwrite(corrected_path, rotated)
    print(f"Corrected image saved to: {corrected_path}")
    return corrected_filename, angle

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                corrected_filename, angle = deskew_image(filepath)
                return render_template('index.html', original=filename, corrected=corrected_filename, angle=angle)
    return render_template('index.html', original=None, corrected=None, angle=None)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        filename = "webcam_capture.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        corrected_filename, angle = deskew_image(filepath)
        return render_template('index.html', original=filename, corrected=corrected_filename, angle=angle)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)