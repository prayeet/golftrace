from flask import Flask, request, send_from_directory, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        processed_filepath = process_video(filepath)
        return jsonify({"processed_file": processed_filepath}), 200
    return jsonify({"error": "Invalid file type"}), 400

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    trajectory = []

    filename = os.path.basename(filepath)
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4')
    out = cv2.VideoWriter(processed_filepath, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            trajectory.append(center)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None:
                continue
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return processed_filename

@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
# Temporary route for manual processing
@app.route('/process_manual', methods=['GET'])
def process_manual():
    filename = 'example_video.mov'  # replace with your video file name
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        processed_filepath = process_video(filepath)
        return jsonify({"processed_file": processed_filepath}), 200
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
    