from flask import Flask, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        model = YOLO('best.pt')
    except Exception as e:
        flash(f'Error loading model: {e}')
        return redirect(url_for('index'))

    results = model.predict(filepath, conf=0.25, save=False)

    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                try:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = model.names[class_id] if class_id in model.names else f"Class_{class_id}"
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 2)
                    })
                except:
                    continue

    result_img = results[0].plot() if results else cv2.imread(filepath)
    result_filename = f"result_{filename}"
    result_path = os.path.join(OUTPUT_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    return render_template('index.html',
                           original_img=f"{UPLOAD_FOLDER}/{filename}",
                           result_img=f"{OUTPUT_FOLDER}/{result_filename}",
                           detections=detections,
                           total_detections=len(detections))

@app.route('/clear')
def clear_results():
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    flash('All images cleared')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
