from flask import Flask, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO('best.pt')
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure 'best.pt' exists in your project directory")
    model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            flash('YOLO model not loaded. Please check if best.pt exists.')
            return redirect(url_for('index'))
        
        # Check if file was uploaded
        if 'image' not in request.files:
            flash('No file selected')
            return redirect(url_for('index'))
        
        file = request.files['image']
        
        # Check if filename is empty
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(url_for('index'))
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Check if file was saved successfully
        if not os.path.exists(filepath):
            flash('Error saving uploaded file')
            return redirect(url_for('index'))
        
        # Run YOLO prediction with different confidence thresholds
        print(f"Running prediction on: {filepath}")
        
        # Try with lower confidence first
        results = model.predict(filepath, conf=0.25, save=False, verbose=True)
        print(f"Number of results: {len(results)}")
        
        # Get detection info first
        detections = []
        total_detections = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                total_detections = len(boxes)
                print(f"Found {total_detections} detections")
                
                for i, box in enumerate(boxes):
                    try:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = model.names[class_id] if class_id in model.names else f"Class_{class_id}"
                        
                        print(f"Detection {i+1}: {class_name} ({confidence:.2f})")
                        
                        detections.append({
                            'class': class_name,
                            'confidence': round(confidence * 100, 2)
                        })
                    except Exception as det_error:
                        print(f"Error processing detection {i}: {det_error}")
            else:
                print("No detections found in this result")
        
        # Plot the results (this will show bounding boxes even if confidence is low)
        try:
            # Get the annotated image
            if len(results) > 0:
                result_img = results[0].plot(
                    conf=True,  # Show confidence
                    labels=True,  # Show labels
                    boxes=True,  # Show bounding boxes
                    line_width=2  # Line width for boxes
                )
            else:
                # If no results, just copy the original image
                result_img = cv2.imread(filepath)
        except Exception as plot_error:
            print(f"Error plotting results: {plot_error}")
            # Fallback: copy original image
            result_img = cv2.imread(filepath)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
        
        # Ensure result_img is valid before saving
        if result_img is not None:
            success = cv2.imwrite(result_path, result_img)
            if success:
                print(f"Result saved to: {result_path}")
            else:
                print(f"Failed to save result image to: {result_path}")
                # Copy original as fallback
                import shutil
                shutil.copy2(filepath, result_path)
        else:
            print("Result image is None, copying original")
            import shutil
            shutil.copy2(filepath, result_path)
        
        # Convert path for web display (use forward slashes)
        web_result_path = result_path.replace('\\', '/')
        web_original_path = filepath.replace('\\', '/')
        
        # Print summary
        print(f"Total detections found: {len(detections)}")
        print(f"Detection classes: {[d['class'] for d in detections]}")
        
        # Add message if no detections
        if len(detections) == 0:
            print("No objects detected - trying with lower confidence...")
            # Try with very low confidence
            results_low = model.predict(filepath, conf=0.1, save=False)
            for result in results_low:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for box in boxes:
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
        
        return render_template('index.html', 
                             original_img=web_original_path,
                             result_img=web_result_path,
                             detections=detections,
                             success=True,
                             total_detections=len(detections))
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error during prediction: {str(e)}')
        return redirect(url_for('index'))

@app.route('/debug')
def debug_model():
    """Debug route to check model info"""
    if model is None:
        return "Model not loaded!"
    
    debug_info = {
        'model_loaded': model is not None,
        'model_names': model.names if hasattr(model, 'names') else 'No names available',
        'model_type': str(type(model)),
    }
    
    return f"""
    <h2>Model Debug Info:</h2>
    <pre>{debug_info}</pre>
    <br>
    <a href="/">Back to Home</a>
    """

@app.route('/clear')
def clear_results():
    """Clear uploaded and result images"""
    try:
        # Clear upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear output folder
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        flash('All images cleared successfully!')
    except Exception as e:
        flash(f'Error clearing images: {str(e)}')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists('best.pt'):
        print("WARNING: 'best.pt' model file not found!")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", [f for f in os.listdir('.') if f.endswith('.pt')])
    
    print("Starting Flask application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)