from fastapi import FastAPI, Request, Depends, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
import models
from database import engine, get_db
import os
import shutil
import cv2
from ultralytics import YOLO

# Create tables
models.Base.metadata.create_all(bind=engine)

from contextlib import asynccontextmanager

def seed_db():
    db = next(get_db())
    if not db.query(models.Work).first():
        works = [
            models.Work(
                title="AI Object Detector",
                description="A powerful YOLOv8-based object detection tool capable of identifying objects in images with high accuracy.",
                image_url="https://via.placeholder.com/600x400?text=Object+Detection", # Placeholder or maybe a static asset
                link="/tool/detection",
                category="AI Tool"
            ),
            models.Work(
                title="Future Project A",
                description="Coming soon. A revolutionary web application.",
                image_url="https://via.placeholder.com/600x400?text=Project+A",
                link="#",
                category="Web App"
            ),
             models.Work(
                title="Future Project B",
                description="Another amazing project in the pipeline.",
                image_url="https://via.placeholder.com/600x400?text=Project+B",
                link="#",
                category="Mobile App"
            )
        ]
        db.add_all(works)
        db.commit()
        print("Database seeded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_db()
    yield

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Config
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
async def read_root(request: Request, db: Session = Depends(get_db)):
    works = db.query(models.Work).all()
    return templates.TemplateResponse(request=request, name="index.html", context={"works": works})

@app.get("/tool/detection")
async def detection_tool(request: Request):
    return templates.TemplateResponse(request=request, name="detection.html", context={})

@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    if not model:
        return templates.TemplateResponse(request=request, name="detection.html", context={"error": "Model not loaded"})

    if not allowed_file(image.filename):
        return templates.TemplateResponse(request=request, name="detection.html", context={"error": "Invalid file type"})

    filename = os.path.basename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Predict
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

    # Save result image
    result_img = results[0].plot() if results else cv2.imread(filepath)
    result_filename = f"result_{filename}"
    result_path = os.path.join(OUTPUT_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    return templates.TemplateResponse(request=request, name="detection.html", context={
        "original_img": f"{UPLOAD_FOLDER}/{filename}",
        "result_img": f"{OUTPUT_FOLDER}/{result_filename}",
        "detections": detections,
        "total_detections": len(detections)
    })
