# Falcon Dataset Model - Local Deployment Guide

## 📦 Project Overview https://spaceobject-detection.onrender.com/

This repository contains the codebase and resources for running object detection using a YOLO-based model on the Falcon dataset. The project allows you to deploy the model locally through a simple Flask web interface where you can upload images and receive detection results.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sidpaji28/Falcon_dataset_model.git
cd Falcon_dataset_model
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # For macOS/Linux
# venv\Scripts\activate       # For Windows
```

### 3. Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Otherwise, install typical dependencies manually:

```bash
pip install ultralytics opencv-python flask onnxruntime
```

### 4. Download or Place Model File

Ensure your trained model file (`best.pt` or `best.onnx`) is placed in the project root directory.

---

## 🌐 Running the Application

### 1. Run the Flask Server

```bash
python app.py
```

The server will start at:

```
http://127.0.0.1:10000
```

### 2. Test in Browser

* Open `http://127.0.0.1:10000` in your web browser.
* Upload an image.
* View detections on the output image.

---

## 📂 Project Structure

```
Falcon_dataset_model/
├── app.py               # Flask application
├── best.pt / best.onnx  # Trained YOLO model
├── static/
│   ├── uploads/         # Uploaded images
│   └── output/          # Prediction results
├── templates/
│   └── index.html       # Frontend HTML interface
├── requirements.txt     # Python dependencies (optional)
└── README.md            # This file
```

---

## 🛠️ Common Issues

* **Model not loading:** Ensure `best.pt` or `best.onnx` exists in the project root.
* **Missing folders:** Run the app once; it creates `static/uploads` and `static/output` automatically.
* **Port issues:** Change the port in `app.py` if `10000` is unavailable.
* **Package errors:** Run `pip install` for missing packages.

---

## 💡 Notes

* ONNX format is recommended for faster, lightweight inference, especially during deployment.
* This project is intended for local testing only. For production or public deployment, security hardening is required.

---

## 📧 Contact

For queries or contributions, please open an issue on the GitHub repository.

---
