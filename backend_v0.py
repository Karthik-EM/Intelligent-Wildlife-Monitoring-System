import os
import shutil
import base64
import json
import subprocess
import sys
import time
import uuid
import torch
import gc
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import Lock
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================

# --- Path Configuration (WINDOWS PORTABLE) ---
BASE_DIR = os.getcwd()
MODEL_FOLDER = os.path.join(BASE_DIR, "speciesnet_model")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
DETECTIONS_DIR = os.path.join(BASE_DIR, "static", "detections")
VIDEO_DIR = os.path.join(BASE_DIR, "uploads", "videos")
TEMP_DIR = os.path.join(BASE_DIR, "temp_inference", "frames")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # Added generic upload dir

# Set environment variables for SpeciesNet
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["SPECIESNET_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR

# Create necessary folders
os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# --- Hardware Check ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📂 WORKING DIR: {BASE_DIR}")
print(f"📂 IMAGES SAVE TO: {DETECTIONS_DIR}")
print(f"🖥️ DEVICE: {DEVICE}")

# --- Database Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "species_data.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Global Video Processor
video_processor = None

# ==========================================
# 2. DATABASE MODELS
# ==========================================
class VideoRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    detections = db.relationship('DetectionResult', backref='video', lazy=True)

class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video_record.id'), nullable=False)
    species = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp_in_video = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)

with app.app_context():
    db.create_all()

# ==========================================
# 3. VIDEO PROCESSOR CLASS
# ==========================================
class BatchVideoProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def process_video_batched(self, video_path, batch_size=8, sample_fps=1, min_confidence=0.3, country='IND'):
        print(f"\n🏆 PROCESSING VIDEO: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0: 
            print("❌ Error: Video FPS is 0. File might be corrupted.")
            return []

        frame_interval = int(max(1, fps / sample_fps))
        
        paths_buffer = []
        timestamps_buffer = []
        all_detections = []
        
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break 

                if frame_count % frame_interval == 0:
                    current_time_sec = frame_count / fps

                    # Save temp frame
                    frame_name = f"frame_{uuid.uuid4().hex}.jpg"
                    frame_path = os.path.join(TEMP_DIR, frame_name)
                    cv2.imwrite(frame_path, frame)

                    paths_buffer.append(frame_path)
                    timestamps_buffer.append(current_time_sec)

                    if len(paths_buffer) >= batch_size:
                        batch_detections = self._process_batch(paths_buffer, timestamps_buffer, country, min_confidence)
                        all_detections.extend(batch_detections)
                        
                        # Cleanup temp files
                        for p in paths_buffer:
                            if os.path.exists(p): os.remove(p)
                        paths_buffer = []
                        timestamps_buffer = []
                        
                        print(f"⏳ Progress: {frame_count/total_frames:.1%}")

                frame_count += 1

            # Process remaining
            if paths_buffer:
                batch_detections = self._process_batch(paths_buffer, timestamps_buffer, country, min_confidence)
                all_detections.extend(batch_detections)
                for p in paths_buffer:
                    if os.path.exists(p): os.remove(p)

        finally:
            cap.release()

        print(f"✅ Video processing complete. Found {len(all_detections)} detections.")
        return all_detections

    def _process_batch(self, filepaths, timestamps, country, min_confidence):
        if not self.model_manager.model: return []
        
        try:
            path_to_time = {fp: ts for fp, ts in zip(filepaths, timestamps)}
            
            # Predict
            result = self.model_manager.model.predict(
                filepaths=filepaths,
                country=country,
                batch_size=len(filepaths)
            )
            
            valid_detections = []
            predictions = result.get('predictions', {})

            if isinstance(predictions, list):
                iterator = zip(filepaths, predictions)
                is_dict = False
            elif isinstance(predictions, dict):
                iterator = predictions.items()
                is_dict = True
            else:
                return []

            for item in iterator:
                path_key, pred_data = item if not is_dict else item
                if not pred_data: continue
                
                class_data = pred_data.get("classifications", {})
                if not class_data: continue

                top_score = class_data.get("scores", [0])[0]

                if top_score >= min_confidence:
                    top_class = class_data.get("classes", ["Unknown"])[0]
                    
                    if ";" in top_class:
                        parts = [p.strip() for p in top_class.split(";") if p.strip()]
                        common_name = parts[-1].title()
                    else:
                        common_name = top_class.title()

                    time_sec = path_to_time.get(path_key, 0)
                    
                    # Save Permanent Image
                    unique_name = f"det_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(DETECTIONS_DIR, unique_name)

                    if os.path.exists(path_key):
                        img = cv2.imread(path_key)
                        if img is not None:
                            # Resize for web (max width 640)
                            h, w = img.shape[:2]
                            if w > 640:
                                scale = 640 / w
                                img = cv2.resize(img, (640, int(h * scale)))
                            cv2.imwrite(save_path, img)
                            image_url = f"/static/detections/{unique_name}"
                        else:
                            image_url = None
                    else:
                        image_url = None

                    valid_detections.append({
                        "species": common_name,
                        "confidence": float(top_score),
                        "timestamp": time_sec,
                        "image_url": image_url
                    })
            return valid_detections
        except Exception as e:
            print(f"❌ Batch error: {e}")
            return []

# ==========================================
# 4. MODEL MANAGER
# ==========================================
class ModelManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = None
                    cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized: return True
        
        if not os.path.exists(MODEL_FOLDER):
            print(f"❌ MODEL NOT FOUND AT: {MODEL_FOLDER}")
            return False
        
        # Check for specific files inside the folder
        required_file = os.path.join(MODEL_FOLDER, "info.json")
        if not os.path.exists(required_file):
             print(f"❌ ERROR: 'info.json' not found inside {MODEL_FOLDER}")
             print("👉 Please ensure you moved all files from the 'archive' subfolder into 'speciesnet_model'")
             return False

        print("="*60)
        print("🧠 LOADING SPECIESNET MODEL...")
        print(f"📍 Device: {DEVICE}")
        print("="*60)

        try:
            from speciesnet import SpeciesNet
            self.model = SpeciesNet(
                model_name=MODEL_FOLDER,
                components='all',
                geofence=True,
                multiprocessing=False 
            )
            print("✅ Model Loaded Successfully")
            self._warmup()
            self.initialized = True
            return True
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False

    def _warmup(self):
        print("🔥 Warming up GPU...")
        try:
            import numpy as np
            from PIL import Image
            dummy_path = os.path.join(BASE_DIR, "warmup.jpg")
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(dummy_path)
            _ = self.model.predict(filepaths=[dummy_path], country="IND", run_mode='single_thread', progress_bars=False)
            if os.path.exists(dummy_path): os.remove(dummy_path)
            print("🔥 Warmup complete")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
            
    # Helper to predict single file (used by detect endpoint)
    def predict(self, filepath):
        if not self.model: return None
        try:
            return self.model.predict(
                filepaths=[filepath],
                country="IND",
                run_mode='single_thread',
                batch_size=1,
                progress_bars=False
            )
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None

model_manager = ModelManager()

# ==========================================
# 5. ROUTES & UTILS
# ==========================================
last_status = {
    "motion": 0, "tilt": 0.0, "gunshot": 0, "temp": None,
    "free_heap": None, "min_heap": None, "rssi": None, "uptime": None
}
last_seen = 0
gunshot_timestamp = 0

def handle_amb82_video(video_file, sample_fps=1, min_conf=0.3, country='IND'):
    filename = secure_filename(f"amb82_{int(time.time())}.mp4")
    file_path = os.path.join(VIDEO_DIR, filename)
    video_file.save(file_path)

    new_video = VideoRecord(filename=filename, filepath=file_path)
    db.session.add(new_video)
    db.session.commit()
    print(f"💾 Video saved to DB: {filename}")

    global video_processor
    if video_processor is None:
        video_processor = BatchVideoProcessor(model_manager)

    try:
        detections = video_processor.process_video_batched(
            file_path, batch_size=8, sample_fps=sample_fps,
            min_confidence=min_conf, country=country
        )

        if detections:
            for d in detections:
                res = DetectionResult(
                    video_id=new_video.id,
                    species=d['species'],
                    confidence=d['confidence'],
                    timestamp_in_video=d['timestamp'],
                    image_url=d.get('image_url')
                )
                db.session.add(res)
        
        new_video.processed = True
        db.session.commit()
        return {"success": True, "video_id": new_video.id, "count": len(detections), "results": detections}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

@app.route('/')
def index(): return render_template('index.html')

@app.route('/sensor')
def sen(): return render_template('sensor.html')

@app.route('/field_unit')
def amb82_analysis(): return render_template('amb82_dashboard.html')

# --- TEST ROUTE FOR SIMPLE FRONTEND ---
@app.route('/test')
def simple_test():
    return render_template('simple_test.html')

## --- IMAGE DETECTION ENDPOINT (FIXED) ---
@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_data = data.get('image')
        if not img_data: return jsonify(success=False, error="No image data"), 400
        
        # 1. Decode base64 image
        if "," in img_data:
            _, encoded = img_data.split(",", 1)
        else:
            encoded = img_data
            
        try:
            binary = base64.b64decode(encoded)
        except Exception:
            return jsonify(success=False, error="Invalid image encoding"), 400
        
        # 2. Save temp file
        filename = f"upload_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as f: 
            f.write(binary)

        # 3. Run Inference
        if model_manager.model:
            result = model_manager.predict(filepath)
            
            # Clean up temp file immediately
            if os.path.exists(filepath): os.remove(filepath)
            
            if result:
                # Parse SpeciesNet result
                predictions = result.get('predictions', {})
                if not predictions: return jsonify(success=False, error="No predictions found"), 500
                
                # --- BUG FIX START ---
                # Check if predictions is a List or a Dictionary
                if isinstance(predictions, list):
                    # It's a list, just take the first item
                    pred_data = predictions[0]
                elif isinstance(predictions, dict):
                    # It's a dictionary, take the first value
                    pred_data = next(iter(predictions.values()))
                else:
                    return jsonify(success=False, error="Unknown prediction format"), 500
                # --- BUG FIX END ---
                
                class_data = pred_data.get("classifications", {})
                if not class_data: return jsonify(success=False, error="No classification data"), 500

                top_class = class_data.get("classes", ["Unknown"])[0]
                top_score = class_data.get("scores", [0])[0]

                # Parse scientific vs common name
                if ";" in top_class:
                    parts = [p.strip() for p in top_class.split(";") if p.strip()]
                    species = parts[-1].title()
                    scientific = parts[-2] if len(parts) >= 2 else species
                else:
                    species = top_class.title()
                    scientific = top_class

                return jsonify({
                    "success": True, 
                    "species": species,
                    "scientific_name": scientific,
                    "confidence": float(top_score)
                })
        
        return jsonify(success=False, error="Model failed to load"), 500

    except Exception as e:
        print(f"❌ Detect Error: {e}")
        # Print the full error to the terminal to help debug
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500
@app.route('/update', methods=['POST'])
def update():
    global last_status, last_seen, gunshot_timestamp
    data = request.json or {}
    print(f"📨 ESP: {data}")
    last_seen = time.time()
    for k in last_status:
        if k in data:
            last_status[k] = data[k]
            if k == "gunshot" and data[k] == 1: gunshot_timestamp = time.time()
    return {"status": "ok"}

@app.route('/status')
def status():
    t = time.time()
    return jsonify({
        **last_status,
        "esp_online": (t - last_seen) < 62,
        "gunshot": 1 if (t - gunshot_timestamp) < 5 else 0,
        "model_loaded": model_manager.model is not None
    })

@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify(success=False), 400
    video_file = request.files['video']
    if video_file.filename == '': return jsonify(success=False), 400
    
    country = request.form.get('country', 'IND')
    response_mode = request.args.get('mode', 'simple')
    
    full_data = handle_amb82_video(video_file, country=country)
    
    if response_mode == 'simple':
        return jsonify({"success": True, "status": "Ack", "id": full_data.get('video_id')}), 200
    else:
        return jsonify(full_data), 200

@app.route('/api/history')
def get_history():
    videos = VideoRecord.query.order_by(VideoRecord.upload_time.desc()).all()
    output = []
    for v in videos:
        dets = [{"species": d.species, "time": d.timestamp_in_video, "image_url": d.image_url} for d in v.detections]
        output.append({
            "id": v.id, "filename": v.filename,
            "time": v.upload_time, "detections": dets
        })
    return jsonify(output)

@app.route('/uploads/videos/<path:filename>')
def serve_video(filename): return send_from_directory(VIDEO_DIR, filename)

@app.route('/static/detections/<path:filename>')
def serve_detections(filename): return send_from_directory(DETECTIONS_DIR, filename)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print("\n🚀 STARTING WILDLIFE SERVER (LOCAL ONLY) 🚀\n")
    
    # Initialize Model
    model_manager.initialize()
    
    print(f"\n{'='*60}")
    print(f"🌐 LOCAL URL: http://127.0.0.1:5000")
    print(f"🌐 TEST URL:  http://127.0.0.1:5000/test")
    print(f"{'='*60}\n")
    
    # Start Flask Server
app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
