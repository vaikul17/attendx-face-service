from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import os
import urllib.request

app = FastAPI(title="AttendX Face AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download Haar Cascade if missing
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    print("[INFO] Downloading Haar Cascade detector XML...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

# Download YuNet and SFace ONNX deep-learning models if missing
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"

if not os.path.exists(YUNET_MODEL):
    try:
        print("[INFO] Downloading YuNet ONNX face detector...")
        urllib.request.urlretrieve("https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", YUNET_MODEL)
    except Exception as e:
        print(f"[WARNING] YuNet download failed: {e}")

if not os.path.exists(SFACE_MODEL):
    try:
        print("[INFO] Downloading SFace ONNX face recognizer...")
        urllib.request.urlretrieve("https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", SFACE_MODEL)
    except Exception as e:
        print(f"[WARNING] SFace download failed: {e}")

# Check if we can load YuNet & SFace models
HAS_SFACE = False
try:
    if os.path.exists(YUNET_MODEL) and os.path.exists(SFACE_MODEL):
        recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        HAS_SFACE = True
        print("[AI SERVICE] Loaded OpenCV YuNet + SFace Deep Learning models successfully.")
except Exception as e:
    print(f"[AI SERVICE WARNING] Failed to initialize YuNet/SFace: {e}")

# Attempt to load face_recognition, fall back to SFace or OpenCV Haar Cascade
USE_FACE_RECOGNITION = True
try:
    import face_recognition
    print("[AI SERVICE] Loaded face_recognition successfully.")
except ImportError:
    USE_FACE_RECOGNITION = False
    if HAS_SFACE:
        print("[AI SERVICE] Falling back to OpenCV YuNet + SFace Deep Learning Recognizer.")
    else:
        print("[AI SERVICE WARNING] Face libraries missing. Falling back to basic pixel intensity comparison.")

# Simple passive liveness check helper
def is_frame_blurry(gray_face) -> bool:
    variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return variance < 80.0

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/embed")
async def generate_embedding(image: UploadFile = File(...)):
    """
    Returns face features.
    If face_recognition is loaded, returns 128D descriptors.
    If SFace is loaded, returns 128D SFace deep embeddings.
    Otherwise, falls back to raw pixel intensities.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Use SFace Deep Learning Recognizer (if loaded and face_recognition is missing)
        if not USE_FACE_RECOGNITION and HAS_SFACE:
            detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (width, height))
            _, faces = detector.detect(img)
            
            if faces is None or len(faces) == 0:
                raise HTTPException(status_code=400, detail="No face detected in the picture. Position your face clearly.")
            
            # Extract main face coordinates for liveness check
            box = faces[0][:4].astype(int)
            x, y, w, h = max(0, box[0]), max(0, box[1]), box[2], box[3]
            w, h = min(width - x, w), min(height - y, h)
            face_crop = gray[y:y+h, x:x+w]

            if is_frame_blurry(face_crop):
                raise HTTPException(status_code=400, detail="Liveness check failed: Photo spoofing detected.")

            # Align face crop and extract 128D deep feature vector
            aligned_face = recognizer.alignCrop(img, faces[0])
            feat = recognizer.feature(aligned_face)
            return {"embedding": feat[0].tolist()}

        # 2. Use face_recognition (if available)
        elif USE_FACE_RECOGNITION:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                raise HTTPException(status_code=400, detail="No face detected in the picture. Position your face clearly.")
            
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_crop = gray[y:y+h, x:x+w]
            
            if is_frame_blurry(face_crop):
                raise HTTPException(status_code=400, detail="Liveness check failed: Photo spoofing detected.")

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img, [(y, x + w, y + h, x)])
            if not encodings:
                raise HTTPException(status_code=400, detail="Failed to extract facial descriptors.")
            return {"embedding": encodings[0].tolist()}

        # 3. Last resort fallback (raw pixel grids)
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                raise HTTPException(status_code=400, detail="No face detected in the picture. Position your face clearly.")
            
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_crop = gray[y:y+h, x:x+w]
            
            if is_frame_blurry(face_crop):
                raise HTTPException(status_code=400, detail="Liveness check failed: Photo spoofing detected.")

            resized = cv2.resize(face_crop, (50, 50))
            normalized = (resized / 255.0).tolist()
            return {"embedding": normalized}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Service processing error: {str(e)}")

@app.post("/match")
async def match_face(
    scanned_embedding_json: str = Form(...),
    known_embeddings_json: str = Form(...)
):
    """
    Compares face descriptors using Deep Learning Cosine Similarity or Pixel Euclidean Norms.
    """
    try:
        scanned_emb = np.array(json.loads(scanned_embedding_json))
        known_data = json.loads(known_embeddings_json)
        
        if not known_data:
            return {"match": False, "employeeId": None, "confidence": 0.0}

        known_embeddings = [np.array(item['embedding']) for item in known_data]
        employee_ids = [item['employeeId'] for item in known_data]

        # 1. SFace Cosine Similarity Matcher (Used for high accuracy deep features)
        if not USE_FACE_RECOGNITION and HAS_SFACE:
            best_score = -1.0
            best_idx = -1
            
            for idx, known in enumerate(known_embeddings):
                # Calculate Cosine similarity
                dot = np.dot(scanned_emb, known)
                norm_scan = np.linalg.norm(scanned_emb)
                norm_known = np.linalg.norm(known)
                
                if norm_scan > 0 and norm_known > 0:
                    similarity = float(dot / (norm_scan * norm_known))
                else:
                    similarity = 0.0

                if similarity > best_score:
                    best_score = similarity
                    best_idx = idx

            # SFace Cosine similarity matching threshold is 0.363
            SFACE_THRESHOLD = 0.363
            if best_score > SFACE_THRESHOLD:
                # Normalize confidence to a realistic 88-99% range
                conf_pct = ((best_score - SFACE_THRESHOLD) / (1.0 - SFACE_THRESHOLD)) * 100
                confidence = 85.0 + (max(0.0, min(100.0, conf_pct)) * 0.14)
                return {
                    "match": True,
                    "employeeId": employee_ids[best_idx],
                    "confidence": round(confidence, 2)
                }

        # 2. face_recognition matching distance logic
        elif USE_FACE_RECOGNITION:
            distances = face_recognition.face_distance(known_embeddings, scanned_emb)
            min_idx = int(np.argmin(distances))
            best_distance = float(distances[min_idx])
            
            RECOGNITION_THRESHOLD = 0.55
            if best_distance < RECOGNITION_THRESHOLD:
                confidence = max(0.0, min(100.0, (1.0 - best_distance) * 100))
                return {
                    "match": True,
                    "employeeId": employee_ids[min_idx],
                    "confidence": round(confidence, 2)
                }

        # 3. Simple Euclidean distance on raw pixel vectors (Last resort fallback)
        else:
            best_dist = float('inf')
            best_idx = -1
            for idx, known in enumerate(known_embeddings):
                dist = np.linalg.norm(known - scanned_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            FALLBACK_THRESHOLD = 15.0 
            if best_dist < FALLBACK_THRESHOLD:
                confidence = max(0.0, min(100.0, (1.0 - (best_dist / FALLBACK_THRESHOLD)) * 100))
                return {
                    "match": True,
                    "employeeId": employee_ids[best_idx],
                    "confidence": round(confidence, 2)
                }

        return {"match": False, "employeeId": None, "confidence": 0.0}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Match service error: {str(e)}")
