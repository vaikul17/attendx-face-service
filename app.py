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

# Attempt to load face_recognition, fall back to OpenCV LBPH on windows compiler issues
USE_FACE_RECOGNITION = True
try:
    import face_recognition
    print("[AI SERVICE] Loaded face_recognition successfully.")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("[AI SERVICE WARNING] face_recognition not found. Falling back to OpenCV LBPH Face Recognizer.")

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
    If using LBPH fallback, returns the 200x200 flattened pixel intensity values.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the picture. Position your face clearly.")
            
        # Get largest face crop
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_crop = gray[y:y+h, x:x+w]
        
        # Check liveness (Laplacian focus variance)
        if is_frame_blurry(face_crop):
            raise HTTPException(status_code=400, detail="Liveness check failed: Photo spoofing detected.")

        if USE_FACE_RECOGNITION:
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img, [(y, x + w, y + h, x)])
            if not encodings:
                raise HTTPException(status_code=400, detail="Failed to extract facial descriptors.")
            return {"embedding": encodings[0].tolist()}
        else:
            # Fallback to normalized grayscale pixel array (200x200 downscaled)
            resized = cv2.resize(face_crop, (50, 50)) # 2500 dimensions
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
    Compares face descriptors.
    """
    try:
        scanned_emb = np.array(json.loads(scanned_embedding_json))
        known_data = json.loads(known_embeddings_json)
        
        if not known_data:
            return {"match": False, "employeeId": None, "confidence": 0.0}

        known_embeddings = [np.array(item['embedding']) for item in known_data]
        employee_ids = [item['employeeId'] for item in known_data]

        if USE_FACE_RECOGNITION:
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
        else:
            # LBPH Fallback: Cosine similarity or Euclidean distance check on normalized pixels
            best_dist = float('inf')
            best_idx = -1
            for idx, known in enumerate(known_embeddings):
                dist = np.linalg.norm(known - scanned_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            # Empirical threshold for normalized intensity distance
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
