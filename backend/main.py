from io import BytesIO
import os
import re
import threading
import base64
import traceback
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
import torch.nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules.conv
import torch.nn.modules.conv
import torch.nn.modules.batchnorm
import torch.nn.modules.activation
import ultralytics.nn.modules.block
import torch.nn.modules.container
import ultralytics.nn.modules.block
import torch.nn.modules.pooling
import torch.nn.modules.upsampling
import ultralytics.nn.modules.head
from typing import List
import easyocr
from utils import extract_license_plate
import logging
from contextlib import asynccontextmanager
from typing import List, Union
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug flag - when True, saves plate crops to disk for inspection
DEBUG_SAVE_CROPS = True
DEBUG_SAVE_DIR = "debug_crops"
if DEBUG_SAVE_CROPS and not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

# Pillow 10+ compatibility fix
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Allowlist the model for unpickling (required PyTorch >=2.6)
torch.serialization.add_safe_globals([
    DetectionModel,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.container.ModuleList,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.Concat,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.SPPF,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    ultralytics.nn.modules.head.Detect,
    ultralytics.nn.modules.block.DFL,
    ultralytics.utils.IterableSimpleNamespace
])

# Initialize FastAPI app with lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_reader
    try:
        ocr_reader = easyocr.Reader(['en'])
        logger.info("OCR Reader loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load OCR Reader at startup: {e}")
        ocr_reader = None

    yield
    

app = FastAPI(
    title="Helmet Violation Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------ FONTS ------
try:
    font = ImageFont.truetype("arial.ttf", 40)
except IOError:
    font = ImageFont.load_default()

# Global model cache and lock for thread-safe loading
loaded_models = {}
models_lock = threading.Lock()
ocr_reader = None  # set in lifespan

def get_model(model_type: str, model_name: str):
    """
    Load the YOLO model for the given model_name if not loaded yet,
    return cached model if already loaded.
    Uses a namespaced cache key so helmet:best and license_plate:best don't collide.
    """
    global loaded_models

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    


    key = f"{model_type}:{model_name}"
    with models_lock:
        if key in loaded_models:
            return loaded_models[key]

        #model_path = f"models/{model_type}/{model_name}.pt"
        model_path = os.path.join(BASE_DIR, "models", "helmet", "best.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            # Special case: license plate model
            if model_type.lower() in ["license_plate", "licenseplate", "lp"]:
                model = YOLO(model_path, task="detect")
            else:
                model = YOLO(model_path)  # normal load for helmet models
            loaded_models[key] = model
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None

@app.get("/model_status")
async def model_status():
    loaded = {name: True for name in loaded_models.keys()}
    ocr_loaded = ocr_reader is not None
    return {"models_loaded": loaded, "ocr_loaded": ocr_loaded}

@app.get("/")
async def root():
    return {"message": "Helmet Violation Detection API is running"}

# ------ DETECTION HELPERS ------
def box_within(inner, outer):
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def select_nearest_to_center(candidate_inds, boxes, center_ref):
    if len(candidate_inds) == 1:
        return candidate_inds[0]
    dists = [distance(box_center(boxes[i]), center_ref) for i in candidate_inds]
    return candidate_inds[int(np.argmin(dists))]

# --- STEP 1: Image Preprocessing (for OCR) ---
def preprocess_plate_crop_for_ocr(plate_crop: Image.Image) -> np.ndarray:
    """
    Enhance and binarize number plate image for better OCR accuracy.
    Returns a uint8 numpy array ready for EasyOCR.
    """
    img = cv2.imread(plate_crop)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise (bilateral)
    denoised = cv2.bilateralFilter(thresh, 11, 30, 30)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(denoised, -1, kernel)
    # Upsample to 2x
    upsampled = cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return upsampled

# --- STEP 2: Pattern-based Filtering & tolerant corrections ---
def extract_indian_plate_candidates_from_texts(texts):
    """
    Extract strings resembling Indian number plates using pattern matching.
    Returns the best candidate or empty string.
    """
    if not texts:
        return ""

    patterns = [
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}",  # KA05AB1234
        r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}",      # DL10CB5678
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{2,4}",  # TN7AB56
    ]

    # A small mapping to correct common OCR mistakes in characters (applied heuristically)
    correction_map = str.maketrans({
        'O': '0',  # O -> 0
        'Q': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        # note: be conservative — aggressive corrections can introduce false positives
    })

    best_candidate = ""
    best_score = 0

    for txt in texts:
        if not txt:
            continue
        raw = str(txt).upper()
        # remove non-alnum
        clean = re.sub(r"[^A-Z0-9]", "", raw)
        # attempt also a corrected version
        corrected = clean.translate(correction_map)

        for candidate in (clean, corrected):
            for pat in patterns:
                match = re.fullmatch(pat, candidate)
                if match:
                    # scoring: longer + more digits preferred
                    score = len(candidate) + sum(c.isdigit() for c in candidate) * 2
                    if score > best_score:
                        best_candidate = candidate
                        best_score = score

    return best_candidate

# --- STEP 3: End-to-End OCR & Plate Text Extraction ---
def read_plate_text_and_raw(ocr_reader_obj, plate_crop: Image.Image):
    """
    Return (plate_text, raw_ocr_joined_text).
    Runs preprocessing, EasyOCR, and candidate extraction.
    """
    try:
        preprocessed_np = preprocess_plate_crop_for_ocr(plate_crop)

        if DEBUG_SAVE_CROPS:
            # useful for debugging; saves the np array as an image
            debug_idx = len(os.listdir(DEBUG_SAVE_DIR)) + 1 if os.path.exists(DEBUG_SAVE_DIR) else 1
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"plate_{debug_idx}.png"), preprocessed_np)

        # EasyOCR accepts numpy images (grayscale or RGB)
        ocr_result = ocr_reader_obj.readtext(preprocessed_np, detail=0, paragraph=False)

        # ocr_result is a list of strings (since detail=0)
        raw_joined = " ".join(ocr_result).strip()
        plate_candidate = extract_indian_plate_candidates_from_texts(ocr_result)

        # if not found, try more aggressive single string correction
        if not plate_candidate and raw_joined:
            joined = re.sub(r"[^A-Z0-9]", "", raw_joined.upper())
            plate_candidate = extract_indian_plate_candidates_from_texts([joined])

        return plate_candidate, raw_joined

    except Exception as e:
        logger.warning(f"OCR read error: {e}")
        return "", ""

@app.post("/detect")
async def detect_violations(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    helmet_model: str = "best",
    license_model: str = "best"
):
    """
    Detect helmet violations and license plates in uploaded image.
    Combines YOLO detection + OCR reading in one endpoint.
    """
    try:
        # --- Pillow Compatibility (for v10+) ---
        if not hasattr(Image, 'Resampling'):
            Image.Resampling = Image
        if not hasattr(Image, 'ANTIALIAS'):
            Image.ANTIALIAS = Image.Resampling.LANCZOS

        # --- Load YOLO Models ---
        yolo_model = get_model("helmet", helmet_model)
        if not yolo_model:
            raise HTTPException(status_code=500, detail=f"Helmet model '{helmet_model}' not loaded")

        lp_model = get_model("license_plate", license_model)
        if not lp_model:
            raise HTTPException(status_code=500, detail=f"License plate model '{license_model}' not loaded")

        if ocr_reader is None:
            raise HTTPException(status_code=500, detail="OCR reader not loaded")

        # --- Read Uploaded Image ---
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # --- Run YOLO Helmet Detection ---
        results = yolo_model(img)
        output_img = img.copy()
        draw = ImageDraw.Draw(output_img)
        detections = []

        if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            raise HTTPException(status_code=400, detail="No detections found")

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        mask = confidences >= confidence
        boxes, classes, confidences = boxes[mask], classes[mask], confidences[mask]
        CLASS_NAMES = ['NumberPlate', 'Person', 'Helmet', 'Head', 'Motorbike']

        # --- Helper: Validate Indian Plate ---
        def clean_plate_text(raw_text):
            txt = raw_text.upper()
            # Map common OCR mistakes
            txt = txt.translate(str.maketrans({
                'O': '0',
                'Q': '0',
                'I': '1',
                'L': '1',
                'S': '5',
                'B': '8',
                '+': '1',
                'Z': '2',
            }))
            # Remove all non-alphanumeric
            txt = re.sub(r'[^A-Z0-9]', '', txt)
            return txt

        def valid_indian_plate(text):
            pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$')
            return bool(pattern.match(text))

        def extract_indian_plate(raw_texts):
            """
            Try to find an Indian format plate from multiple OCR outputs.
            """
            for raw in raw_texts:
                cleaned = clean_plate_text(raw)
                if valid_indian_plate(cleaned):
                    return cleaned
            # If no strict match, try partial match
            for raw in raw_texts:
                cleaned = clean_plate_text(raw)
                if len(cleaned) >= 6 and re.match(r'[A-Z]{2}\d{1,2}', cleaned):
                    return cleaned
            return ""

        # --- Iterate through all Motorbikes ---
        motorbike_inds = [i for i, cid in enumerate(classes) if CLASS_NAMES[cid] == 'Motorbike']
        for m_idx in motorbike_inds:
            m_box = boxes[m_idx]
            m_center = box_center(m_box)
            m_conf = confidences[m_idx]
            helmet_status = "No Helmet"
            helmet_color = "red"
            helmet_conf = None

            # --- Helmet Check ---
            for i, cid in enumerate(classes):
                if CLASS_NAMES[cid] == 'Helmet' and box_within(boxes[i], m_box):
                    helmet_status = "Helmet"
                    helmet_color = "green"
                    helmet_conf = confidences[i]
                    break

            # --- Draw Helmet Label ---
            draw.rectangle(m_box, outline="red", width=3)
            label_text = f"{helmet_status} ({helmet_conf:.2f})" if helmet_conf else f"{helmet_status} ({m_conf:.2f})"
            draw.text((m_box[0], m_box[1] - 45), label_text, fill=helmet_color, font=font)

            # --- License Plate Detection ---
            numberplate_inds = [i for i, cid in enumerate(classes)
                                if CLASS_NAMES[cid] == 'NumberPlate' and box_within(boxes[i], m_box)]
            number_plate_text = ""
            np_conf = None

            if numberplate_inds:
                np_idx = select_nearest_to_center(numberplate_inds, boxes, m_center)
                np_box = boxes[np_idx]
                np_conf = confidences[np_idx]

                # Crop plate region and refine using LP model
                plate_crop = img.crop([int(x) for x in np_box])
                lp_results = lp_model(plate_crop)

                if (lp_results and hasattr(lp_results[0], 'boxes') and
                    lp_results[0].boxes is not None and len(lp_results[0].boxes.xyxy) > 0):
                    lp_boxes = lp_results[0].boxes.xyxy.cpu().numpy()
                    largest_idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in lp_boxes])
                    fine_plate_crop = plate_crop.crop([int(x) for x in lp_boxes[largest_idx]])
                else:
                    fine_plate_crop = plate_crop

                # --- OCR Phase (Enhanced) ---
                gray = fine_plate_crop.convert("L")
                gray = ImageOps.autocontrast(gray)
                gray = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
                np_plate = np.array(gray)

                try:
                    easy_texts = [r[1] for r in ocr_reader.readtext(np_plate)]
                    tess_text = pytesseract.image_to_string(np_plate, config="--psm 7 --oem 3").strip()
                    all_texts = easy_texts + [tess_text]
                    candidate = extract_indian_plate(tess_text)
                    number_plate_text = easy_texts
                    if not candidate:
                        logger.warning(f"No valid Indian plate found. Raw OCR: {all_texts}")
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    number_plate_text = ""

            detections.append({
                "helmet": helmet_status,
                "helmet_confidence": round(float(helmet_conf or m_conf), 2),
                "number_plate_text": number_plate_text,
                "number_plate_confidence": round(float(np_conf or 0), 2),
                "motorbike_box": [float(x) for x in m_box.tolist()]
            })

        # --- Convert to Base64 ---
        annotated_cv = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_cv)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "detections": detections,
            "annotated_image": annotated_base64,
            "total_detections": len(detections),
            "violations": len([d for d in detections if d["helmet"] == "No Helmet"])
        })

    except Exception as e:
        logger.error(f"Detection error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list", response_model=Union[List[str], dict])
async def list_models(type: str = Query(..., description="Type of model: helmet or license_plate")):
    base_dir = "models"
    models_dir = os.path.join(base_dir, type)

    try:
        if not os.path.exists(models_dir):
            return {"error": f"Model type '{type}' not found."}

        # List all .pt files (without extension)
        files = [
            os.path.splitext(f)[0]
            for f in os.listdir(models_dir)
            if os.path.isfile(os.path.join(models_dir, f)) and f.endswith(".pt")
        ]
        return files

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
