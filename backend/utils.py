import cv2
import numpy as np
from PIL import Image
import easyocr
import re

def process_image(image_bytes):
    """
    Process uploaded image bytes and return opencv image
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def extract_license_plate(image, rider_bbox, ocr_reader):
    """
    Extract license plate from rider detection area
    """
    try:
        x1, y1, x2, y2 = rider_bbox
        
        # Expand search area for license plate (typically below the rider)
        plate_search_area = image[int(y2):int(y2+100), int(x1):int(x2)]
        
        if plate_search_area.size == 0:
            return None
            
        # Use OCR to detect text in the search area
        results = ocr_reader.readtext(plate_search_area)
        
        for (bbox, text, confidence) in results:
            # Filter for license plate patterns (customize based on your region)
            if confidence > 0.5 and len(text) >= 6:
                # Adjust bbox coordinates to original image coordinates
                adjusted_bbox = [
                    int(x1 + bbox[0][0]), int(y2 + bbox[0][1]),
                    int(x1 + bbox[2][0]), int(y2 + bbox[2][1])
                ]
                
                return {
                    "number_plate_bbox": adjusted_bbox,
                    "plate_text": text.strip(),
                    "plate_confidence": confidence
                }
        
        return None
        
    except Exception as e:
        print(f"License plate extraction error: {e}")
        return None

def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on image
    """
    annotated = image.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        status = detection["status"]
        confidence = detection["confidence"]
        
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if status == "helmet" else (0, 0, 255)
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{status}: {confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1-10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated

def validate_license_plate(text):
    """
    Validate license plate format (customize based on your region)
    """
    # Example pattern for Indian license plates
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, text.replace(' ', ''))
