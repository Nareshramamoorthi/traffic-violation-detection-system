import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from paddleocr import PaddleOCR
from inference_sdk import InferenceHTTPClient

# Initialize OCR and API client
ocr = PaddleOCR(use_angle_cls=True, lang='en')
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="OqzVg3h0ZTssMowCV2Pg"
)

# Load YOLO model
model = YOLO("mohan.pt")  # Replace with trained YOLO model

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    violations = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame for efficiency
            continue
        
        results = model(frame)
        helmet_violation_detected = False
        license_plate_text = ""
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0])
                
                if class_id == 1 and confidence > 0.5:  # License plate class
                    cropped_plate = frame[y1:y2, x1:x2]
                    cv2.imwrite("cropped_plate.jpg", cropped_plate)
                    result = CLIENT.infer("cropped_plate.jpg", model_id="license-plate-character-extraction/2")
                    if 'predictions' in result and result['predictions']:
                        sorted_predictions = sorted(result['predictions'], key=lambda x: (x['y'], x['x']))
                        extracted_text = ''.join([pred['class'] for pred in sorted_predictions])
                        license_plate_text = extracted_text
                
                if class_id == 2 and confidence > 0.5:  # No helmet class
                    helmet_violation_detected = True
        
        if helmet_violation_detected and license_plate_text:
            violation_entry = f"{license_plate_text} - Not Wearing Helmet"
            violations.append(violation_entry)
            print("🚨 Violation Detected:", violation_entry)
    
    cap.release()
    
    with open("violations.txt", "w") as f:
        for entry in violations:
            f.write(entry + "\n")
    
    return "\n".join(violations)

# Gradio UI
def gradio_interface(video_file):
    return process_video(video_file)

ui = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="📂 Upload Video"),
    outputs=gr.Textbox(label="📄 Detected Violations"),
    title="Helmet & Number Plate Detection System",
    description="🚗 Upload a video to detect helmet violations and extract number plates!"
)

ui.launch()





# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load image (bullet.jpg)
# image_path = "bullet.jpg"
# image = cv2.imread(image_path)

# # Check if image is loaded correctly
# if image is None:
#     print("❌ Error: Image not found.")
#     exit()

# # === 🔥 Advanced Image Preprocessing for OCR 🔥 ===
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# # 🔹 Histogram Equalization to improve contrast
# gray = cv2.equalizeHist(gray)

# # 🔹 Gaussian Blur (removes noise while keeping edges)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# # 🔹 Adaptive Thresholding for better text extraction
# adaptive_thresh = cv2.adaptiveThreshold(
#     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpened_image = cv2.filter2D(adaptive_thresh, -1, kernel)

# # Save processed image
# cv2.imwrite("processed_bullet.jpg", sharpened_image)

# # === 🔥 OCR using PaddleOCR 🔥 ===
# result = ocr.ocr(sharpened_image, cls=True)

# # Check if OCR result is None or empty
# if result and isinstance(result, list):
#     plate_number = ""
#     for line in result:
#         if line:  # Check if line is not None or empty
#             for word in line:
#                 plate_number += word[1][0] + " "  # Extract detected text

#     plate_number = plate_number.strip().replace("\n", "").replace("\f", "")
#     print("🚗 Detected Text from bullet.jpg:", plate_number)
# else:
#     print("❌ OCR failed to detect any text.")
