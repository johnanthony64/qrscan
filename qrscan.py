import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set high resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize QRCodeDetector
qr_detector = cv2.QRCodeDetector()

# Preprocessing function to enhance QR detection
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpen the image
    sharpened_frame = cv2.filter2D(blurred_frame, -1, kernel)
    return sharpened_frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Scale up the frame for better small object detection
    scale_factor = 2
    resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Detect and decode QR code using OpenCV
    preprocessed_frame = preprocess_frame(resized_frame)
    data, bbox, _ = qr_detector.detectAndDecode(preprocessed_frame)

    if bbox is not None:
        bbox = (bbox / scale_factor).astype(int)  # Scale bbox back to original size
        for i in range(len(bbox)):
            point1 = tuple(bbox[i][0])
            point2 = tuple(bbox[(i + 1) % len(bbox)][0])
            cv2.line(frame, point1, point2, (0, 255, 0), 3)  # Draw bounding box

        if data:
            print(f"Detected QR Code: {data}")
            cv2.putText(frame, data, (bbox[0][0][0], bbox[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Fallback using pyzbar if OpenCV fails
    if not data:
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            points = obj.polygon
            if points:
                pts = [(int(point.x), int(point.y)) for point in points]
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)
            # Display decoded data
            barcode_data = obj.data.decode('utf-8')
            print(f"Decoded (pyzbar): {barcode_data}")
            cv2.putText(frame, barcode_data, (obj.rect.left, obj.rect.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the live feed with annotations
    cv2.imshow('Enhanced QR Code Scanner', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
