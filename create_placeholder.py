import numpy as np
import cv2
import os

# Create directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Create a blank image (640x480) with dark gray background
img = np.ones((480, 640, 3), dtype=np.uint8) * 40  # Dark gray background

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Fire and Smoke Detection', (120, 180), font, 1.2, (255, 100, 50), 2)
cv2.putText(img, 'Upload a video to begin', (160, 240), font, 1, (200, 200, 200), 2)
cv2.putText(img, 'Supported formats: MP4, AVI, MOV', (140, 300), font, 0.8, (200, 200, 200), 2)

# Add a fire icon (simple triangle)
pts = np.array([[320, 100], [280, 160], [360, 160]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (0, 100, 255))

# Save the image
cv2.imwrite('static/placeholder.jpg', img)
print("Placeholder image created at static/placeholder.jpg") 