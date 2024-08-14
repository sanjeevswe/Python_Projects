import cv2
import pytesseract
import numpy as np

def detect_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use edge detection to find potential plate regions
    edged = cv2.Canny(gray, 170, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and look for potential plates
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # Assume the plate is a rectangle
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate = image[y:y + h, x:x + w]

            # Use OCR to read text from the plate
            text = pytesseract.image_to_string(plate)
            print("Detected License Plate Number:", text)
            cv2.imshow("Plate Detected", plate)
            cv2.waitKey(0)
