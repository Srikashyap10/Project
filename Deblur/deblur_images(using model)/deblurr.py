import streamlit as st
import os
import cv2
import json
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

# Pre-trained FSRCNN model path
model_path = r"C:/Users/kashy/Downloads/FSRCNN_x4 (1).pb"

if not os.path.exists(model_path):
    # Download the model file if it does not exist
    url = 'https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/fsrcnn_x4.pb'
    import urllib.request
    urllib.request.urlretrieve(url, model_path)

def enhance_image_with_fsrcnn(image):
    # Load the FSRCNN model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)  # Use FSRCNN with scale factor 4

    # Enhance the image
    enhanced_image = sr.upsample(image)
    return enhanced_image

# Function to read OCR
def ocr_core(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to extract and save OCR text
def extract_and_save_info(ocr_text, image_name, output_path):
    info = {
        "OCR_Text": ocr_text,
    }

    # Save info to a JSON file
    output_json_path = os.path.join(output_path, f"extracted_info_{image_name}.json")
    with open(output_json_path, 'w') as f:
        json.dump(info, f, indent=4)

    return info

# Process a single image
def process_image(image_path, output_image_path, output_path):
    try:
        # Read the image
        original_image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        enhanced_image = enhance_image_with_fsrcnn(gray_image)

        pil_image = Image.fromarray(enhanced_image)
        ocr_text = ocr_core(pil_image)

        # Save the enhanced image
        cv2.imwrite(output_image_path, enhanced_image)

        image_name = os.path.basename(image_path)
        extract_and_save_info(ocr_text, image_name, output_path)

        return original_image, enhanced_image, ocr_text

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

# Streamlit UI
def main():
    st.title("Image Enhancement and OCR Extraction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = uploaded_file.name
        output_path = "./output"
        os.makedirs(output_path, exist_ok=True)
        output_image_path = os.path.join(output_path, f"enhanced_{image_path}")

        # Save the uploaded file to disk
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        original_image, enhanced_image, ocr_text = process_image(image_path, output_image_path, output_path)

        if original_image is not None and enhanced_image is not None:
            st.image(original_image, caption='Original Image', use_column_width=True)
            st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
            st.text_area("OCR Text", ocr_text)

if __name__ == "__main__":
    main()