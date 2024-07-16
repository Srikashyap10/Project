import streamlit as st
import cv2
import json
from PIL import Image
import pytesseract
import numpy as np
import os
from matplotlib import pyplot as plt

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

# Function to enhance the image
def enhance_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    return enhanced

# Function to perform OCR
def ocr_core(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to extract and save info
def extract_and_save_info(ocr_text):
    info = {"OCR_Text": ocr_text}
    return info

# Function to process image
def process_image(image_path, output_image_path):
    try:
        enhanced_image = enhance_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB))
        ocr_text = ocr_core(pil_image)
        info = extract_and_save_info(ocr_text)
        cv2.imwrite(output_image_path, enhanced_image)
        return enhanced_image, info
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        return None, None

# Streamlit UI
st.title("Image Enhancing for Data Extraction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = f"temp_{uploaded_file.name}"
    image.save(image_path)
    output_image_path = f"output_{uploaded_file.name}"

    # Process the uploaded image
    original_image = cv2.imread(image_path)
    enhanced_image, info = process_image(image_path, output_image_path)

    if enhanced_image is not None:
        # Display original and enhanced images
        st.image(original_image, caption='Original Image', use_column_width=True)
        st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

        # Display extracted information
        st.write("Extracted Information:")
        st.json(info)

        # Provide a download link for the enhanced image
        with open(output_image_path, "rb") as file:
            btn = st.download_button(
                label="Download enhanced image",
                data=file,
                file_name=output_image_path,
                mime="image/png"
            )

        # Provide a download link for the extracted info JSON
        output_json_path = f"extracted_info_{uploaded_file.name}.json"
        with open(output_json_path, 'w') as f:
            json.dump(info, f, indent=4)
        with open(output_json_path, "rb") as file:
            btn = st.download_button(
                label="Download extracted info",
                data=file,
                file_name=output_json_path,
                mime="application/json"
            )

    os.remove(image_path)
    os.remove(output_image_path)
