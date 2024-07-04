import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
from PIL import Image
import pytesseract
import easyocr

# Load tokenizer and model
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Ensure pytesseract can find the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])

# Function to extract text using OCR
def extract_text_from_image(uploaded_file):
    try:
        # Ensure file is an image and can be processed
        image = Image.open(uploaded_file)
        image_path = f"temp_image.{image.format.lower()}"
        image.save(image_path)

        # Use EasyOCR and Tesseract for OCR
        ocr_result_easyocr = reader.readtext(image_path, detail=0)
        ocr_result_tesseract = pytesseract.image_to_string(image)
        combined_result = ' '.join(ocr_result_easyocr) + ' ' + ocr_result_tesseract
        return combined_result.lower()  # Convert the combined result to lowercase
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return ""

# Function to extract key-value pairs
def extract_key_value(text, key):
    # Tokenize input
    inputs = tokenizer.encode_plus(key, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Truncate input_ids if longer than 512 tokens
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
        inputs["attention_mask"] = inputs["attention_mask"][:, :512]
        inputs["token_type_ids"] = inputs["token_type_ids"][:, :512]

    # Get start and end logits
    outputs = model(input_ids=input_ids, attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"])
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Determine start and end positions
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    return answer

# Streamlit UI
st.title('Key-Value Extractor')

option = st.selectbox("Choose input type", ("Text", "Image"))

if option == "Text":
    text = st.text_area("Enter the text")
elif option == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        text = extract_text_from_image(uploaded_file)
        st.write("Extracted Text:")
        st.write(text)

key = st.text_input("Enter the key")

if st.button('Extract Value'):
    if text and key:
        value = extract_key_value(text, key)
        st.write(f"Extracted {key}: {value}")
    else:
        st.write("Please enter both text and key.")