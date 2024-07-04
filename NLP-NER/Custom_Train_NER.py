import spacy
from spacy.training import Example
import random
import streamlit as st
from PIL import Image
import pytesseract
import easyocr
import io

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


# Function to train NER model
def train_ner_model():
    nlp = spacy.blank('en')  # Use a blank spaCy model
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    TRAIN_DATA = [
        # Reel Data Examples
        ("TYPE: SS90100E_R1_00001", {"entities": [(6, 15, "TYPE")]}),
        ("MARKING: G10", {"entities": [(9, 12, "MARKING")]}),
        ("PART NO: 4E214034", {"entities": [(9, 17, "PART_NO")]}),
        ("LOT NO: 2815CE297", {"entities": [(8, 16, "LOT_NO")]}),
        ("QTY: 3000", {"entities": [(5, 9, "QUANTITY")]}),
        ("TYPE: AS5601_B3_45567", {"entities": [(6, 15, "TYPE")]}),
        ("MARKING: K20", {"entities": [(9, 12, "MARKING")]}),
        ("PART NO: 9F567123", {"entities": [(9, 17, "PART_NO")]}),
        ("LOT NO: 9876GH123", {"entities": [(8, 16, "LOT_NO")]}),
        ("QTY: 1500", {"entities": [(5, 9, "QUANTITY")]}),

        # Resume Data Examples
        ("NAME: John Doe", {"entities": [(6, 14, "NAME")]}),
        ("EMAIL: johndoe@example.com", {"entities": [(7, 26, "EMAIL")]}),
        ("PHONE: 123-456-7890", {"entities": [(7, 19, "PHONE")]}),
        ("NAME: Jane Smith", {"entities": [(6, 16, "NAME")]}),
        ("EMAIL: janesmith@sample.net", {"entities": [(7, 25, "EMAIL")]}),
        ("PHONE: 987-654-3210", {"entities": [(7, 19, "PHONE")]}),

        # Medical Data Examples
        ("SYMPTOM: Fever", {"entities": [(9, 14, "SYMPTOM")]}),
        ("DIAGNOSIS: Influenza", {"entities": [(11, 20, "DIAGNOSIS")]}),
        ("PRESCRIPTION: Amoxicillin 500mg", {"entities": [(14, 32, "PRESCRIPTION")]}),
        ("SYMPTOM: Cough", {"entities": [(9, 14, "SYMPTOM")]}),
        ("DIAGNOSIS: Common Cold", {"entities": [(11, 22, "DIAGNOSIS")]}),
        ("PRESCRIPTION: Paracetamol 500mg", {"entities": [(14, 33, "PRESCRIPTION")]}),
        ("SYMPTOM: Headache", {"entities": [(9, 17, "SYMPTOM")]}),
        ("DIAGNOSIS: Migraine", {"entities": [(11, 18, "DIAGNOSIS")]}),
        ("PRESCRIPTION: Ibuprofen 200mg", {"entities": [(14, 30, "PRESCRIPTION")]}),
    ]

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    train_examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        entities = annotations['entities']
        try:
            # Check alignment
            tags = spacy.training.offsets_to_biluo_tags(doc, entities)
            example = Example.from_dict(doc, annotations)
            train_examples.append(example)
        except Exception as e:
            print(f"Error with text: {text}\n{str(e)}")
            continue

    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        for i in range(30):
            random.shuffle(train_examples)
            batches = spacy.util.minibatch(train_examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.5, sgd=optimizer)

    nlp.to_disk("ner_model")  # Save model to disk
    return nlp


# Function to classify the type of dataset
def classify_dataset(text):
    if any(keyword in text.lower() for keyword in ["type:", "marking:", "part no:", "lot no:", "qty:"]):
        return "SMART_REEL"
    elif any(keyword in text.lower() for keyword in ["name:", "email:", "phone:", "skills"]):
        return "resume"
    elif any(keyword in text.lower() for keyword in ["symptom:", "diagnosis:", "prescription:"]):
        return "medical"
    else:
        return "unknown"


# Function to display named entities based on dataset type
def display_entities(doc, dataset_type):
    key_value_pairs = {}
    if (dataset_type == "SMART_REEL"):
        relevant_entities = {'TYPE': "TYPE", 'PART_NO': "PART_NO", 'LOT_NO': "LOT_NO", 'MARKING': "MARKING",
                             'QUANTITY': "QUANTITY"}
        extracted_entities = []
        for ent in doc.ents:
            if ent.label_ in relevant_entities:
                extracted_entities.append(f"Entity: {ent.text}, Label: {relevant_entities[ent.label_]}")
                if relevant_entities[ent.label_] in key_value_pairs:
                    key_value_pairs[relevant_entities[ent.label_]].append(ent.text)
                else:
                    key_value_pairs[relevant_entities[ent.label_]] = [ent.text]
        return extracted_entities, key_value_pairs
    return [], key_value_pairs


# Function to search for a key in the extracted entities
def search_for_key(key_value_pairs, query_key):
    query_key_lower = query_key.lower()
    for key, values in key_value_pairs.items():
        if key.lower() == query_key_lower:
            return values
    return None


# Initialize the NER model
nlp = train_ner_model()

# Streamlit UI
st.title("OCR and NER Processing")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        combined_ocr_text = extract_text_from_image(uploaded_file)
        if combined_ocr_text:
            st.subheader("Extracted Text:")
            st.write(combined_ocr_text)

            dataset_type = classify_dataset(combined_ocr_text)
            st.write(f"Detected dataset type: {dataset_type}")

            doc = nlp(combined_ocr_text)
            extracted_entities, key_value_pairs = display_entities(doc, dataset_type)

            if extracted_entities:
                st.subheader("Extracted Entities:")
                for entity in extracted_entities:
                    st.write(entity)

                query_key = st.text_input("Enter the key you want to search for:")

                if query_key:
                    values = search_for_key(key_value_pairs, query_key)
                    if values:
                        st.write(f"Values for '{query_key}': {', '.join(values)}")
                    else:
                        st.write(f"Key '{query_key}' not found.")
            else:
                st.write("No relevant entities found.")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")