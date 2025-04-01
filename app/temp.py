import streamlit as st
import torch
import clip
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from groq import Groq
from io import BytesIO

import streamlit as st
import qrcode
import tempfile
import requests


UPLOAD_URL="http://192.168.0.185:5000"
data_dir = "../data"  # Adjust path as needed
classes_file = os.path.join(data_dir, "classes.txt")
attributes_file = os.path.join(data_dir, "predicates.txt")

# Load classes
with open(classes_file, "r") as f:
    classes = [line.split("\t")[1].strip() for line in f.readlines()]

# Load attributes
with open(attributes_file, "r") as f:
    attributes = [line.split("\t")[1].strip() for line in f.readlines()]

# Load attribute matrix
# Load attribute matrix with continuous values (assuming it's formatted correctly)
attribute_matrix_file = os.path.join(data_dir, "predicate-matrix-continuous.txt")

attribute_matrix = pd.read_csv(attribute_matrix_file, sep=r'\s+', header=None)
attribute_matrix.columns = attributes
attribute_matrix.index = classes

# Check the matrix
# print(f"Attribute Matrix Shape: {attribute_matrix.shape}")
# print(attribute_matrix.head())


# # Check loaded data
# print(f"Loaded {len(classes)} classes and {len(attributes)} attributes.")
# print("Sample Classes:", classes[:5])
# print("Sample Attributes:", attributes[:5])
# print(f"Attribute Matrix Shape: {attribute_matrix.shape}")


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model_clip, preprocess = clip.load("ViT-B/32", device=device)

def generate_class_embeddings(classes):
    class_embeddings = {}
    with torch.no_grad():
        for class_name in classes:
            text_tokens = clip.tokenize([class_name]).to(device)
            embedding = model_clip.encode_text(text_tokens).cpu().numpy()
            class_embeddings[class_name] = embedding
    return class_embeddings

class_embeddings = generate_class_embeddings(classes)


def attribute_to_embedding_incremental(attributes, attribute_matrix, threshold=50, max_tokens=77):
    attribute_embeddings = {}
    with torch.no_grad():
        for index, class_name in enumerate(attribute_matrix.index):
            # Retrieve the class's attribute vector
            attribute_vector = attribute_matrix.iloc[index]

            # Select attributes based on the threshold (ignore -1 values)
            selected_attributes = [
                attributes[i] for i, val in enumerate(attribute_vector) if val >= threshold and val != -1
            ]

            # Tokenize selected attributes
            tokenized_attributes = {attr: clip.tokenize([attr])[0] for attr in selected_attributes}

            valid_attributes = []
            total_tokens = 0
            for attr, tokens in tokenized_attributes.items():
                if total_tokens + len(tokens) <= max_tokens:
                    valid_attributes.append(attr)
                    total_tokens += len(tokens)
                else:
                    break

            # Create a descriptive sentence for the attributes
            attribute_description = f"This animal is described as {', '.join(valid_attributes)}."
            text_tokens = clip.tokenize([attribute_description]).to(device)
            embedding = model_clip.encode_text(text_tokens).cpu().numpy()
            attribute_embeddings[class_name] = embedding
    return attribute_embeddings

attribute_embeddings = attribute_to_embedding_incremental(attributes, attribute_matrix)

def align_embeddings(class_embeddings, attribute_embeddings):
    combined_embeddings = {}
    for class_name in class_embeddings.keys():
        visual_emb = class_embeddings[class_name]
        text_emb = attribute_embeddings[class_name]
        
        # Combine embeddings (keeping visual and attribute embeddings separate or modifying based on continuous data)
        combined_emb = np.concatenate((visual_emb, text_emb), axis=1)  # This is fine for now

        combined_embeddings[class_name] = combined_emb
    return combined_embeddings

combined_embeddings = align_embeddings(class_embeddings, attribute_embeddings)
visual_embeddings = {class_name: embedding.flatten()[:512] for class_name, embedding in combined_embeddings.items()}

def generate_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)  # Preprocess directly
    with torch.no_grad():
        image_embedding = model_clip.encode_image(image).cpu().numpy().flatten()
    return image_embedding



# Modify zero-shot classification to return top 5 predictions
def zero_shot_classification(image_embedding, visual_embeddings, class_names, top_k=5):
    similarities = []
    
    for class_name, visual_embedding in visual_embeddings.items():
        similarity = np.dot(image_embedding, visual_embedding) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(visual_embedding)
        )
        similarities.append((class_name, similarity))

    # Sort and get top_k predictions
    top_predictions = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return top_predictions

def get_active_attributes(normalized_attribute_matrix, attributes, class_name, threshold=50):
    class_index = attribute_matrix.index.get_loc(class_name)
    attribute_vector = attribute_matrix.iloc[class_index]
    
    # Include only attributes with a value above the threshold
    active_attributes = [attributes[i] for i in range(len(attribute_vector)) if attribute_vector.iloc[i] >= threshold]
    
    return active_attributes


def plot_top_predictions(predictions):
    class_names, scores = zip(*predictions)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(class_names[::-1], scores[::-1], color="skyblue")  # Reverse for correct ordering

    plt.xlabel("Similarity Score")
    plt.title("Top 5 Predictions")
    plt.xlim(0, 1)  # Cosine similarity range
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add score values on bars
    for bar, score in zip(bars, scores[::-1]):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{score:.4f}",
                 va='center', ha='left', fontsize=12, color="black")

    # Display the plot in Streamlit
    st.pyplot(plt)


with open("categorized_attributes.json", "r") as file:
    categorized_attributes = json.load(file)

class DynamicQASystem:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key="api-key")
        self.model = model

    def dynamic_question_answering(self, context, animal_class, question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in 100 - 200 words."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_completion_tokens=200
        )

        return response.choices[0].message.content.strip()

qa_system = DynamicQASystem()


# Function to format lists with "and" before the last item
def format_list(traits):
    if len(traits) > 1:
        return ", ".join(traits[:-1]) + " and " + traits[-1]
    return traits[0]

def plot_top_predictions(predictions):
    class_names, scores = zip(*predictions)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(class_names[::-1], scores[::-1], color="skyblue")  # Reverse for correct ordering

    plt.xlabel("Similarity Score")
    plt.title("Top 5 Predictions")
    plt.xlim(0, 1)  # Cosine similarity range
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add score values on bars
    for bar, score in zip(bars, scores[::-1]):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{score:.4f}",
                 va='center', ha='left', fontsize=12, color="black")

    # Display the plot in Streamlit
    st.pyplot(plt)


def generate_qr_code(upload_url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(upload_url)
    qr.make(fit=True)

    img = qr.make_image(fill="black", back_color="white")

    # Save QR Code to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)
    
    return temp_file.name


# Streamlit App
st.title("Interactive Kiosk System")
st.header("Zero-Shot Image Classification with Dynamic Q&A")

st.subheader("Upload Image via Mobile")
qr_code_path = generate_qr_code(UPLOAD_URL)
st.image(qr_code_path, caption="Scan to Upload", width=250)


st.subheader("Fetch Latest Uploaded Image")
# Store the image URL and context in session state
if "image_url" not in st.session_state:
    st.session_state.image_url = None

if "context_string" not in st.session_state:
    st.session_state.context_string = None

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

if "img" not in st.session_state:
    st.session_state.img=None

if "top5" not in st.session_state:
    st.session_state.top5=None

if st.button("Fetch Image"):
    response = requests.get(f"{UPLOAD_URL}/latest-image")
    
    if response.status_code == 200:
        data = response.json()
        image_url = f"{UPLOAD_URL}{data['url']}"
        st.session_state.image_url = image_url
        
        # Fetch and process the image as if it were uploaded
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            image = Image.open(BytesIO(img_response.content))
            st.session_state.img=image
            st.image(image, caption="Processed Image", use_column_width=True)
            
            # Now you can use this image for embedding and classification
            image_embedding = generate_image_embedding(image)
            top_predictions = zero_shot_classification(image_embedding, visual_embeddings, classes)
            predicted_class, score = top_predictions[0]
            st.session_state.top5=top_predictions
            
            st.session_state.predicted_class = predicted_class
            
            active_attributes = get_active_attributes(attribute_matrix, attributes, predicted_class)

            active_attributes_categorized = {
                category: [attr for attr in active_attributes if attr in attributes]
                for category, attributes in categorized_attributes.items()
            }

            animal_name = predicted_class

            category_templates = {
                "Physical Traits": f"{animal_name.capitalize()} has these physical traits: {{traits}}.",
                "Behavior": f"These attributes define {animal_name}'s behavior: {{traits}}.",
                "Habitat": f"It inhabits places with these characteristics: {{traits}}.",
                "Diet": f"Its diet consists of: {{traits}}.",
                "Appearance": f"{animal_name.capitalize()} has these distinctive appearance traits: {{traits}}.",
                "Other Characteristics": f"Some additional characteristics of {animal_name} are: {{traits}}."
            }

            sentence_parts = []
            for category, traits in active_attributes_categorized.items():
                if traits:
                    traits_str = format_list(traits)
                    sentence = category_templates.get(category, "{} has these traits: {{traits}}.").format(traits=traits_str)
                    sentence_parts.append(sentence)
            
            # Combine all sentences into a final context string
            context_string = " ".join(sentence_parts)
            
            # Save context string to session state
            st.session_state.context_string = context_string

            st.subheader(f"Predicted Class: {predicted_class}")
            st.write(f"Similarity Score: {score:.4f}")
        else:
            st.warning("Failed to load image from the URL.")
    else:
        st.warning("No image uploaded yet!")

# Check if context_string and predicted_class are in session state
if st.session_state.context_string and st.session_state.predicted_class:
    st.subheader("Ask a Question about the Animal")
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question.strip():
            answer = qa_system.dynamic_question_answering(
                st.session_state.context_string,
                st.session_state.predicted_class,
                user_question
            )
            predicted_class, score = st.session_state.top5[0]
            st.image(st.session_state.img, caption="Processed Image", use_column_width=True)
            st.subheader(f"Predicted Class: {predicted_class}")
            st.write(f"Similarity Score: {score:.4f}")
            st.write(f"**Q:** {user_question}")
            st.write(f"**A:** {answer}")
            plot_top_predictions(st.session_state.top5)
        else:
            st.warning("Please enter a question.")



# Footer
st.write("Developed as part of the Zero-Shot Image Classification Project.")
