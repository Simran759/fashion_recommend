import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
from io import BytesIO


import gdown  # You may need to add this to your requirements.txt

# Google Drive file IDs
embed_id = "1uPjeV28ViLMP54Aqa2VgB_kaiPXXQeK7"
name_id = "173YdpN3d3vMNq0FjnXLP4TCiAzU_kePE"

@st.cache_resource
def download_and_load_pickle(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"/tmp/{file_id}.pkl"
    gdown.download(url, output, quiet=False)
    with open(output, 'rb') as f:
        return pickle.load(f)

# Load files
st.title("👗 Fashion Recommendation System")

try:
    feature_list = np.array(download_and_load_pickle(embed_id))
    filenames = download_and_load_pickle(name_id)
    st.success("✅ Pickle files loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading pickle files: {e}")



# --- Load ResNet50 model ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# --- Extract features directly from uploaded PIL image ---
def feature_extraction_pil(pil_img, model):
    img = pil_img.resize((224, 224))  # Ensure size
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    return normalized

# --- Find similar products ---
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image to get recommendations", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image from memory without saving
    img = Image.open(BytesIO(uploaded_file.read())).convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract features and recommend
    with st.spinner('Analyzing and finding recommendations...'):
        features = feature_extraction_pil(img, model)
        indices = recommend(features, feature_list)

    st.subheader("Recommended Products:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(filenames[indices[0][i]], use_container_width=True)
