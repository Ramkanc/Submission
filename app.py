import streamlit as st
import torch
import clip
from PIL import Image
import os

# Load your fine-tuned CLIP model (or the pre-trained one for demonstration)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model.load_state_dict(torch.load("fine_tuned_clip.pth", map_location=device))  # Load if you have a saved model
model.eval()

# Sample image directory (replace with your actual image folder)
image_dir = "path/to/your/images"  
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Function to get image features
def get_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

# Precompute features for sample images (optional for speed)
image_features_dict = {}
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image_features_dict[image_file] = get_image_features(image_path)

# Function to retrieve similar images based on caption
def get_similar_images(caption, top_k=5):
    text_input = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    
    similarities = {}
    for image_file, image_features in image_features_dict.items():
        similarity = torch.cosine_similarity(text_features, image_features).item()
        similarities[image_file] = similarity
    
    sorted_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return [os.path.join(image_dir, img[0]) for img in sorted_images[:top_k]]

# Streamlit UI
st.title("Image Captioning and Retrieval (Simplified)")

# Option to enter a caption
caption_input = st.text_input("Enter a caption:")
if caption_input:
    similar_images = get_similar_images(caption_input)
    st.write("Similar Images:")
    for image_path in similar_images:
        st.image(Image.open(image_path), caption=os.path.basename(image_path))