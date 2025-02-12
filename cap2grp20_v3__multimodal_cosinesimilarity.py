# -*- coding: utf-8 -*-
"""Cap2Grp20_V3__MultiModal_cosinesimilarity.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18vvTC8hpOLFTzy0Pj9P2dn2Bhabed4jN
"""

# Downloading the datasets using wget
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -O Flickr8k_Dataset.zip
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -O Flickr8k_Captions.zip

# Creating directories for extraction
import os
import zipfile

# Paths
images_zip_path = "Flickr8k_Dataset.zip"
captions_zip_path = "Flickr8k_Captions.zip"
images_dir = "Flickr8k_Images"
captions_dir = "Flickr8k_Captions"

# Extracting images
os.makedirs(images_dir, exist_ok=True)
with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
    zip_ref.extractall(images_dir)

# Extracting captions
os.makedirs(captions_dir, exist_ok=True)
with zipfile.ZipFile(captions_zip_path, 'r') as zip_ref:
    zip_ref.extractall(captions_dir)

print("Download and extraction complete.")

!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git

import torch
import clip
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
#from transformers import CLIPProcessor, CLIPModel

# Define paths
flickr8k_images_dir = "/content/Flickr8k_Images/Flicker8k_Dataset"
flickr8k_captions_file = "/content/Flickr8k_Captions/Flickr8k.token.txt"
#model_save_path = "/content/clip_flickr8k.pth"

def idx_bad_images(df):
  bad_idx = []
  for idx, row in df.iterrows():
    image_path = os.path.join(flickr8k_images_dir, row["image"]+".jpg")
    try:
      image = Image.open(image_path).convert("RGB")
    except Exception as e:
      print(f"Error processing {image_path}: {e}")
      bad_idx.append(idx)
  return bad_idx

raw_df = pd.read_csv(flickr8k_captions_file, delimiter="\t", header=None, names=["image", "caption"])

raw_df["image"] = raw_df["image"].str.split(".").str[0]
raw_df.head()

bad_imgs = idx_bad_images(raw_df)
clean_df = raw_df.drop(bad_imgs).reset_index(drop=True)

clean_df.head()

# Load the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

from sklearn.model_selection import train_test_split

# Load the Flickr8k dataset
class Flickr8kDataset(Dataset):
    def __init__(self, image_folder, captions_file, transform=None):
        self.image_folder = image_folder
        self.captions = captions_file #pd.read_csv(captions_file)
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        image_id, caption = row['image'], row['caption']
        image_path = os.path.join(self.image_folder, image_id+ ".jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption

# Load the dataset
image_folder = "/content/Flickr8k_Images/Flicker8k_Dataset"  # Replace with the path to Flickr8k images
captions_file = clean_df # Replace with the path to captions file
dataset = Flickr8kDataset(image_folder, captions_file, transform=preprocess)

# Split into train and test datasets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Compute the symmetric contrastive loss for image-text similarity.
    :param image_features: Normalized image embeddings (batch_size, embedding_dim)
    :param text_features: Normalized text embeddings (batch_size, embedding_dim)
    :param temperature: Temperature parameter for scaling logits
    :return: Contrastive loss
    """
    # Compute similarity scores
    logits_per_image = (image_features @ text_features.T) / temperature
    logits_per_text = logits_per_image.T

    # Create labels (diagonal elements are positive pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=device)

    # Compute cross-entropy loss
    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    loss = (loss_image + loss_text) / 2  # Symmetric loss
    return loss

import torch.optim as optim

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Fine-tune the model
num_epochs = 10
temperature = 0.07  # Temperature parameter for scaling logits

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    for images, captions in train_loader:
        images = images.to(device)
        captions = clip.tokenize(captions).to(device)

        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(captions)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity scores
        logits_per_image = (image_features @ text_features.T) / temperature
        logits_per_text = logits_per_image.T

        # Create labels (diagonal elements are positive pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=device)

        # Compute cross-entropy loss
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2  # Symmetric loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            # Predictions are the indices of the maximum similarity scores
            preds_image = logits_per_image.argmax(dim=-1)
            preds_text = logits_per_text.argmax(dim=-1)

            # Calculate accuracy for image-to-text and text-to-image
            accuracy_image = (preds_image == labels).float().mean()
            accuracy_text = (preds_text == labels).float().mean()
            accuracy = (accuracy_image + accuracy_text) / 2  # Average accuracy

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    # Print epoch statistics
    avg_loss = epoch_loss / len(train_loader)
    avg_accuracy = epoch_accuracy / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_clip.pth")

#from google.colab import files
files.download("fine_tuned_clip.pth")

#  load saved model for testing loop , also print test loss and accuracies

import torch
import clip
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Load the saved model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("fine_tuned_clip.pth", map_location=device))
model.eval()

# Testing loop
test_loss = 0.0
test_accuracy = 0.0
temperature = 0.07

with torch.no_grad():
    for images, captions in test_loader:
        images = images.to(device)
        captions = clip.tokenize(captions).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(captions)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (image_features @ text_features.T) / temperature
        logits_per_text = logits_per_image.T

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=device)

        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        preds_image = logits_per_image.argmax(dim=-1)
        preds_text = logits_per_text.argmax(dim=-1)

        accuracy_image = (preds_image == labels).float().mean()
        accuracy_text = (preds_text == labels).float().mean()
        accuracy = (accuracy_image + accuracy_text) / 2

        test_loss += loss.item()
        test_accuracy += accuracy.item()

avg_test_loss = test_loss / len(test_loader)
avg_test_accuracy = test_accuracy / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")



# code to generate requirements file

!pip freeze > requirements.txt





!pip install gradio

# prompt: provide code to create Gradio UI which takes images and retrieves most similiar caption also takes caption and retrieves most similar image

import gradio as gr
import torch
import clip
from PIL import Image

# Load the fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("fine_tuned_clip.pth", map_location=device))
model.eval()

# Function to find the most similar caption for an image
def find_similar_caption(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Placeholder for text embeddings and captions (replace with actual data)
    # In real application load your pre-calculated text embeddings from file
    text_features = torch.randn(len(test_dataset), 512).to(device) # Replace with your actual embeddings
    captions = [f"Caption {i}" for i in range(len(test_dataset))] # Replace with your actual captions


    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    return captions[indices[0]]

# Function to find the most similar image for a caption
def find_similar_image(caption):
    text = clip.tokenize(caption).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Placeholder for image embeddings (replace with your actual data)
    # Load your pre-calculated embeddings instead of calculating them here.
    image_features = torch.randn(len(test_dataset), 512).to(device) #replace with your image embeddings

    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    # Placeholder for image paths. Replace with the actual paths from your dataset.
    image_paths = [f"/content/Flickr8k_Images/Flicker8k_Dataset/image_{i}.jpg" for i in range(len(test_dataset))]  #replace with your actual image paths

    return image_paths[indices[0]] #return the image_path


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image-Caption Similarity Search")
    with gr.Tab("Image to Caption"):
        image_input = gr.Image(type="pil")
        caption_output = gr.Textbox(label="Most Similar Caption")
        submit_btn1 = gr.Button("Find Caption")
        submit_btn1.click(fn=find_similar_caption, inputs=image_input, outputs=caption_output)
    with gr.Tab("Caption to Image"):
        caption_input = gr.Textbox(label="Enter Caption")
        image_output = gr.Image(type="filepath") #Use filepath for display
        submit_btn2 = gr.Button("Find Image")
        submit_btn2.click(fn=find_similar_image, inputs=caption_input, outputs=image_output)
demo.launch()











