import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
import json
import tqdm

# Constants
ROOT = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker'
DATASET_ROOT = os.path.join(ROOT, 'static', 'uploads', 'tuning', 'Indian Food Dataset')
EMBEDDINGS_PATH = os.path.join(ROOT, 'data', 'class_centroids.json')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_centroids():
    print(f"Loading OwlViT model on {DEVICE}...")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model.eval()

    centroids = {}
    classes = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    for class_name in classes:
        print(f"Generating centroid for {class_name}...")
        class_dir = os.path.join(DATASET_ROOT, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take up to 30 images to keep it fast but representative
        images = images[:30]
        class_embeddings = []

        for fname in tqdm.tqdm(images):
            img_path = os.path.join(class_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    # Get image features from the backbone
                    vision_outputs = model.owlvit.vision_model(inputs.pixel_values)
                    # Use the pooler output or the mean of patch tokens
                    # pooler_output is (batch, hidden_size)
                    emb = vision_outputs.pooler_output[0].cpu().numpy()
                    class_embeddings.append(emb)
            except Exception as e:
                continue
        
        if class_embeddings:
            centroid = np.mean(class_embeddings, axis=0)
            # Normalize for cosine similarity
            centroid = centroid / np.linalg.norm(centroid)
            centroids[class_name.lower()] = centroid.tolist()

    with open(EMBEDDINGS_PATH, 'w') as f:
        json.dump(centroids, f)
    print(f"Centroids saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    generate_centroids()
