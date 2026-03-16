import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import tqdm
import sys

# Constants
ROOT = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker'
DATASET_ROOT = os.path.join(ROOT, 'static', 'uploads', 'tuning', 'Indian Food Dataset')
OUTPUT_ROOT = os.path.join(ROOT, 'static', 'labeled_dataset')
DEVICE = 0 if torch.cuda.is_available() else -1
LOG_FILE = os.path.join(ROOT, 'labeling_log.txt')

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)

def safe_font(size=24):
    try:
        return ImageFont.truetype('arialbd.ttf', size)
    except Exception:
        try:
            return ImageFont.truetype('arial.ttf', size)
        except:
            return ImageFont.load_default()

def process_dataset():
    if not os.path.exists(DATASET_ROOT):
        log(f"Dataset not found at {DATASET_ROOT}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    log(f"Initializing OwlViT-B/32 on {'GPU' if DEVICE == 0 else 'CPU'}...")
    try:
        detector = pipeline(
            model="google/owlvit-base-patch32", 
            task="zero-shot-object-detection",
            device=DEVICE
        )
    except Exception as e:
        log(f"CRITICAL: Failed to load detector: {e}")
        return

    font = safe_font(24)
    
    classes = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    log(f"Found classes: {classes}")

    for class_name in classes:
        log(f"\nProcessing class: {class_name}")
        class_dir = os.path.join(DATASET_ROOT, class_name)
        out_class_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for fname in tqdm.tqdm(images, desc=f"Labeling {class_name}"):
            src_path = os.path.join(class_dir, fname)
            out_path = os.path.join(out_class_dir, fname)

            if os.path.exists(out_path):
                continue

            try:
                img = Image.open(src_path).convert("RGB")
                w, h = img.size
                # Resize if too large to avoid memory issues
                if w > 1024 or h > 1024:
                    ratio = 1024 / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                
                prompt = class_name.lower().replace('_', ' ')
                
                results = detector(
                    img,
                    candidate_labels=[prompt, "plate", "table", "container"],
                    threshold=0.08
                )

                draw = ImageDraw.Draw(img)
                food_results = [r for r in results if r['label'] == prompt]
                
                if food_results:
                    food_results = sorted(food_results, key=lambda x: x['score'], reverse=True)
                    for res in food_results[:3]: # Draw top 3 items
                        box = res['box']
                        score = res['score']
                        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                        
                        # Draw high-quality white box
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="white", width=4)
                        
                        # Box Label
                        label_text = f"{class_name} ({int(score*100)}%)"
                        try:
                            # Use textbbox for precise placement
                            t_bbox = draw.textbbox((xmin, ymin - 35), label_text, font=font)
                            # Shift if it exceeds top
                            if t_bbox[1] < 0:
                                t_bbox = draw.textbbox((xmin, ymax + 5), label_text, font=font)
                                text_y = ymax + 5
                            else:
                                text_y = ymin - 35
                                
                            draw.rectangle([t_bbox[0]-5, t_bbox[1]-5, t_bbox[2]+5, t_bbox[3]+5], fill="white")
                            draw.text((t_bbox[0], text_y), label_text, fill="black", font=font)
                        except:
                            draw.text((xmin, ymin - 25), label_text, fill="white")

                img.save(out_path)
                
                # Cleanup to save memory
                del img
                if DEVICE == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                log(f"Error processing {fname}: {e}")

    log(f"\nSuccess! Labeled images saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_dataset()
