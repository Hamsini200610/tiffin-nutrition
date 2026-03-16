import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

def test():
    img_path = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\tuning\Indian Food Dataset\Idli\Idli (1).jpg'
    if not os.path.exists(img_path):
        print("Image not found")
        return

    print("Loading detector...")
    detector = pipeline(
        model="google/owlvit-base-patch32", 
        task="zero-shot-object-detection",
        device=-1
    )

    img = Image.open(img_path).convert("RGB")
    label = "idli"
    
    print(f"Detecting {label}...")
    results = detector(
        img,
        candidate_labels=[label, "table"],
        threshold=0.01
    )

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except:
        font = ImageFont.load_default()

    for res in results:
        if res['label'] == label:
            box = res['box']
            score = res['score']
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            
            # Draw white box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="white", width=6)
            
            # Label
            label_text = f"Idli ({int(score*100)}%)"
            t_bbox = draw.textbbox((xmin, ymin-35), label_text, font=font)
            draw.rectangle([t_bbox[0]-5, t_bbox[1]-5, t_bbox[2]+5, t_bbox[3]+5], fill="white")
            draw.text((xmin, ymin-35), label_text, fill="black", font=font)
            print(f"Found {label} at {box} with score {score}")

    out_path = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\test_owlvit_result.jpg'
    img.save(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    test()
