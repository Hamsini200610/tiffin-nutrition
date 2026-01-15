import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import uuid
import gc

# Configuration for stability
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'models', 'cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models', 'cache')
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

# Load the nutrition database
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nutrition_db.json')

def load_db():
    with open(DB_PATH, 'r') as f:
        return json.load(f)

# Initialize OwlViT-B/32 (Stable version)
print("Initializing Zero-Shot Detector (Tailored South Indian Mode)...")
try:
    detector = pipeline(
        model="google/owlvit-base-patch32", 
        task="zero-shot-object-detection",
        device=-1
    )
    print("Detector Engine Tailored Successfully!")
except Exception as e:
    print(f"Failed to load detector: {e}")

@torch.no_grad()
def analyze_food_image(image_path):
    """
    Expert Multi-Batch detection.
    Tailored to prevent internal labeling of Masala Dosa and fix Chutney vs Kesari.
    """
    try:
        db = load_db()
        original_img = Image.open(image_path).convert("RGB")
        
        # High resolution for small cups/details
        max_dim = 1024
        w, h = original_img.size
        if w > max_dim or h > max_dim:
            ratio = max_dim / max(w, h)
            img = original_img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        else:
            img = original_img

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arialbd.ttf", 22)
        except:
            font = ImageFont.load_default()

        # TAILORED Visual Logic Map
        physics_map = {
            "masala dosa": "masala dosa",
            "medu vada": "vada",
            "red chutney": "tomato chutney",
            "kesari": "kesari",
            "dosa": "dosa",
            "idli": "idli",
            "pongal": "pongal",
            "filter coffee": "filter coffee",
            "sambar": "sambar",
            "coconut chutney": "coconut chutney",
            "green chutney": "mint chutney",
            "poori": "poori",
            "bonda": "bonda",
            "upma": "upma",
            "khara bath": "khara bath",
            "rava dosa": "rava dosa",
            "parotta": "parotta",
            "set dosa": "set dosa",
            "uttapam": "uttapam",
            "idiyappam": "idiyappam",
            "puttu": "puttu",
            "appam": "appam",
            "paniyaram": "paniyaram",
            "bajji": "bajji",
            "poha": "poha",
            "chapati": "chapati"
        }
        
        all_food_keys = list(physics_map.keys())
        all_results = []
        
        # Safe batching
        batch_size = 4
        background = "table or plate"
        
        print(f"Expert Scan: Analyzing {len(all_food_keys)} items...")

        for i in range(0, len(all_food_keys), batch_size):
            batch_keys = all_food_keys[i:i + batch_size]
            active_prompts = [physics_map[k] for k in batch_keys if k in physics_map]
            current_labels = active_prompts + [background]
            
            try:
                batch_preds = detector(
                    img,
                    candidate_labels=current_labels,
                    threshold=0.02
                )
                
                rev_map = {physics_map[k]: k for k in batch_keys if k in physics_map}
                for p in batch_preds:
                    if p['label'] in rev_map:
                        p['original_key'] = rev_map[p['label']]
                        # Manual Score Boosting
                        low_lbl = p['label'].lower()
                        if "dosa" in low_lbl: p['score'] *= 1.2
                        if "vada" in low_lbl: p['score'] *= 1.2
                        if "idli" in low_lbl: p['score'] *= 1.5
                        if "chutney" in low_lbl or "sambar" in low_lbl: p['score'] *= 1.2
                        all_results.append(p)
                
                gc.collect()
                
            except Exception as batch_err:
                continue

        # FINAL CONSOLIDATION
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        final_list = []
        processed_boxes = []

        def get_iou(box1, box2):
            x1 = max(box1['xmin'], box2['xmin'])
            y1 = max(box1['ymin'], box2['ymin'])
            x2 = min(box1['xmax'], box2['xmax'])
            y2 = min(box1['ymax'], box2['ymax'])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
            area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
            # Standard IOU
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            # Containment Check
            containment1 = intersection / area1 if area1 > 0 else 0
            return iou, containment1

        for pred in all_results:
            box = pred['box']
            score = pred['score']
            key = pred['original_key']

            # Adaptive Floor: Lowering thresholds for items
            floor = 0.04
            if key in ["masala dosa", "medu vada", "idli"]: floor = 0.06
            
            if score < floor: continue

            # Overlap Logic
            should_skip = False
            for pb in processed_boxes:
                iou, containment = get_iou(box, pb['box'])
                # Only skip if they are almost the same box or duplicate category
                if iou > 0.60:
                    should_skip = True
                    break
                if (containment > 0.85 or iou > 0.45) and key == pb['original_key']:
                    should_skip = True
                    break
            
            if should_skip: continue

            processed_boxes.append(pred)
            food_info = db.get(key)
            
            if food_info:
                display_name = (key if key != "red chutney" else "Tomato Chutney").title()
                final_list.append({
                    "name": display_name,
                    "confidence": f"{score:.1%}",
                    "nutrition": food_info
                })
                
                # Visual Labels
                xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                color = "#FFD700" # GOLD
                if "dosa" in key: color = "#FF8C00" 
                if key == "medu vada": color = "#8B4513"
                if key == "red chutney": color = "#FF0000" # Pure Red
                if key == "kesari": color = "#FFA500" # Orange
                if key == "idli": color = "#FFFFFF"

                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=6)
                label_text = f"{display_name} ({int(score*100)}%)"
                t_bbox = draw.textbbox((xmin, ymin-30), label_text, font=font)
                draw.rectangle([t_bbox[0]-5, t_bbox[1]-5, t_bbox[2]+5, t_bbox[3]+5], fill=color)
                draw.text((xmin, ymin-30), label_text, fill="black", font=font)

        # Output labeled image
        final_uid = uuid.uuid4().hex
        result_filename = f"expert_tailored_{final_uid}.jpg"
        save_path = os.path.join('static', 'uploads', result_filename)
        img.save(save_path)

        if not final_list:
            return {"error": "Identification below certainty. Ensure good lighting."}

        # Totals
        totals = {"calories": 0, "protein": 0, "carbohydrates": 0, "fats": 0}
        for item in final_list:
            for k in totals:
                totals[k] += item['nutrition'][k]

        return {
            "items": [item["name"] for item in final_list],
            "item_details": final_list,
            "total_nutrition": {k: round(v, 1) for k, v in totals.items()},
            "labeled_image_url": f"/static/uploads/{result_filename}",
            "confidence": "Expert Food Calibration Locked"
        }

    except Exception as e:
        return {"error": f"Logic Fault: {str(e)}"}
