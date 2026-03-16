import os
import json
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageStat
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import uuid
import gc
import numpy as np

# Configuration for stability
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'models', 'cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models', 'cache')
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nutrition_db.json')
CENTROID_PATH = os.path.join(os.path.dirname(__file__), 'data', 'class_centroids.json')

def load_db():
    if not os.path.exists(DB_PATH): return {}
    with open(DB_PATH, 'r') as f:
        return json.load(f)

def load_centroids():
    if not os.path.exists(CENTROID_PATH): return {}
    with open(CENTROID_PATH, 'r') as f:
        return json.load(f)

# Global model pointers
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        print(f"Loading OwlViT model on {DEVICE}...")
        MODEL = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)
        PROCESSOR = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        MODEL.eval()
    return MODEL, PROCESSOR

@torch.no_grad()
def get_box_embedding(image, box):
    try:
        model, processor = get_model()
        xmin, ymin, xmax, ymax = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
        pad = 20
        xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
        xmax, ymax = min(image.width, xmax + pad), min(image.height, ymax + pad)
        if xmax <= xmin or ymax <= ymin: return None
        
        crop = image.crop((xmin, ymin, xmax, ymax)).convert("RGB")
        inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
        vision_outputs = model.owlvit.vision_model(inputs.pixel_values)
        embedding = vision_outputs.pooler_output[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    except:
        return None

def calculate_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    a2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0
    # Also return Intersection over Smallest (IoS) to detect inclusion
    ios = inter / min(a1, a2) if min(a1, a2) > 0 else 0
    return iou, ios

def get_premium_color(key):
    colors = {
        "idli": "#FFFFFF", "dosa": "#FF8C00", "vada": "#8B4513",
        "sambar": "#FF4500", "coffee": "#6F4E37", "chutney": "#90EE90",
        "paratha": "#D2B48C", "poli": "#FFD700", "samosa": "#CD853F",
        "dhokla": "#FFFF00", "kofta": "#DEB887"
    }
    k_low = key.lower()
    for k, v in colors.items():
        if k in k_low: return v
    return "#FFD700"

def analyze_food_image(image_path):
    try:
        db = load_db()
        centroids = load_centroids()
        model, processor = get_model()
        
        original_img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = original_img.size
        # Consistent scale for processing
        img = original_img.resize((800, 800), Image.Resampling.LANCZOS)
        
        physics_map = {
            "masala dosa": "masala dosa roll",
            "dosa": "crispy dosa crepe",
            "idli": "white idli cakes",
            "vada": "medu vada donut",
            "poori": "puffed poori bread",
            "sambar": "bowl of sambar soup",
            "kesari": "kesari sweet",
            "pongal": "pongal porridge",
            "upma": "upma porridge",
            "coconut chutney": "white coconut chutney dip",
            "tomato chutney": "orange tomato chutney dip",
            "chapati": "chapati flatbread",
            "curd": "bowl of curd yogurt",
            "rasam": "rasam soup"
        }


        all_food_keys = list(physics_map.keys())
        background_label = "cloth table plate rim cutlery"

        labels_all = [physics_map[k] for k in all_food_keys] + [background_label]
        raw_results = []
        
        # OwlViT limit: 16 queries per image
        BATCH_SIZE = 12
        
        for i in range(0, len(labels_all), BATCH_SIZE):
            batch_labels = labels_all[i:i + BATCH_SIZE]
            try:
                inputs = processor(text=[batch_labels], images=img, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                target_sizes = torch.Tensor([img.size[::-1]]).to(DEVICE)
                # Sensitivity threshold (0.005) for balance between coverage and noise
                results_batch = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.005)
                
                # Each batch_labels result is at index 0 (single image)
                res = results_batch[0]
                batch_boxes, batch_scores, batch_lbl_ids = res["boxes"], res["scores"], res["labels"]
                
                for b, s, l in zip(batch_boxes, batch_scores, batch_lbl_ids):
                    # Local index to global label index
                    l_idx = int(l)
                    global_idx = i + l_idx
                    
                    if global_idx >= len(all_food_keys): continue 
                    
                    key = all_food_keys[global_idx]
                    box_dict = {
                        "xmin": float(b[0]), "ymin": float(b[1]),
                        "xmax": float(b[2]), "ymax": float(b[3])
                    }
                    box_orig = {
                        "xmin": box_dict["xmin"] * w_orig / 800,
                        "ymin": box_dict["ymin"] * h_orig / 800,
                        "xmax": box_dict["xmax"] * w_orig / 800,
                        "ymax": box_dict["ymax"] * h_orig / 800
                    }
                    raw_results.append({
                        "key": key,
                        "score": float(s),
                        "box": box_orig
                    })
            except Exception as e:
                print(f"Batch Error at {i}: {e}")
                continue

        # Phase 2: Refine with visual embeddings
        refined_results = []
        for p in raw_results:
            key = p['key']
            emb = get_box_embedding(original_img, p['box'])
            final_score = p['score']
            
            if emb is not None and key in centroids:
                centroid = np.array(centroids[key])
                sim = np.dot(emb, centroid)
                # Ultra-Inclusive similarity for challenging lighting
                calibrated_sim = max(0, (sim - 0.15) / 0.75)
                print(f"DEBUG_EMB: {key} sim:{sim:.3f} cal_sim:{calibrated_sim:.3f}")
                # Weighted fusion: Preferring Visual DNA but staying stable
                final_score = (final_score * 0.3) + (calibrated_sim * 0.7)
            
            # Simple color check for paratha vs poli
            xmin, ymin, xmax, ymax = map(int, [p['box']['xmin'], p['box']['ymin'], p['box']['xmax'], p['box']['ymax']])
            crop = original_img.crop((xmin, ymin, xmax, ymax))
            stat = ImageStat.Stat(crop)
            r, g, b = stat.mean[:3]
            std = np.mean(stat.stddev[:3])
            print(f"DEBUG: {key} | R:{r:.1f} G:{g:.1f} B:{b:.1f} Std:{std:.1f} OrigScore:{final_score:.3f}")
            
            # Calibration: Chapati (Dry/Wheat DNA)
            if key == "chapati":
                # Chapatis are dry-toasted (matte).
                # Broaden golden range for wheat (especially on brown paper or wooden plates)
                if 1.0 < (r/b) < 2.0: final_score *= 1.8 
                
                # DNA: Texture variance (allow for more charred spots or uneven lighting)
                if 10 < std < 60: final_score *= 1.4
                
                # If surface is EXTREMELY oily (very high R/B ratio), it's likely a Dosa
                if r > 2.2 * b: final_score *= 0.4
                
                # GLOSS REJECTION: Still reject very bright gloss but be more lenient
                if r > 245 and g > 245 and b > 240: final_score *= 0.1 
                
                # SHAPE: Relaxed aspect ratio for perspective and loose crops
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                if 0.6 < aspect < 1.6: final_score *= 1.5
                else: final_score *= 0.2

            # Calibration: Poori (Puffed 3D Balloon DNA)
            if key == "poori":
                # Poories are smooth but can have some frying bubbles.
                if std < 18: final_score *= 2.0
                elif std > 22: final_score *= 0.1 # Too pitted (Dosa)
                
                # Color: Deep golden brown (Fried oil DNA)
                # Fried items have strong Red and good Green, very low Blue.
                if r > 160 and g > 120 and b < 100: final_score *= 1.5
                
                # Shape: Pillowy/Round
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                if 0.8 < aspect < 1.3: final_score *= 1.5
                else: final_score *= 0.1 

            # Calibration: Idli (Pure White Focus)
            if key == "idli":
                # Idli submerged in Sambar might have some orange tint, but core is white.
                if r > 195 and g > 195 and b > 190: final_score *= 3.0 # Snowy white boost
                
                # REJECTION: Idli is white/cold. If it has ANY golden or brown signature, demote.
                # If Red is significantly higher than Blue, it's NOT an Idli.
                if r > b * 1.15: final_score *= 0.01 
                
                # Balanced color check
                diff = max(abs(r-g), abs(g-b), abs(r-b))
                if diff < 20: final_score *= 1.6
                elif diff > 40: final_score *= 0.05 # Not white
                
                # Shape Guard
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                if 0.8 < aspect < 1.25: final_score *= 1.4
                else: final_score *= 0.1

            # Calibration: Dosa vs Masala Dosa (Structural Roll DNA)
            if "dosa" in key:
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                horizontal = aspect > 1
                
                # DNA: Masala Dosa is a stuffed roll (FATTER). Plain Dosa is a slim roll or flat crepe.
                if key == "masala dosa":
                    # Masala dosa roll is fatter because of the potato stuffing.
                    if horizontal:
                        if 1.4 < aspect < 4.0: final_score *= 2.5 
                        elif aspect > 5.0: final_score *= 0.3 
                    else: # Vertical roll
                        inv_aspect = 1/aspect
                        if 1.4 < inv_aspect < 4.0: final_score *= 2.5
                        elif inv_aspect > 5.0: final_score *= 0.3
                else: # Plain Dosa
                    if horizontal:
                        if aspect > 3.0: final_score *= 2.2 # Slim paper roll boost
                        elif 0.6 < aspect < 1.6: 
                            final_score *= 1.8 # Folded circular/square appearance
                            if std < 15: final_score *= 0.05 
                    else:
                        if (1/aspect) > 3.0: final_score *= 2.2
                        elif 0.6 < (1/aspect) < 1.6: final_score *= 1.6
                
                # Color (Deep Golden Brown - Saturated)
                if r > 1.15 * b: final_score *= 2.5 
                else: final_score *= 0.1
                
                # Texture: Pitted netted DNA (Fermentation marks)
                if 18 < std < 65: final_score *= 2.5
                elif std < 12: final_score *= 0.01 

            # Calibration: Pongal (Soft Mashed Porridge)
            if key == "pongal":
                # Pongal is a soft, uniform porridge (std < 12)
                if std < 12: final_score *= 1.6
                if std > 20: final_score *= 0.05 
                # Color: Pale/Whitish yellow
                if r > 180 and g > 170 and b > 100: final_score *= 1.4
                else: final_score *= 0.2

            # Calibration: Tomato Chutney (Spicy Orange/Red)
            if key == "tomato chutney":
                if std < 18: final_score *= 1.6
                if r > 150 and r > g * 1.3: final_score *= 1.8 
                else: final_score *= 0.1

            # Calibration: Coconut Chutney (Creamy White)
            if key == "coconut chutney":
                diff = max(abs(r-g), abs(g-b), abs(r-b))
                # Creamy white is balanced
                if diff < 25 and r > 175: final_score *= 1.8
                else: final_score *= 0.05

            # Calibration: Upma (Coarse Grain DNA)
            if key == "upma":
                # Upma is matte white AND grainy.
                if r > 165 and g > 165 and b > 155: 
                    # Texture: Upma is grainy (std > 15). Idli is spongy but smoother (std < 12).
                    if 15 < std < 32: final_score *= 1.8
                    elif std < 12: final_score *= 0.1 # Likely idli/chutney
                    else: final_score *= 1.2
                    
                    # Green Shield: Upma often has curry leaves, but shouldn't just be a leaf.
                    if g > r * 1.15: final_score *= 0.01 
                else: 
                    final_score *= 0.1 # Too dark/not white

            # Calibration: Sambar (Liquid Stew DNA)
            if key == "sambar":
                # Liquid/Semi-liquid smooth. Standard Sambars vary in texture.
                if std < 24: final_score *= 1.8 
                # Color: Orange-Red dominance. (Allowing lower R for dim light)
                if r > 110 and r > b * 1.3: final_score *= 1.8
                # Sambar bowl is usually round but allow for some crop variance
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                if 0.7 < aspect < 1.4: final_score *= 1.5
                else: final_score *= 0.5

            # Calibration: Rasam (Thin Soup)
            if key == "rasam":
                if std < 12: final_score *= 1.6
                if r > 110 and r > g * 1.1: final_score *= 1.4
                else: final_score *= 0.1

            # Calibration: Curd (Thick White Liquid)
            if key == "curd":
                if std < 12: final_score *= 1.6
                if r > 200 and g > 200 and b > 200: final_score *= 1.5
                else: final_score *= 0.1

            # Calibration: Kesari (Vibrant Orange Scoop)
            if key == "kesari":
                # Kesari MUST be bright neon orange. Fabric/Towels are usually duller.
                # Red must be very dominant over Blue.
                if r > 180 and r > g * 1.15 and b < 90: final_score *= 3.0
                else: final_score *= 0.001 # HARD demotion for non-neon colors
                # Area and aspect
                area_ratio = (xmax - xmin) * (ymax - ymin) / (w_orig * h_orig)
                if area_ratio > 0.08: final_score *= 0.1 # Kesari is a small side
                aspect = (xmax - xmin) / (ymax - ymin + 1e-6)
                if aspect > 1.5 or aspect < 0.6: final_score *= 0.1

            p['score'] = min(0.99, final_score)
            refined_results.append(p)

        # Phase 3: Spatial Filtering (NMS)
        refined_results = sorted(refined_results, key=lambda x: x['score'], reverse=True)
        final_picks = []
        for p in refined_results:
            # High-Sensitivity thresholds for clinical accuracy
            side_items = ["sambar", "chutney", "rasam", "curd", "coffee"]
            # Main dishes: 0.01 | Side accessories: 0.15
            cutoff = 0.15 if any(s in p['key'] for s in side_items) else 0.01
            
            if p['score'] < cutoff: continue
            
            should_add = True
            for fp in final_picks:
                iou, ios = calculate_iou(p['box'], fp['box'])
                # Reject if moderate IoU (overlap) OR high IoS (contained inside)
                if iou > 0.4 or ios > 0.7:
                    should_add = False
                    break
            if should_add: final_picks.append(p)

        # Fallback: Top-Candidate Backup (Prioritize Food over Background)
        if not final_picks and refined_results:
            print(f"Fallback mode: final_picks empty. Checking {len(refined_results)} refined items.")
            # Filter out background results to find the best food match
            food_only_refined = [r for r in refined_results if r['key'] != background_label]
            if food_only_refined:
                best_refined = max(food_only_refined, key=lambda x: x['score'])
                print(f"Fallback selecting best food item: {best_refined['key']} with score {best_refined['score']:.4f}")
                final_picks.append(best_refined)
            else:
                # If truly only background exists, use the top background match
                best_refined = max(refined_results, key=lambda x: x['score'])
                print(f"Fallback selecting best item (background): {best_refined['key']} with score {best_refined['score']:.4f}")
                final_picks.append(best_refined)
        elif not refined_results:
            print("CRITICAL: No raw detections found at all (raw_results empty).")

        # Phase 4: Draw and Finalize
        draw = ImageDraw.Draw(original_img)
        try: font = ImageFont.truetype("arialbd.ttf", 32)
        except: font = ImageFont.load_default()
        
        final_list = []
        for p in final_picks:
            key = p['key']
            info = db.get(key)
            if not info: continue
            
            display_name = key.title()
            final_list.append({"name": display_name, "confidence": f"{p['score']:.1%}", "nutrition": info})
            
            color = get_premium_color(key)
            box = p['box']
            draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline=color, width=8)
            
            label_text = f"{display_name} ({int(p['score']*100)}%)"
            try:
                t_box = draw.textbbox((box['xmin'], box['ymin']-45), label_text, font=font)
                text_y = box['ymin']-45 if box['ymin'] > 50 else box['ymax']+5
                draw.rectangle([t_box[0]-5, t_box[1]-5, t_box[2]+5, t_box[3]+5], fill=color)
                draw.text((box['xmin'], text_y), label_text, fill="black", font=font)
            except:
                draw.text((box['xmin'], box['ymin']), label_text, fill=color)

        final_uid = uuid.uuid4().hex
        result_filename = f"deep_tuned_{final_uid}.jpg"
        save_path = os.path.join('static', 'uploads', result_filename)
        original_img.save(save_path)

        if not final_list: 
            return {"error": "Identification below certainty. The engine could not find clear food markers. Please ensure the food is well-lit and not obstructed."}

        totals = {"calories": 0, "protein": 0, "carbohydrates": 0, "fats": 0}
        for item in final_list:
            for k in totals: totals[k] += item['nutrition'].get(k, 0)

        return {
            "items": [item["name"] for item in final_list],
            "item_details": final_list,
            "total_nutrition": {k: round(v, 1) for k, v in totals.items()},
            "labeled_image_url": f"/static/uploads/{result_filename}",
            "confidence": "Data-Centric Vision Engine Active"
        }

    except Exception as e:
        return {"error": f"Vision Module Error: {str(e)}"}
    finally:
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
