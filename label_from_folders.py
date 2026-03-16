import os
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(__file__)
TUNING_ROOT = os.path.join(ROOT, 'static', 'uploads', 'tuning', 'Indian Food Dataset')
GT_OUT = os.path.join(ROOT, 'static', 'uploads', 'tuning', 'groundtruth')
DB_PATH = os.path.join(ROOT, 'data', 'nutrition_db.json')

os.makedirs(GT_OUT, exist_ok=True)

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def safe_font(size=24):
    try:
        return ImageFont.truetype('arialbd.ttf', size)
    except Exception:
        return ImageFont.load_default()

def main():
    db = load_db()
    added = {}
    labels_map = {}
    font = safe_font(28)

    if not os.path.exists(TUNING_ROOT):
        print(f"Tuning dataset folder not found: {TUNING_ROOT}")
        return

    for class_name in sorted(os.listdir(TUNING_ROOT)):
        class_dir = os.path.join(TUNING_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        out_dir = os.path.join(GT_OUT, class_name)
        os.makedirs(out_dir, exist_ok=True)

        # ensure nutrition_db has entry for this class (lowercase key)
        key = class_name.strip().lower()
        if key not in db:
            db[key] = {
                'calories': 0,
                'protein': 0,
                'carbohydrates': 0,
                'fats': 0
            }
            added[key] = db[key]

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            src = os.path.join(class_dir, fname)
            try:
                im = Image.open(src).convert('RGB')
            except Exception as e:
                print(f"Failed to open {src}: {e}")
                continue

            draw = ImageDraw.Draw(im)
            text = class_name.replace('_', ' ')
            # draw a semi-opaque box and text
            w, h = im.size
            try:
                tb = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                try:
                    text_w, text_h = font.getsize(text)
                except Exception:
                    text_w, text_h = (len(text) * 8, 16)
            pad = 8
            box = [5, 5, 5 + text_w + pad*2, 5 + text_h + pad*2]
            draw.rectangle(box, fill=(0,0,0,180))
            draw.text((5 + pad, 5 + pad), text, fill=(255,255,255), font=font)

            out_path = os.path.join(out_dir, fname)
            try:
                im.save(out_path)
            except Exception as e:
                print(f"Failed to save {out_path}: {e}")
                continue

            rel = os.path.relpath(out_path, GT_OUT)
            labels_map[rel] = [key]

    if added:
        save_db(db)
        print(f"Added placeholder nutrition entries for {len(added)} items: {', '.join(sorted(added.keys()))}")
    else:
        print("No missing nutrition entries found.")

    # write labels.json next to groundtruth
    labels_file = os.path.join(GT_OUT, 'labels.json')
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels_map, f, indent=2, ensure_ascii=False)

    print(f"Ground-truth labeled images written to {GT_OUT}. Labels file: {labels_file}")

if __name__ == '__main__':
    main()
