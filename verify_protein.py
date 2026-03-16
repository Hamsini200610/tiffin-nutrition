import os
import json
from detector import analyze_food_image

def verify():
    dataset_root = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\tuning\Indian Food Dataset'
    samples = {
        "Aloo Paratha": "Paratha (1).jpg",
        "Dhokla": "Dhokla (1).jpg",
        "Malai Kofta": "Malai_Kofta (1).jpg",
        "Puran Poli": "Puran_Poli (1).jpg",
        "Samosa": "Samosa (1).jpg"
    }

    print(f"{'Food Item':<20} | {'Detected':<20} | {'Protein (g)':<12}")
    print("-" * 56)

    for class_name, img_name in samples.items():
        img_path = os.path.join(dataset_root, class_name, img_name)
        if not os.path.exists(img_path):
            # Try other extensions if .jpg fails
            basename = os.path.splitext(img_name)[0]
            for ext in ['.jpeg', '.png', '.JPG']:
                temp_path = os.path.join(dataset_root, class_name, basename + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
        
        if os.path.exists(img_path):
            result = analyze_food_image(img_path)
            detected = ", ".join(result.get('items', ['None']))
            protein = result.get('total_nutrition', {}).get('protein', 0)
            print(f"{class_name:<20} | {detected:<20} | {protein:<12}")
        else:
            print(f"{class_name:<20} | {'NotFound':<20} | {'N/A':<12}")

if __name__ == "__main__":
    verify()
