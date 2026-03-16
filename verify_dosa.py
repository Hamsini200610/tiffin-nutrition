import os
from detector import analyze_food_image
import json

def verify_dosa():
    # Find a Dosa sample
    dosa_dir = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\tuning\Indian Food Dataset\Dosa'
    if not os.path.exists(dosa_dir):
        print(f"Dosa directory not found at {dosa_dir}")
        return
    
    imgs = [f for f in os.listdir(dosa_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        print("No images found in Dosa directory")
        return
    
    img_path = os.path.join(dosa_dir, imgs[0])
    print(f"Running Data-Tuned Detection on {os.path.basename(img_path)}...")
    result = analyze_food_image(img_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\nDetection Results:")
    for item in result['item_details']:
        print(f"- {item['name']}: Confidence={item['confidence']}")
    
    print(f"\nFinal Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    verify_dosa()
