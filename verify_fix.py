import os
import json
from detector import analyze_food_image

img_path = os.path.join('static', 'uploads', 'chapathi.jpg')
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
else:
    print(f"Testing detection on {img_path}...")
    result = analyze_food_image(img_path)
    print("Result:")
    print(json.dumps(result, indent=2))
