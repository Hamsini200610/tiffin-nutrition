from detector import analyze_food_image
import os
import json

test_img = r"c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\idli.jpeg"

try:
    print(f"Testing analyze_food_image with {test_img}...")
    result = analyze_food_image(test_img)
    print("Result:", json.dumps(result, indent=2))
except Exception as e:
    print("Error:", e)
