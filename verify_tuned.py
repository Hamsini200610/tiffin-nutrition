import os
from detector import analyze_food_image

def verify():
    # Test a Puran Poli sample since it was failing in the screenshot
    img_path = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\tuning\Indian Food Dataset\Puran Poli\Puran_Poli (1).jpg'
    if not os.path.exists(img_path):
        # try common fallback
        img_path = r'c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\tuning\Indian Food Dataset\Puran Poli\Puran Poli (1).jpg'
    
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}")
        return

    print(f"Running Data-Tuned Detection on {os.path.basename(img_path)}...")
    result = analyze_food_image(img_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\nDetection Results:")
    for item in result['item_details']:
        print(f"- {item['name']}: Confidence={item['confidence']}, Protein={item['nutrition']['protein']}g")
    
    print(f"\nTotal Protein: {result['total_nutrition']['protein']}g")
    print(f"Labeled Image: {result['labeled_image_url']}")

if __name__ == "__main__":
    verify()
