from PIL import Image
import os

img_path = r"c:\Users\HAMSINI\Desktop\tiffin nutrition  tracker\static\uploads\idli.jpeg"
try:
    img = Image.open(img_path)
    print(f"Image format: {img.format}")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
except Exception as e:
    print(f"Error opening image: {e}")
