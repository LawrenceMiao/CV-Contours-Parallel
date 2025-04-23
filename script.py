import os
from PIL import Image
import sys

def stack_image(image_path, n, output_folder):
    """
    Stack an image vertically n times and save the result to the output folder.
    
    Args:
        image_path (str): Path to the input image
        n (int): Number of times to stack the image
        output_folder (str): Path to the output folder
    
    Returns:
        str: Path to the saved output image
    """
    # Open the image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    # Get image dimensions
    width, height = img.size
    
    # Create a new image with height = n * original height
    new_img = Image.new('RGBA', (width, height * n))
    
    # Paste the original image n times
    for i in range(n):
        new_img.paste(img, (0, i * height))
    
    # Create output filename
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(output_folder, f"{name}_stacked_{n}x{ext}")
    
    # Save the new image
    try:
        new_img.save(output_file)
        print(f"Saved {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def main():
    # Check if enough arguments were provided
    if len(sys.argv) < 2:
        print("Usage: python stack_image.py <input_image.png>")
        return
    
    # Get the input image path
    image_path = sys.argv[1]
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return
    
    # Create output folder
    output_folder = "stacked_images"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process the image with different n values
    n_values = [2, 4, 6, 8, 10, 12]
    
    for n in n_values:
        stack_image(image_path, n, output_folder)
    
    print(f"All stacked images saved to folder: {output_folder}")

if __name__ == "__main__":
    main()