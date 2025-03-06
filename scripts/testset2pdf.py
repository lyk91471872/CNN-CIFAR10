import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sys
sys.path.append('..')  # Add project root to path
import config as conf
from dataset import CIFAR10TestDataset

def testset_to_pdf(dataset, output_pdf, use_grayscale=False):
    """
    Exports images from CIFAR10TestDataset directly to a PDF file without saving individual PNG files.
    
    Parameters:
        dataset: an instance of CIFAR10TestDataset
        output_pdf (str): the PDF file path where images will be saved
        use_grayscale (bool): whether to convert images to grayscale
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_pdf)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    
    num_images = len(dataset)
    if num_images == 0:
        print("Dataset is empty.")
        return
    
    # Prepare list for pages (each page will contain 100 images)
    pages = []
    images_per_page = 100
    grid_size = 10  # 10x10 grid per page
    page_width = page_height = 320 * grid_size  # 3200 x 3200 pixels per page
    
    # Process images in chunks of 100
    for i in tqdm(range(0, num_images, images_per_page), desc="Processing pages"):
        # Create a blank white page
        page = Image.new("RGB", (page_width, page_height), "white")
        
        # Process the next 100 images (or fewer if at the end)
        for j in range(min(images_per_page, num_images - i)):
            try:
                img, idx = dataset[i + j]
                
                # If the image is a tensor, convert it to a PIL Image
                if isinstance(img, torch.Tensor):
                    img = to_pil(img.cpu())
                elif not isinstance(img, Image.Image):
                    # If it's not a tensor or PIL Image, try to convert it
                    img = Image.fromarray(img)
                
                # Resize image to 320x320 using nearest neighbor interpolation
                img = img.resize((320, 320), Image.NEAREST)
                
                # Apply grayscale if requested
                if use_grayscale:
                    img = img.convert("L").convert("RGB")
                    text_color = "magenta"
                else:
                    text_color = "black"
                
                # Draw the image id at the bottom left corner
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("bedstead-boldcondensed.otf", 24)
                except IOError:
                    # Fallback to default font if specified font not available
                    font = ImageFont.load_default()
                    
                text = f'ID: {idx}'
                
                # Calculate text dimensions
                try:
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # For older PIL versions that don't have getbbox
                    text_width, text_height = font.getsize(text)
                
                padding = 5
                # Position: a few pixels from the left and bottom edge
                text_position = (padding, 320 - text_height - padding)
                draw.text(text_position, text, fill=text_color, font=font)
                
                # Determine grid position for this image
                row = j // grid_size
                col = j % grid_size
                x = col * 320
                y = row * 320
                
                page.paste(img, (x, y))
            except Exception as e:
                print(f"Error processing image {i + j}: {e}")
        
        pages.append(page)
    
    # Save all pages to a single PDF file
    if pages:
        pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
        print(f"PDF successfully saved as '{output_pdf}'")
    else:
        print("No pages were created.")

if __name__ == '__main__':
    output_pdf_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "test_images.pdf")
    # Set use_grayscale to True if you want to convert images to grayscale and use magenta for the ID
    test_dataset = CIFAR10TestDataset(conf.TEST_DATA_PATH)
    testset_to_pdf(test_dataset, output_pdf_path, use_grayscale=False) 