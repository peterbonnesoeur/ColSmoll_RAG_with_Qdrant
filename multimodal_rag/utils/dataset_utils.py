import os
import requests
from tqdm import tqdm
import zipfile
from datasets import load_dataset
from PIL import Image
import sys

DATASET_URLS = {
    "ourworldindata": {
        "url": "sergiopaniego/ourworldindata_example",  # Fixed URL format for HuggingFace
        "is_hf": True,
        "description": "Dataset containing visualizations from Our World in Data"
    }
}

def download_and_save_dataset(dataset_name, output_dir):
    """Download and save dataset images"""
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_info = DATASET_URLS[dataset_name]
    output_folder = os.path.join(output_dir, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        if dataset_info["is_hf"]:
            # Load from HuggingFace
            print(f"Downloading {dataset_name} from HuggingFace...", file=sys.stderr)
            try:
                dataset = load_dataset(dataset_info["url"], split="train")
            except Exception as e:
                print(f"Error loading dataset from HuggingFace: {str(e)}", file=sys.stderr)
                print("Attempting to load from HuggingFace Hub...", file=sys.stderr)
                # Try alternative loading method
                dataset = load_dataset(dataset_info["url"], split="train", trust_remote_code=True)
            
            with tqdm(total=len(dataset), desc="Saving images", file=sys.stderr) as pbar:
                for idx, item in enumerate(dataset):
                    try:
                        image = item["image"]
                        if isinstance(image, str):
                            # If image is a URL or path, download or open it
                            if image.startswith(('http://', 'https://')):
                                response = requests.get(image)
                                image = Image.open(requests.get(image).raw)
                            else:
                                image = Image.open(image)
                        elif isinstance(image, bytes):
                            # If image is bytes, convert to PIL Image
                            from io import BytesIO
                            image = Image.open(BytesIO(image))
                        
                        output_path = os.path.join(output_folder, f"image_{idx}.png")
                        image.save(output_path)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing image {idx}: {str(e)}", file=sys.stderr)
                        continue
        else:
            # Download zip file
            print(f"Downloading {dataset_name} zip file...", file=sys.stderr)
            try:
                response = requests.get(dataset_info["url"], stream=True)
                response.raise_for_status()  # Raise an error for bad status codes
                total_size = int(response.headers.get('content-length', 0))
                zip_path = os.path.join(output_dir, f"{dataset_name}.zip")
                
                with open(zip_path, "wb") as f, tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc="Downloading",
                    file=sys.stderr
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)
                
                print("Extracting files...", file=sys.stderr)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_folder)
                
                os.remove(zip_path)
                print("Download complete!", file=sys.stderr)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading zip file: {str(e)}", file=sys.stderr)
                raise
        
        return output_folder
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}", file=sys.stderr)
        if os.path.exists(output_folder):
            print("Cleaning up partial downloads...", file=sys.stderr)
            import shutil
            shutil.rmtree(output_folder)
        raise

def get_dataset_status(dataset_folder):
    """Check if dataset exists and return status"""
    try:
        if not os.path.exists(dataset_folder):
            return "Not downloaded", 0
        
        n_images = len([f for f in os.listdir(dataset_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return "Downloaded" if n_images > 0 else "Empty", n_images
    except Exception as e:
        print(f"Error checking dataset status: {str(e)}", file=sys.stderr)
        return "Error", 0 