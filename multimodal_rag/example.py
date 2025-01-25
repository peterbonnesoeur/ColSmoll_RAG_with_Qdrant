from datasets import load_dataset
from PIL import Image
import os
from models.multimodal_rag import MultimodalRAG

def save_images_to_local(dataset, output_folder="data/"):
    """Save dataset images to local folder"""
    os.makedirs(output_folder, exist_ok=True)
    saved_images = []
    
    for image_id, image_data in enumerate(dataset):
        image = image_data["image"]
        if isinstance(image, str):
            image = Image.open(image)
            
        output_path = os.path.join(output_folder, f"image_{image_id}.png")
        image.save(output_path, format="PNG")
        saved_images.append(output_path)
        print(f"Image saved in: {output_path}")
    
    return saved_images

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("sergiopaniego/ourworldindata_example", split="train")
    
    # Save images locally
    print("Saving images locally...")
    image_folder = "multimodal_rag/data"
    save_images_to_local(dataset, image_folder)
    
    # Initialize MultimodalRAG
    print("Initializing MultimodalRAG system...")
    rag_system = MultimodalRAG(use_quantization=True)
    
    # Index images
    print("Indexing images...")
    rag_system.index_images(image_folder)
    
    # Example query
    query = "What is the overall trend in life expectancy across different countries and regions?"
    print(f"\nQuery: {query}")
    
    # Search for relevant images
    print("Searching for relevant images...")
    results = rag_system.search(query, k=1)
    
    if results:
        # Load the most relevant image
        image_path = results[0]["image_path"]
        image = Image.open(image_path)
        
        # Generate answer
        print("Generating answer...")
        answer = rag_system.answer_query(query, image)
        print(f"\nAnswer: {answer}")
    else:
        print("No relevant images found.")

if __name__ == "__main__":
    main() 