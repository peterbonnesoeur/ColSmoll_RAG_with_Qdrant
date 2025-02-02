import os
from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

class MultimodalRAG:
    def __init__(
        self,
        collection_name: str = "multimodal_collection",
        use_quantization: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.collection_name = collection_name

        retrieval_model = "vidore/colSmol-256M"
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(":memory:")  # Using in-memory storage for demo
        
        # Initialize ColSmolVLM model for embeddings
        self.image_model = ColIdefics3.from_pretrained(
            retrieval_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        ).eval()
        
        # Initialize processor
        self.processor = ColIdefics3Processor.from_pretrained(
            retrieval_model
        )
        
        # Initialize SmolVLM for question answering
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-Instruct",
                quantization_config=bnb_config,
                device_map="auto",
                _attn_implementation="eager"
            )
        else:
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-Instruct",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager"
            )
        
        self.vlm_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        
        # Create Qdrant collection
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection with proper schema and binary quantization"""
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=256,  # ColSmolVLM embedding size
                distance=models.Distance.COSINE,
                # Use binary quantization for better performance
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True,
                    )
                ),
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100  # Lower threshold for faster indexing
            ),
            on_disk_payload=True  # Store payload on disk to save RAM
        )
    
    def _get_embeddings(self, images: List[Image.Image] = None, queries: List[str] = None):
        """Get embeddings for images or queries"""
        with torch.no_grad():
            if images is not None:
                # Process images using the processor
                inputs = self.processor.process_images(images)
                # Move processed inputs to device
                inputs = {k: v.to(self.image_model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                # Get embeddings
                embeddings = self.image_model(**inputs)
                return embeddings
                
            elif queries is not None:
                # Process queries using the processor
                inputs = self.processor.process_queries(queries)
                # Move processed inputs to device
                inputs = {k: v.to(self.image_model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                # Get embeddings
                embeddings = self.image_model(**inputs)
                return embeddings
    
    def index_images(self, image_folder: str, batch_size: int = 6):
        """Index images from a folder into Qdrant"""
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            images = []
            
            for filename in batch_files:
                image_path = os.path.join(image_folder, filename)
                image = Image.open(image_path)
                images.append(image)
            
            # Get image embeddings using ColSmolVLM
            embeddings = self._get_embeddings(images=images)
            
            # Store in Qdrant
            points = []
            for j, embedding in enumerate(embeddings):
                image_path = os.path.join(image_folder, batch_files[j])
                points.append(
                    models.PointStruct(
                        id=i + j,
                        vector=embedding.cpu().float().numpy().tolist(),
                        payload={
                            "image_path": image_path,
                            "filename": batch_files[j]
                        }
                    )
                )
            
            # Upload points with retry mechanism
            for _ in range(3):  # Simple retry mechanism
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=False
                    )
                    break
                except Exception as e:
                    print(f"Error during upsert (will retry): {e}")
                    continue
    
    def search(self, query: str, k: int = 1, timeout: int = 60) -> List[Dict[str, Any]]:
        """Search for relevant images based on text query"""
        # Get query embedding using ColSmolVLM
        query_embedding = self._get_embeddings(queries=[query])
        
        # Search in Qdrant with timeout
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].cpu().float().numpy().tolist(),
            limit=k,
            timeout=timeout  # Add timeout to prevent hanging
        )
        
        return [
            {
                "doc_id": hit.id,
                "score": hit.score,
                "image_path": hit.payload["image_path"],
                "filename": hit.payload["filename"]
            }
            for hit in search_results
        ]
    
    def answer_query(self, query: str, image: Image.Image, max_new_tokens: int = 500) -> str:
        """Generate answer for a query given an image"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        # Prepare inputs
        prompt = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.vlm_processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode the generated text
        generated_text = self.vlm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_text[0] 