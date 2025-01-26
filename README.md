# Multimodal RAG with Qdrant and SmolVLM

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system using [Qdrant](https://qdrant.tech/) for vector storage and [SmolVLM](hhttps://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) for visual question answering. The system uses the latest [Colsmol](https://huggingface.co/vidore/colSmol-256M) model for generating image and text embeddings for retrieval.


Also, great article on Colpali and Qdrant here for those that want to do a deep dive:
[Colpali x Qdrant](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html )

## Features

- Efficient image indexing and retrieval using Qdrant
- Quantized SmolVLM for reduced memory usage
- Colsmol embeddings for semantic image search
- Easy-to-use Python API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG_with_qdrant.git
cd RAG_with_qdrant
```

2. Install dependencies:
```bash
uv sync
```

## Usage

The system can be used in two ways:

### 1. Using the Example Script

Run the example script that demonstrates the complete pipeline:

```bash
python multimodal_rag/example.py
```

This will:
- Download a sample dataset from HuggingFace
- Index the images using CLIP embeddings
- Run a sample query
- Generate an answer using SmolVLM

### 2. Using the MultimodalRAG Class

```python
from multimodal_rag.models.multimodal_rag import MultimodalRAG
from PIL import Image

# Initialize the system
rag = MultimodalRAG(use_quantization=True)

# Index your images
rag.index_images("path/to/your/images")

# Search for relevant images
results = rag.search("your query here", k=1)

# Get answer for a specific image
image = Image.open(results[0]["image_path"])
answer = rag.answer_query("your query here", image)
print(answer)
```

## System Requirements

- CUDA-capable GPU (recommended)
- Python 3.8+
- At least 8GB of GPU memory (with quantization)
- At least 16GB of system RAM

## Models Used

- SmolVLM: A lightweight vision-language model for visual question answering
- Colsmol: For generating image and text embeddings for retrieval
- Qdrant: Vector database for efficient similarity search

## License

MIT
