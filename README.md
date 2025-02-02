# Multimodal RAG System for Data Visualization Analysis

A powerful Retrieval Augmented Generation (RAG) system designed to analyze and answer questions about data visualizations. The system combines computer vision and natural language processing to understand charts, graphs, and other data visualizations.

## Features

- 🔍 **Smart Retrieval**: Automatically finds relevant visualizations based on your questions
- 🧠 **Intelligent Analysis**: Provides detailed analysis of multiple visualizations simultaneously
- 📊 **Multiple Dataset Support**: Works with both pre-loaded datasets and custom visualizations
- 🔄 **Real-time Updates**: Automatic refresh of dataset explorer and real-time indexing
- 🎨 **Modern UI**: Clean and intuitive interface built with Gradio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG_with_qdrant.git
cd RAG_with_qdrant
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Install dependencies:
```
uv sync
```

Note: Using `uv` instead of regular `pip` provides faster installation times and better dependency resolution. The project's dependencies are configured in `pyproject.toml` and will be automatically installed with the correct versions.

## Usage

1. Start the application:
```bash
uv run python multimodal_rag/app.py
```

2. Open your browser and navigate to:
```
http://localhost:7860
```

## Available Datasets

### Our World in Data
- Pre-loaded dataset containing visualizations about global development trends
- Includes charts about life expectancy, GDP, healthcare access, and more
- Automatically downloaded on first run

### Custom Dataset
- Upload your own data visualizations
- Supports PNG, JPG, and JPEG formats
- Real-time indexing and analysis

## Features in Detail

### Dataset Explorer
- Real-time preview of available visualizations
- Automatic refresh every 5 seconds
- File information including size and modification date
- Easy deletion of selected images

### Query System
- Natural language questions about visualizations
- Multi-visualization analysis
- Configurable number of results (1-5 visualizations)
- Detailed analysis for each retrieved visualization

### Upload System
- Drag-and-drop file upload
- Multiple file upload support
- Automatic indexing of new files
- Real-time gallery updates

## Technical Details

The system uses:
- Qdrant for vector similarity search
- Transformers for image and text processing
- Gradio for the web interface
- PyTorch for deep learning operations
- IDEFICS for multimodal understanding

## Requirements

- Python 3.11+
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM recommended
- Storage space for datasets and indexes

## Project Structure

```
multimodal_rag/
├── app.py              # Main application file
├── models/
│   └── multimodal_rag.py  # RAG system implementation
├── utils/
│   └── dataset_utils.py   # Dataset management utilities
└── data/
    ├── ourworldindata/    # Pre-loaded dataset
    └── custom/            # Custom uploads directory
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Our World in Data for the dataset
- Hugging Face for transformer models
- Qdrant team for the vector database
- Gradio team for the UI framework
