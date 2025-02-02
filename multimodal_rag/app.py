import gradio as gr
import torch
from PIL import Image
import os
from models.multimodal_rag import MultimodalRAG
from utils.dataset_utils import download_and_save_dataset, get_dataset_status
import shutil
from datetime import datetime

# Device selection
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Initialize the RAG system
DEVICE = get_device()
print(f"Using device: {DEVICE}")
rag_system = MultimodalRAG(use_quantization=True, device=DEVICE)

# Available datasets
DATASETS = {
    "ourworldindata": {
        "name": "Our World in Data",
        "folder": "multimodal_rag/data/ourworldindata",
        "description": "Visualizations about global development trends",
        "examples": [
            ["What is the overall trend in life expectancy across different countries and regions?"],
            ["Can you analyze the correlation between GDP and life expectancy shown in this visualization?"],
            ["What are the main disparities in healthcare access between developed and developing nations?"],
        ]
    },
    "custom": {
        "name": "Custom Dataset",
        "folder": "multimodal_rag/data/custom",
        "description": "Your own data visualizations",
        "examples": []
    }
}

def process_query(query, top_k=3):
    """Process the query and return results with images"""
    try:
        # Search for relevant images
        results = rag_system.search(query, k=top_k)
        
        if not results:
            return "No relevant images found.", []
        
        # Process each result
        images = []
        answers = []
        
        for result in results:
            image_path = result["image_path"]
            image = Image.open(image_path)
            answer = rag_system.answer_query(query, image)
            
            images.append(image)
            answers.append(f"Analysis for image {len(images)}:\n{answer}\n")
        
        final_answer = "\n".join(answers)
        return final_answer, images
    
    except Exception as e:
        return f"An error occurred: {str(e)}", []

# Create the Gradio interface
def create_ui():
    # First, ensure default dataset is downloaded
    default_dataset = "ourworldindata"
    status, _ = get_dataset_status(DATASETS[default_dataset]["folder"])
    if status != "Downloaded":
        print(f"Downloading default dataset ({default_dataset})...")
        download_and_save_dataset(default_dataset, "multimodal_rag/data")
        rag_system.index_images(DATASETS[default_dataset]["folder"])
    else:
        print(f"Default dataset ({default_dataset}) already downloaded.")
        rag_system.index_images(DATASETS[default_dataset]["folder"])

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # Multimodal RAG System for Data Visualization Analysis
        
        This system uses RAG (Retrieval Augmented Generation) to analyze data visualizations and answer questions about them.
        It combines computer vision and natural language processing to understand charts, graphs, and other data visualizations.
        
        **Currently running on: {DEVICE}**
        
        ## Features
        - Automatically retrieves relevant visualizations based on your query
        - Provides detailed analysis of multiple visualizations
        - Supports multiple datasets and custom visualizations
        """)
        
        # Dataset Explorer
        with gr.Row():
            gr.Markdown("## Dataset Explorer")
        
        with gr.Row():
            with gr.Column(scale=2):
                preview_gallery = gr.Gallery(
                    label="Current Dataset Images",
                    show_label=True,
                    columns=4,
                    height=300,
                    preview=True,
                    object_fit="contain",
                    allow_preview=True,
                    elem_id="preview_gallery"
                )
            
            with gr.Column(scale=1):
                image_list = gr.Dataframe(
                    headers=["Filename", "Size", "Date"],
                    datatype=["str", "str", "str"],
                    row_count=10,
                    interactive=False,
                    elem_id="image_list"
                )
                delete_btn = gr.Button("Delete Selected Images", variant="stop", visible=False)
                refresh_btn = gr.Button("Refresh Explorer", variant="secondary")
        
        # Dataset management
        with gr.Row():
            with gr.Column(scale=1):
                dataset_dropdown = gr.Dropdown(
                    choices=list(DATASETS.keys()),
                    value="ourworldindata",
                    label="Select Dataset",
                    info="Choose which dataset to analyze"
                )
                
                # Dataset status and management
                dataset_status = gr.Markdown()
                
                # Custom dataset upload - moved before download
                custom_upload_group = gr.Group(visible=False)
                with custom_upload_group:
                    gr.Markdown("""
                    ### Upload Custom Visualizations
                    Upload your own charts and graphs (PNG, JPG, JPEG)
                    """)
                    file_upload = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Files"
                    )
                    upload_button = gr.Button("Index Uploaded Files")
                    upload_status = gr.Markdown()
                
                # Download button moved after upload
                # download_btn = gr.Button("Download Selected Dataset", visible=True)

            with gr.Column(scale=2):
                dataset_info = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question about the data visualizations...",
                    lines=3
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of visualizations to analyze"
                )
                submit_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Analysis",
                    lines=10
                )
        
        gallery = gr.Gallery(
            label="Retrieved Visualizations",
            show_label=True,
            elem_id="gallery",
            columns=2,
            height=400
        )
        
        # Create separate examples for each dataset
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Example Queries")
                example_queries = gr.Dataset(
                    components=[query_input],
                    samples=DATASETS["ourworldindata"]["examples"],
                    label="Click an example to load it",
                    samples_per_page=5,
                )
                
                def load_example(example):
                    """Load an example query"""
                    return example[0]  # Return just the query text
                
                example_queries.click(
                    load_example,
                    inputs=[example_queries],
                    outputs=[query_input]
                )

        def get_dataset_info(dataset_name):
            """Get detailed information about dataset images"""
            try:
                dataset = DATASETS[dataset_name]
                folder = dataset["folder"]
                if not os.path.exists(folder):
                    return [], [], []
                
                image_files = [f for f in os.listdir(folder) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                preview_images = []
                file_info = []
                
                for img_file in image_files:
                    img_path = os.path.join(folder, img_file)
                    try:
                        # Get file info
                        stats = os.stat(img_path)
                        size = f"{stats.st_size / 1024:.1f} KB"
                        date = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')
                        file_info.append([img_file, size, date])
                        
                        # Get image for preview
                        img = Image.open(img_path)
                        preview_images.append(img)
                    except Exception as e:
                        print(f"Error processing image {img_file}: {e}")
                
                return preview_images, file_info, image_files
            except Exception as e:
                print(f"Error getting dataset info: {e}")
                return [], [], []

        def delete_images(dataset_name, selected_indices):
            """Delete selected images from dataset"""
            try:
                if not selected_indices:
                    return "No images selected for deletion.", [], [], False
                
                dataset = DATASETS[dataset_name]
                folder = dataset["folder"]
                
                # Get current images
                _, _, image_files = get_dataset_info(dataset_name)
                
                # Delete selected images
                deleted_files = []
                for idx in selected_indices:
                    if 0 <= idx < len(image_files):
                        file_to_delete = os.path.join(folder, image_files[idx])
                        try:
                            os.remove(file_to_delete)
                            deleted_files.append(image_files[idx])
                        except Exception as e:
                            print(f"Error deleting {file_to_delete}: {e}")
                
                # Get updated dataset info
                preview_images, file_info, _ = get_dataset_info(dataset_name)
                
                # Reindex the dataset if any files were deleted
                if deleted_files:
                    rag_system.index_images(folder)
                
                status_text = f"Successfully deleted {len(deleted_files)} images: {', '.join(deleted_files)}"
                show_delete = bool(file_info)  # Show delete button if there are still images
                
                return status_text, preview_images, file_info, show_delete
            except Exception as e:
                return f"Error during deletion: {str(e)}", [], [], False

        def update_explorer(dataset_name):
            """Update the dataset explorer"""
            preview_images, file_info, _ = get_dataset_info(dataset_name)
            show_delete = bool(file_info)  # Show delete button if there are images
            return preview_images, file_info, show_delete

        def update_dataset_status(dataset_name):
            """Get the current status of a dataset"""
            try:
                dataset = DATASETS[dataset_name]
                status, n_images = get_dataset_status(dataset["folder"])
                status_text = f"Status: {status}\nNumber of images: {n_images}"
                show_download = status == "Not downloaded"
                return status_text, show_download
            except Exception as e:
                return f"Error checking status: {str(e)}", False

        def update_dataset(dataset_name):
            """Handle dataset changes"""
            try:
                dataset = DATASETS[dataset_name]
                custom_visible = dataset_name == "custom"
                
                # Get current status
                status, n_images = get_dataset_status(dataset["folder"])
                
                description = f"""
                ## {dataset['name']}
                
                {dataset['description']}
                
                Current status: {status}
                Number of images: {n_images}
                """
                
                # Index the selected dataset if it exists and has images
                if status == "Downloaded" and n_images > 0:
                    rag_system.index_images(dataset["folder"])
                
                # Update status
                status_text, show_download = update_dataset_status(dataset_name)
                
                # Get preview images and file info
                preview_images, file_info, _ = get_dataset_info(dataset_name)
                show_delete = bool(file_info)
                
                # Update current dataset
                current_dataset.value = dataset_name
                
                return [
                    description,
                    gr.update(visible=custom_visible),
                    status_text,
                    preview_images,
                    file_info,
                    gr.update(visible=show_delete),
                    gr.update(value=dataset.get('examples', []), visible=bool(dataset.get('examples', [])))
                ]
            except Exception as e:
                error_msg = f"Error updating dataset: {str(e)}"
                return [
                    error_msg,
                    gr.update(visible=False),
                    error_msg,
                    [],
                    [],
                    gr.update(visible=False),
                    gr.update(value=[], visible=False)
                ]
        
        # Connect components
        dataset_dropdown.change(
            update_dataset,
            inputs=[dataset_dropdown],
            outputs=[
                dataset_info,
                custom_upload_group,
                dataset_status,
                preview_gallery,
                image_list,
                delete_btn,
                example_queries
            ]
        )
        
        refresh_btn.click(
            update_explorer,
            inputs=[dataset_dropdown],
            outputs=[preview_gallery, image_list, delete_btn]
        )
        
        # Add selection handling for deletion
        selected_indices = gr.State([])
        
        def handle_selection(evt: gr.SelectData):
            """Handle image selection in gallery"""
            selected = selected_indices.value
            if evt.index in selected:
                selected.remove(evt.index)
            else:
                selected.append(evt.index)
            return selected
        
        preview_gallery.select(
            handle_selection,
            inputs=[],
            outputs=[selected_indices]
        )
        
        delete_btn.click(
            delete_images,
            inputs=[dataset_dropdown, selected_indices],
            outputs=[dataset_status, preview_gallery, image_list, delete_btn]
        )
        
        # Update file upload handler
        def handle_file_upload(files):
            """Handle custom file uploads"""
            try:
                if not files:
                    return "No files uploaded.", "Please select files to upload.", [], [], False
                
                # Create custom dataset directory if it doesn't exist
                os.makedirs(DATASETS["custom"]["folder"], exist_ok=True)
                
                # Save uploaded files
                saved_files = []
                for file in files:
                    try:
                        filename = os.path.basename(file.name)
                        dest_path = os.path.join(DATASETS["custom"]["folder"], filename)
                        Image.open(file.name).save(dest_path)
                        saved_files.append(filename)
                    except Exception as e:
                        return f"Error processing {filename}: {str(e)}", f"Upload failed: {str(e)}", [], [], False
                
                # Index the custom dataset
                rag_system.index_images(DATASETS["custom"]["folder"])
                
                status_text = f"""
                Successfully uploaded and indexed {len(saved_files)} files:
                - {', '.join(saved_files)}
                """
                
                # Get updated dataset info
                preview_images, file_info, _ = get_dataset_info("custom")
                show_delete = bool(file_info)
                
                # Trigger an immediate refresh
                current_dataset.value = "custom"
                
                return status_text, status_text, preview_images, file_info, show_delete
            except Exception as e:
                return f"Error during upload: {str(e)}", "Upload failed.", [], [], False
        
        upload_button.click(
            handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_status, dataset_status, preview_gallery, image_list, delete_btn]
        )
        
        # Handle main query submission
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, top_k],
            outputs=[output_text, gallery]
        )

        # Add a timer for automatic refresh
        refresh_timer = gr.Number(value=0, visible=False, precision=0)
        current_dataset = gr.State(value=default_dataset)

        def auto_refresh():
            """Automatically refresh the current dataset"""
            dataset_name = current_dataset.value
            preview_images, file_info, _ = get_dataset_info(dataset_name)
            show_delete = bool(file_info)
            return preview_images, file_info, show_delete, gr.update(value=refresh_timer.value + 1)

        # Add periodic refresh every 5 seconds
        demo.load(
            auto_refresh,
            outputs=[preview_gallery, image_list, delete_btn, refresh_timer],
            every=5
        )
    
    return demo

if __name__ == "__main__":
    # Create necessary directories
    for dataset in DATASETS.values():
        os.makedirs(dataset["folder"], exist_ok=True)
    
    demo = create_ui()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 