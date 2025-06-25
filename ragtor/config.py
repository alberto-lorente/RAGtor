import json 
from pathlib import Path

############################MODELS#####################################################

SUMMARY_OLLAMA_MODEL        =       "llama3.2:1b"
VLM_OLLAMA_MODEL            =       "llava-phi3:latest"  
QUERY_OLLAMA_MODEL          =       "llama3.2:1b"
EMBEDDINGS_MODEL            =       "Qwen/Qwen3-Embedding-0.6B"


############################CLUSTERING######################################################

MAX_CLUSTERS                =       10 

############################PATHS###########################################################

cwd                         =       Path(__file__).resolve().parent

relative_parent             =       cwd.parent
data_path                   =       relative_parent / "data"

POPPLER_PATH                =       data_path / "poppler-24.08.0/Library/bin"  
IMAGES_PATH                 =       data_path / "pdf_to_images"

PDFS_PATH                   =       data_path / "pdfs"
PDFS_LOADED_ID_FILE_PATH    =       data_path / "loaded_pdfs_id.json"

VECTOR_DB_PATH              =       data_path / "db"     
PROMPTS_PATH                =       data_path / "prompts" / "prompts.json"    

EXAMPLE_PDF_FILE            =       data_path / "example.pdf"
EXAMPLE_MARKDOWN_FILE       =       data_path / "example_markdown.md"
EXAMPLE_TEXT_FILE           =       data_path / "example_md_to_text.txt"
EXAMPLE_IMAGE_FILE          =       data_path / "example_table_image.png"

with open(PROMPTS_PATH, "r") as f:
    PROMPTS = json.load(f)