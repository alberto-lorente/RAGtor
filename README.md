## Repository Overview

```
RAGtor/
├── .gitignore
├── additional-requirements.txt
├── checking_vector_db_things.ipynb
├── data/
│   ├── db/
│   ├── pdf_to_images/
│   ├── pdfs/
│   └── prompts/
│       ├── example_markdown.md
│       ├── example_md_to_text.txt
│       ├── example_table_image.png
│       ├── example.pdf
│       └── loaded_pdfs_id.json
├── Literature/
├── OVERVIEW QUERY.png
├── poppler-24.08.0/
├── processing and loading workflow.ipynb
├── pyproject.toml
├── querying workflow.ipynb
├── ragtor/
│   ├── __init__.py
│   ├── app.py
│   ├── chunk_class.py
│   ├── clustering_tutils.py
│   ├── config.py
│   ├── doc_class.py
│   ├── encoding_tutils.py
│   ├── experiments.ipynb
│   ├── generation.py
│   ├── rag.py
│   ├── reusable.py
│   └── to do's.txt
└── README.md
```

- The data folder contains the default path for the vector db, the default path where the app will look for new pdfs as well as the default prompts for querying.
- Within the Liretature folder you will find the RAPTOR paper which this repo is based on.
- FInally, the ragtor folder includes the package which deals with the classes, functions, utils and logic for the ragtor cli.

## Raptor Pipeline Flow

1. **Pre-processing**
   - Reads PDF and converts to text
   - Splits text into sentences using NLTK
   - Groups sentences into paragraphs
   - Generates embeddings of paragraphs using sentence transformers
   - Performs clustering using Gaussian Mixture Models to identify related content

2. **Data Transformations**
   - Join the paragraphs that belong to the same cluster 
   - Generates summaries for the clusters

3. **RAG Pipeline**
   - At a first step, we query the cluster summaries.
   - Then we query those chunks which belonged to the cluster returned in the previous step as well as the tables.
   - This information is formated together for the augmented generation.

## Current Configuration

Models:
   - SUMMARY_OLLAMA_MODEL        =       "llama3.2:1b"
   - VLM_OLLAMA_MODEL            =       "llava-phi3:latest"  
   - RETRIEVAL_OLLAMA_MODEL      =       "llama3.2:1b"
   - EMBEDDINGS_MODEL            =       "Snowflake/snowflake-arctic-embed-s"

In order to change the models used or the paths of the vector db or pdf folder, just go to 

RAGtor/

   ├── ragtor/

   │----├── config.py


and modify the constants.


### Requirements

In order for the app to run you should have Ollama installed and the NLTK packages punkt and stopwords downloaded.

### Installation:
Navigate to the repository directory and
```bash
pip install -e .
```

Then the app will be accessible from the terminal via the command  **ragtor** . 

## Under Costruction
 []

