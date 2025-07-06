## Repository Overview

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
   - RETRIEVAL_OLLAMA_MODEL          =       "llama3.2:1b"
   - EMBEDDINGS_MODEL            =       "Snowflake/snowflake-arctic-embed-s"


### Requirements

In order for the app to run you should have Ollama installed.

### Installation:
Navigate to the repository directory and
```bash
pip install -e .
```

Then the app will be accessible from the terminal via the command  **ragtor** . 
