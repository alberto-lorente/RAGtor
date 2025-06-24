## Repository Overview

This repository contains the code for the Raptor side of the project.

   - Given that there are no major RAPTOR libraries and a lot fo the popular vector dbs lack the flexibility to implement the querying and calculation of the embeddings in a way that felt satisfactory, we tried using pg_vector alongside PostGres. This proved extremely difficult to set up in windows so we switched to using FAISS since it allows to query based on metadata. That being said, the benefit of not having to compute all the embeddings at once in an elegant way is lost with this approach. Still, we developed the schema we would have used as a sql script and it compiled correctly when we tried it in postgres. It can be found under `Future Work`.

   - Throughout the modules/notebooks we are using a naive check length method to compute string lengths. We acknowledge that this is not ideal, but comparing the token segmentation of spacy with this naive method, we found that the final number of tokens was very similar.


   - Ideally the min number of clusters would be computed dynamically (since as the text length grows, the clusters will be bigger and we may run into LLM querying limits) but here we are setting them manually as an argument to our main preprocessing function. As a rule of thumb, 8 clusters works fine for documents of around 40-50 pages.


## Raptor Pipeline Flow

1. **Pre-processing**
   - Reads markdown and PDF documents
   - Splits text into sentences using spaCy (French language model)
   - Groups sentences into paragraphs
   - Generates embeddings using Gemma 2B model
   - Performs clustering using Gaussian Mixture Models to identify related content

2. **Data Transformations**
To deal with tables:
   - Extracts tables from PDF pages
   - Converts PDF pages to images
   - Encodes images for multimodal context
   - Generates table information with a VL Model
To deal with longer context:
   - Generates summaries for the clusters

3. **RAG Pipeline**
Diagram of the RAG query flow:
![OVERVIEW QUERY](https://github.com/user-attachments/assets/a559f7e0-62db-455f-859f-86b27a53eb10)

- At a first step, we query the cluster summaries.
- Then we query those chunks which belonged to the cluster returned in the previous step as well as the tables.
- This information is formated together for the augmented generation.

## Current Configuration

- Llama Parse for PDF to Markdown, im2table to detect tables for the VL Model.
- Models: Gemma-2b embed, Gemma2-9b, Llama-3.2-11B-Vision.
- 8 sentences per paragraph split with spacy's small French News Core Model. Minimum of 8 clusters, maximum of 12.

### Requirements

In order to run the code, you need to have the following files:
- `HF_TOKEN.txt` - Hugging Face API token
- `GROQ_KEY.txt` - GROQ API key
- A Llama Parse API key if your pdfs have not been converted to markdown yet.

Install the requirements:
```bash
pip install -r requirements.txt
```
And check that you comply with the other requirements file.