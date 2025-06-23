## Repository Overview

This repository contains the code for the Raptor side of the project.
The folder `council_rag` contains the modules we developed to perform our experiments and is documented.

The folder `Evaluation data` contains the pdfs and the script we used to evaluate the performance of the summary and table augmentation model as well as the evaluation data.

`poppler` is required a required dependency to convert the pdfs to markdown. Do not delete it!

The jupyter notebooks show the final steps we followed to develop the pipeline cleaned up and you should be able to run them.

## Comments on the Development

Some comments on specific matters regarding the project:

   - The data is all over the place. While randomly clicking on links provided in the csv, it was a 50/50 whether the pdf would be related to geothermal projects when we thought the vast majority of pdfs concerned that topic. We have saved the more amusing documents we found in the folder `Evaluation data/PDFs for Clustering Eval wrt Embed Model/unrelated/`.
   - We tried a variety of french embedding models from the MTEB(fra, v1) BenchMark but at the end they were all too heavy. We settled for Gemma since it is the only one that we could run locally. We found it difficult to find monolingual French sentence embedding models with an appropiate context window but at the same time, multilingual models where way too big for our use case.
   - Given that there are no major RAPTOR libraries and a lot fo the popular vector dbs lack the flexibility to implement the querying and calculation of the embeddings in a way that felt satisfactory, we tried using pg_vector alongside PostGres. This proved extremely difficult to set up in windows so we switched to using FAISS since it allows to query based on metadata. That being said, the benefit of not having to compute all the embeddings at once in an elegant way is lost with this approach. Still, we developed the schema we would have used as a sql script and it compiled correctly when we tried it in postgres. It can be found under `Future Work`.
   - Throughout the modules/notebooks we are using a naive check length method to compute string lengths. We acknowledge that this is not ideal, but comparing the token segmentation of spacy with this naive method, we found that the final number of tokens was very similar.
   - Most Groq models have a low token per minute limite so we are using the one which has the highest one. It would be fairly difficult to surpass it but just in case, we have added a naive token track and sleep functionality when we call it just in case.
   - Ideally the min number of clusters would be computed dynamically (since as the text length grows, the clusters will be bigger and we may run into LLM querying limits) but here we are setting them manually as an argument to our main preprocessing function. As a rule of thumb, 8 clusters works fine for documents of around 40-50 pages.
   - Given the size, volume and quality of the documents, a well as our access to compute, a in-depth evaluation is not possible and would probably not be worth it.
   - The Prompts used are stored in `french_prompts.json` and `prompts_preprocessing.json`.
   - You may find remarques for the generation of summaries and the quality of the descriptions generated for tables.
   - For an overview of the pipeline, you can check the Notebook 6. It says CUDA out of Memory but it is actually run.
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
And check that you comply with the other requirements file. Note: your kernel will crash if CUDA is not available.
