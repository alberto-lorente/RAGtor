import os
import base64
from pprint import pprint as pp

from .encoding_utils import encode_image_to_bytes
from .chunk_class import Chunk

from pydantic import BaseModel, computed_field, ConfigDict, Field
from typing import Dict, Optional, List, Tuple

# from pdfminer.high_level import extract_text # acts a bit weird
import pdfplumber
from pdf2image import convert_from_path # requires poppeler

import ollama
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import VLM_OLLAMA_MODEL, EMBEDDINGS_OLLAMA_MODEL, PDFS_PATH

class Doc(BaseModel):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path:           str                               # the path where the pdf is stored
    
    compute_image:  bool                = False
    # save_images: Optional[bool] = False  # will prob never use it since converting the img to a file wouldnt be that useful if we are already encoding it without needing to save

    table_pages:    Optional[List[int]] = None # optional list of integers (the pages that contain tables)
    
    text_splitter:  RecursiveCharacterTextSplitter
    # chunk_size:     Optional[int] = 512
    # chunk_overlap:  Optional[int] = 32
    
    clusters:       Optional[Dict[str, list]]   = None
    n_clusters:     Optional[int]               = None
    
    @computed_field
    @property
    def txt(self)-> str:
        
        with pdfplumber.open(self.path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        
        return text

    @computed_field
    @property
    def images(self)-> list: # list of length = number of pages - 1 img per page
        
        if self.compute_image:
            
            images = convert_from_path(self.path)
            images = [encode_image_to_bytes(image) for image in images]
            
            return images
        
        else:
            
            images = None
            return images
        
    @computed_field
    @property
    def sents(self)-> list:
        
        sentences = nltk.sent_tokenize(self.txt)
        
        return sentences
    
    @computed_field
    @property
    def n_sents(self)-> int:
        
        return len(self.sents)
        
    @computed_field
    @property
    def chunks(self)-> list:
        
        chunks = self.text_splitter.split_text(self.txt)
        
        return chunks

def compute_doc_emd_chunks(doc:              Doc,
                        pdf_path:            str=PDFS_PATH,
                        length_type:         str="naive",
                        chunking_emb_model:  str=EMBEDDINGS_OLLAMA_MODEL) -> List[Chunk]:


    # clusters      -> if it's true
    # table_pages   -> no because i would use it to augment the context of the text itself 
    # images        -> if compute image = True
    # sents         -> if it's true
    # chunks        -> always true

    pdf_id = doc.path.replace(str(pdf_path) + "\\", "")
    # print("Processing pdf: ", pdf_id)
    # print("Processing chunks")
    chunks_chunks = [Chunk(content=chunk,
                chunk_doc_source=pdf_id,
                chunk_type="chunk",
                length_type=length_type,
                chunking_emb_model=chunking_emb_model) 
            for chunk in doc.chunks]

    # print("Processing sentences")
    chunks_sents = [Chunk(content=sent,
                chunk_doc_source=pdf_id,
                chunk_type="sent",
                length_type=length_type,
                chunking_emb_model=chunking_emb_model) 
            for sent in doc.sents]

    chunks = chunks_chunks + chunks_sents

    if doc.clusters != None:
        # print("Processing cluster summaries")
        # print(doc.clusters.keys())
        # pp(doc.clusters)
        # pp(doc.model_dump())
        chunks_cluster_summaries = [Chunk(content=cluster_data["summary"],
                chunk_doc_source=pdf_id,
                chunk_type="cluster_summary",
                chunk_source=cluster_id,
                length_type=length_type,
                chunking_emb_model=chunking_emb_model) 
            for cluster_id, cluster_data in doc.clusters.items()]

        # print("Processing cluster texts")
        chunks_cluster_texts = [Chunk(content=cluster_data["txt"],
                chunk_doc_source=pdf_id,
                chunk_type="cluster_text",
                chunk_source=cluster_id,
                length_type=length_type,
                chunking_emb_model=chunking_emb_model) 
            for cluster_id, cluster_data in doc.clusters.items()]

        chunks_cluster_sents = []
        for cluster_id, cluster_data in doc.clusters.items():
            for cluster_sent in cluster_data["sents"]:
                cluster_sent_chunk = Chunk(content=cluster_sent,
                                            chunk_doc_source=pdf_id,
                                            chunk_type="sents",
                                            chunk_source=cluster_id, # I will filter the id so no need to do a chunk_type = cluster_sents
                                            length_type=length_type,
                                            chunking_emb_model=chunking_emb_model)
                                            
                chunks_cluster_sents.append(cluster_sent_chunk)

        chunks += chunks_cluster_summaries + chunks_cluster_texts + chunks_cluster_sents

    # if these are images and NOT image descriptions, would need a MM embbedding model

    # if doc.compute_images:

    #     chunks_images = [Chunk(content=image,
    #                         doc_=pdf_id,
    #                         chunk_type="image",
    #                         length_type="naive",
    #                         chunking_emb_model=EMBEDDINGS_OLLAMA_MODEL) 
    #                     for image in doc.images]

    #     chunks += chunks_images

    return chunks


# if __name__ == "__main__":
    
#     example_path = r"C:\Users\alber\Desktop\Git Projects\Council-Minutes\Raptor\council_raptor_rag\data\pdfs\page_36.pdf"
#     doc_pdf_example = Doc(path=example_path, 
#                         compute_image=True)
    
#     print(doc_pdf_example)
#     print(len(doc_pdf_example.images)) 
    
#     mm_model = VLM_OLLAMA_MODEL
    
#     response = ollama.chat(
#         model=mm_model,
#         messages=[{
#             'role': 'user',
#             'content': 'What is in this image?',
#             'images': [doc_pdf_example.images[0]]
#         }]
#     )

#     print(response.message.content)
