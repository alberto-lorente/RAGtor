import ollama 
import torch

import re
import nltk

from pprint import pprint

from PIL import Image
import io
import base64

from typing import Tuple

from .config import EMBEDDINGS_OLLAMA_MODEL, PDFS_PATH


def encode_image_to_bytes(image): ############## ADD THE TYPE

    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    byte_image = imgByteArr.getvalue()
    
    return byte_image


def compute_embeddings( text:       str, 
                        model:      str = EMBEDDINGS_OLLAMA_MODEL) -> torch.Tensor:
       
    embeddings = ollama.embed(model=model, input=text).embeddings
    # print(embeddings)
    torch_embds = torch.Tensor(embeddings)
    # [batch_size, embed_dim]
    
    return torch_embds

def compute_token_length(   text:       str, 
                            len_type:   str = "naive") -> Tuple:
    
    assert len_type in ["naive", "exhaustive"]
    
    if len_type == "naive":
        tokens = re.findall(r'\w+', text)
        length = len(tokens)
    
    else:
        tokens = nltk.word_tokenize(text) # could compare how fast spacy/nlt/other options are
        length = len(tokens)
            
    return length, tokens

def compute_sent_length(text: str) -> Tuple:
    
    sentences = nltk.sent_tokenize(text) # could compare how fast spacy/nlt/other options are
    length = len(tokens)
            
    return length, sentences

# if __name__ == "__main__":
    
#     # pprint(env_vars.keys())
    
#     text  = "THIS IS SOME EXAMPLE TEXT"
#     emb = compute_embeddings(text)
    
#     print(type(emb))
#     print(emb.shape)
#     print(emb)
    
#     # <class 'torch.Tensor'>
#     # tensor([[-0.1105,  0.0292,  0.0422,  ...,  0.0085,  0.0074, -0.0091]])
#     # torch.Size([1, 3072])
    
#     text  = ["THIS IS SOME EXAMPLE TEXT", "THIS IS ANOTHER TEXT", "This is a test. So is this. so is this. but this as well. And this too."]
#     emb = compute_embeddings(text)
    
#     print(type(emb))
#     print(emb.shape)
#     print(emb)
    
#     # <class 'torch.Tensor'>
#     # torch.Size([2, 3072])
#     # tensor([[-0.1105,  0.0292,  0.0422,  ...,  0.0085,  0.0074, -0.0091],
#             # [-0.1052,  0.0515,  0.0295,  ..., -0.0069,  0.0066, -0.0050]])
            
#     length, tokens = compute_token_length(text[0])
#     print(length)
#     print(tokens)
    
#     # 5
#     # ['THIS', 'IS', 'SOME', 'EXAMPLE', 'TEXT']