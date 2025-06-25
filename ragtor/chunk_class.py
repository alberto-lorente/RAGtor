from pydantic import BaseModel, computed_field, ConfigDict, Field
from typing import Optional, List, Tuple

import torch
from .encoding_utils import compute_embeddings, compute_token_length
from .config import EMBEDDINGS_MODEL

class Chunk(BaseModel):

    model_config                        = ConfigDict(arbitrary_types_allowed=True) # ALLOW FOR TORCH.TENSOR TYPE
    chunk_doc_source:   Optional[str]   = "Unknown"
    content:            str 
    chunk_type:         str             = "chunk"
    chunk_source:       Optional[str]   = "Unknown" # source within the document, which cluster does it belong to
    
    length_type:        Optional[str]   = "naive"
    chunking_emb_model: str             = EMBEDDINGS_MODEL

    assert chunk_type in ["cluster_summary", "cluster_text", "paragraph", "chunk", "sent", "sents"]
    
    @computed_field
    @property
    def tokens(self, len_type:str = length_type) -> List:
        
        _, tokens = compute_token_length(self.content, len_type=len_type)
        return tokens
    
    @computed_field
    @property
    def token_length(self) -> int:
        
        token_len = len(self.tokens)
        return token_len


    @computed_field
    @property
    def chunk_embeddings(self) -> torch.Tensor:    
            
        chunk_embeddings = compute_embeddings(  text        = self.content, 
                                                embeddings_model= self.chunking_emb_model)            
        return chunk_embeddings
#############################################################################################33    


# if __name__ == "__main__":
    
#     chunk_example = Chunk(  content = "This is the description of an image",
#                             chunk_type = "text")

#     print(chunk_example)
    
    # chunk_source='Unknown' 
    # content='This is the description of an image' 
    # chunk_type='text' length_type='naive' 
    # chunking_emb_model='EMBEDDINGS_OLLAMA_MODEL' 
    # tokens=['This', 'is', 'the', 'description', 'of', 'an', 'image'] 
    # token_length=7 
    # chunk_embeddings=tensor([[-0.1357,  0.0297,  0.0078,  ..., -0.0277,  0.0064, -0.0207]])