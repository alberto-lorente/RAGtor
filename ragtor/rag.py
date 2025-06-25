import os
import json
from typing import List, Dict, Tuple

# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from pprint import pprint

from .config import EMBEDDINGS_MODEL, VECTOR_DB_PATH, PDFS_LOADED_ID_FILE_PATH

# docs for faiss indexes https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html

def set_up_RAG_index(embedding_model_id: str = EMBEDDINGS_MODEL,
                     vector_db_path: str = VECTOR_DB_PATH,
                     pdfs_loaded_ids_path: str = PDFS_LOADED_ID_FILE_PATH) -> Tuple:
    
    """
    Sets up a Faiss RAG index based on the dimensions of the embedding model.
    Returns the vector store, the index and the embedding dimensions.
    """
        
    db_found = check_if_db_exists()
    
    if db_found == True:
        print("Vector Db already set up in path:\n")
        for path in db_found:
            print(path)
        return db_found
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
    
    embds = embeddings.embed_query("Hello Word!") # embed docs if multiple text but since i only want the dimensions of the embddings, we can just pass a short string
    emd_dims =  len(embds)
    index = faiss.IndexFlatL2(emd_dims) # dimensions of the embeddings
    
    vector_store = FAISS(embedding_function=embeddings, 
                        index=index, 
                        docstore=InMemoryDocstore(), 
                        index_to_docstore_id={})
    
    # since this is a new db, there will be no pdfs saved so I can already create the tracker
    # I may not need it if loading the vector is quick
    with open(pdfs_loaded_ids_path, "w") as f:
        json.dump([] , f)
    
    vector_store.save_local(vector_db_path)
    
    return embeddings, emd_dims, type(embds)

def check_if_db_exists(vector_db_path: str = VECTOR_DB_PATH) -> bool | List[str]:
    
    if os.path.exists(vector_db_path):
        print("DB folder path exists")
        exists = os.listdir(vector_db_path)
        if len(exists) == 0:
            print("No db found")
            return False
        
        else:
            db_paths = [os.path.join(vector_db_path, db_file) for db_file in exists]
            return db_paths
    else:
        os.makedirs(vector_db_path)
        return False
    
def load_vector_db(vector_db_path:      str = VECTOR_DB_PATH,
                   embeddings_model_id: str = EMBEDDINGS_MODEL):
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_id)

    vector_store = FAISS.load_local(vector_db_path , embeddings, allow_dangerous_deserialization=True)
    
    return vector_store

def set_up_rag_db(vector_db_path: str = VECTOR_DB_PATH,
                  embeddings_model_id: str = EMBEDDINGS_MODEL,
                  pdfs_loaded_ids_path: str = PDFS_LOADED_ID_FILE_PATH):
    
    """Checks if the Vector DB exists. 
    If it does, it loads it with the ollama embedder set in the enviroment variables. 
    If it doesn't, it creates it and laods it. """

    db_exists = check_if_db_exists(vector_db_path=vector_db_path)
    if db_exists:
        vector_store = load_vector_db(vector_db_path=vector_db_path)
        print("Vector Db already set up")
        print("Loading vector store")
        return vector_store
    else:
        set_up_RAG_index(embedding_model_id=embeddings_model_id,
                         vector_db_path=vector_db_path,
                         pdfs_loaded_ids_path=pdfs_loaded_ids_path)
        vector_store = load_vector_db(vector_db_path=vector_db_path)
        print("Vector Db set up")
        print("Loading vector store")
        return vector_store

def raptor_search(vector_store:        FAISS,
                    query:              str, 
                    embeddings_model:    str = EMBEDDINGS_MODEL, 
                    k:                  int = 3) -> List[Document] | List:

    filter_args = {}
    filter_args["chunk_type"] = "cluster_summary"

    results_cluster = vector_store.similarity_search(
        query,
        k=1,
        filter=filter_args)
    
    try:
        metadata = results_cluster[0].metadata
        metadata = {k:v for k,v in metadata.items()}
        metadata["chunk_type"] = "sents"

        results_sents = vector_store.similarity_search(
        query,
        k=k,
        filter=metadata)

    except Exception as e:
        print("Raptor search not successful")
        print(e)
        
        return []
    return results_cluster + results_sents

def default_search(vector_store:        FAISS,
                    query:              str, 
                    embeddings_model:   str = EMBEDDINGS_MODEL, 
                    k:                  int = 3,
                    chunk_type:         str | bool = "sent", 
                    chunk_source:       str | bool = False)  -> List[Document]:
    
    filter_args = {}
           
    if chunk_type:
        filter_args["chunk_type"] = chunk_type
    if chunk_source:
        filter_args["chunk_source"] = chunk_source
    
    results = vector_store.similarity_search(
        query,
        k=k,
        filter=filter_args)
    
    return results

def query_vector_store(vector_store:        FAISS,
                        query:              str, 
                        embeddings_model:   str = EMBEDDINGS_MODEL, 
                        k:                  int = 3, 
                        chunk_type:         str | bool = "sent", 
                        chunk_source:       str | bool = False,
                        mode:               str = "default") -> List[Document]:
           
    if mode == "default":
        results = default_search(vector_store,
                                query,
                                embeddings_model,
                                k,
                                chunk_type,
                                chunk_source)
        return results

    elif mode == "raptor":
        results = raptor_search(vector_store,
                                query,
                                embeddings_model,
                                k)   

        return results

    elif mode == "ensemble":
        
        results_default = default_search(vector_store,
                                query,
                                embeddings_model,
                                k,
                                chunk_type,
                                chunk_source)
        results_raptor = raptor_search(vector_store,
                                query,
                                embeddings_model,
                                k)    

        return results_default + results_raptor




# if __name__ == "__main__":
    
#     # embedder, emd_dims, type_embd = set_up_RAG_index()
    
#     # # pprint(dir(embedder))
#     # print(emd_dims, type_embd)
#     # print(embedder.model)
#     # print(dir(FAISS))
    
#     # print("Does the Db already exist")
#     # print(check_if_db_exists())
    
#     print(os.getcwd())
#     print("Running main")
#     vstore = set_up_rag_db()
#     print(vstore)