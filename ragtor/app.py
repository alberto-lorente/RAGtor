import os
import json
from pprint import pprint as pp

from .clustering_utils import process_doc_clusters
from .doc_class import Doc, compute_doc_emd_chunks
from .chunk_class import Chunk
from .rag import set_up_rag_db, load_vector_db, query_vector_store
from .generation import format_context_string

from .config import PDFS_PATH, PDFS_LOADED_ID_FILE_PATH, VECTOR_DB_PATH, EMBEDDINGS_MODEL, PROMPTS, SUMMARY_OLLAMA_MODEL, QUERY_OLLAMA_MODEL

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_into_vector_db(pdfs_files_path:        str     = PDFS_PATH,
                        loaded_pdfs_ids_path:   str     = PDFS_LOADED_ID_FILE_PATH,
                        vector_db_path:         str     = VECTOR_DB_PATH,
                        chunk_size:             int     = 800,
                        chunk_overlap:          int     = 52,
                        length_type:            str     = "naive",
                        chunking_emb_model:     str     = EMBEDDINGS_MODEL,
                        compute_image:          bool    = False,
                        min_n_clusters:         int     = 3,
                        summarize_cluster:      bool    = True,
                        summary_model:          str     = SUMMARY_OLLAMA_MODEL,
                        summary_prompt:         str     = PROMPTS["summary_prompt"]):
    
    ###############LOADING PDFS#####################

    pdfs_list = os.listdir(pdfs_files_path)
        
    with open(loaded_pdfs_ids_path, "r") as f:
        loaded_pdfs_ids = json.load(f)

    unprocessed_pdfs_ids = list(set(pdfs_list) - set(loaded_pdfs_ids))
    unprocessed_pdfs_paths = [os.path.join(pdfs_files_path, pdf_id) for pdf_id in unprocessed_pdfs_ids if pdf_id.endswith(".pdf")]
    
    if unprocessed_pdfs_paths == []:

        print("No unprocessed pdfs left to load.\n")
        return

    print("Unprocessed pdfs paths:")
    pp(unprocessed_pdfs_paths)


    ###############PROCESSING TEXT #####################

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap)

    print("Processing docs")
    pdf_docs = [Doc(path=pdf_path, 
                    compute_image=compute_image,
                    table_pages=None, 
                    text_splitter=text_splitter,
                    clusters=None,
                    n_clusters=None) 
                for pdf_path in unprocessed_pdfs_paths]

    pdf_docs = [process_doc_clusters(doc,
                                    text_splitter=text_splitter,
                                    min_n_clusters=min_n_clusters,
                                    summarize_cluster=summarize_cluster,
                                    summary_model=summary_model,
                                    summary_prompt=summary_prompt) 
                for doc in pdf_docs]

    pdf_chunks = [compute_doc_emd_chunks(doc,
                                    pdf_path=pdfs_files_path,
                                    length_type=length_type,
                                    chunking_emb_model=chunking_emb_model) 
                for doc in pdf_docs]

    pdf_chunks = [chunk for doc in pdf_chunks for chunk in doc] # unfolded list

    #############LOADING INTO VECTOR DB#####################

    vector_store = set_up_rag_db(vector_db_path) # if i want to load one specifically, i can pass the path of the db
    metadatas = [{

        "chunk_doc_source": chunk.chunk_doc_source,
        "chunk_source": chunk.chunk_source, 
        "chunk_type": chunk.chunk_type,
        "chunking_emb_model": chunk.chunking_emb_model}

                for chunk in pdf_chunks]

    # make sure the chunk.chunk_embeddings' shape is (emb_dims) and not (batch_size x emb_dims)
    iter_text_embs = [(chunk.content, chunk.chunk_embeddings) for chunk in pdf_chunks]

    print("Adding doc chunks into the vector store")
    vector_store.add_embeddings(text_embeddings=iter_text_embs, metadatas=metadatas)
    vector_store.save_local(vector_db_path)

    print("Updating the loaded pdfs ids file")
    processed_pdfs = loaded_pdfs_ids + unprocessed_pdfs_ids

    with open(loaded_pdfs_ids_path, "w") as f:
        json.dump(processed_pdfs, f)

    print("LOADING AND INGESTION OF DOCUMENTS COMPLETED")

######################################QUERYING#####################

def query_vector_db(vector_db_path:         str         = VECTOR_DB_PATH,
                    chunking_emb_model:     str         = EMBEDDINGS_MODEL,
                    query_ollama_model:     str         = QUERY_OLLAMA_MODEL,
                    prompt_template:        str         = PROMPTS["augmented_generation_prompt"],
                    k:                      int         = 10,
                    chunk_type:             str         = "chunk",
                    turn_limit:             int | bool  = False,
                    mode:                   str         = "ensemble"):

    print("Loading vector store.")
    vector_store = load_vector_db(vector_db_path)
    chat_history = []
    print("Vector store loaded.")
    i = 0

    while True:

        query = input("User: ")

        if query == "\\bye":
            return chat_history
        
        print("----Searching for relevant chunks.----")
        search = query_vector_store(vector_store=vector_store,
                                    query=query,
                                    embeddings_model=chunking_emb_model,
                                    k=k,
                                    chunk_type=chunk_type,
                                    mode=mode)
        
        key_points_str = format_context_string(search)
        rag_prompt = prompt_template.format(query, key_points_str)
        user_turn = {
                    "role": "user",
                    "content": rag_prompt
                    }
        if i == 0:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your task is to answer the users question/s and help them with any doubts they may have."
                },
                user_turn
            ]
            response = ollama.chat(model=query_ollama_model,
                            messages=messages)

            response_turn = {"role": "assistant",
                            "content": response.message.content}

            chat_history.append(messages + [response_turn])
            print("Assistant:")
            print(response.message.content)

        else:
            chat_history.append(user_turn)
            response = ollama.chat(model=query_ollama_model,
                            messages=chat_history)   

            response_turn = {"role": "assistant",
                            "content": response.message.content}
            chat_history.append(response_turn)
            print("Assistant:")
            print(response.message.content)

            i += 1
            if turn_limit and i == turn_limit:
                print("\n\nTurns exceeded limit.\n\n")
                break

    return chat_history


def main():

    print("RUNNING APP")
    print("\n\n")

    while True:

        user_input = input(r"Would you like to \load documents , \chat or \exit ? ")
        chat_history = []

        if user_input == r"\load":
            load_into_vector_db()

        elif user_input == r"\chat":
            chat_history = query_vector_db()

        elif user_input == r"\exit":
            print("CHAT HISTORY \n", chat_history)
            break
        
        else:
            print("Invalid input")

    return print("THANKS FOR CHATTING :)")

if __name__ == "__main__":
    main()

