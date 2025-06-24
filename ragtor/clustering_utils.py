from pydantic import BaseModel, computed_field, ConfigDict, Field
from typing import Optional, List, Tuple

import torch
import ollama

from .config import SUMMARY_OLLAMA_MODEL, PROMPTS

from .encoding_utils import compute_embeddings, compute_token_length
from .generation import ollama_generate_text
from .doc_class import Doc
from .chunk_class import Chunk

from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np


# need to input all the embdding arrays [num_sents/paragraphs,emb_dims]

def cluster_n(  cluster_model, 
                embeddings: torch.Tensor, 
                scoring_function) -> Tuple[ List[int], 
                                            np.float32]:
    """
    Clusters the embeddings using the cluster model, a Gaussian Mixture.
    Returns the clusters and the silhouette score.
    """
    clusters = cluster_model.fit_predict(embeddings)
    # print("Number of clusters: ", len(clusters))
    # print("Number of embeddings: ", len(embeddings))
    # print("Clustering algo fit done")
    sil_sc = scoring_function(embeddings, clusters)

    return clusters, sil_sc

# bare in mind depending on the size of the text, if the cluster is too big in terms of token, the model's 
# context window may be smaller!!!!
# should be easy to check sicne I have the n_tokens in a text
def get_optimal_n_clusters( embeddings:     torch.Tensor, 
                            min_n_clusters: int     =   2, 
                            max_n_clusters: int     =   12) -> Tuple[np.int64, 
                                                                        np.ndarray[np.int64],
                                                                        List[np.float32]]:
    """
    Gets the optimal number of clusters for the embeddings based on the max silhouette score.
    The range of clusters to test is from 8 to max_n_clusters. Ideally it would be computed dinamycally based on the data.
    Returns the optimal number of clusters, the final clusters labels for the optimal number of clusters and the silhouette scores.
    """

    #ranges of clusters to test
    range_clusters = np.arange(start=min_n_clusters, stop=max_n_clusters, step=1)

    # print("min_n_clusters:", min_n_clusters)
    # print("max_n_clusters:", max_n_clusters)
    # print("range_clusters:", range_clusters)
    # print(len(embeddings))

    # Compute the silhouette scores for each number of clusters
    silhouette_scores = []
    clusters_labels = []
    for n_cluster in range_clusters:

        # print(f"Clustering with {n_cluster} number of clusters")
        # print()
        n_cluster = int(n_cluster)
        gm = GaussianMixture(n_components=n_cluster, random_state=42) # using Gaussian Mixture as in the paper references
        clusters, sil_sc = cluster_n(gm, embeddings, silhouette_score)
        silhouette_scores.append(sil_sc)
        clusters_labels.append(clusters) # saving the labels so that we don't need to recompute them after getting the optimal n

    # Getting the optimal number of clusters
    # print("silhouette_scores:", silhouette_scores)
    max_ = np.argmax(silhouette_scores)
    optimal_n = int(range_clusters[max_])
    # print("Index", max_)
    # print("Optimal Number of Clusters", optimal_n)

    # Getting the labels for the optimal number of clusters
    final_clusters = clusters_labels[max_]
    
    return optimal_n, final_clusters, silhouette_scores

def process_doc_clusters(doc:               Doc,
                        min_n_clusters:     int=4,
                        max_n_clusters:     int=9,
                        summarize_cluster:  bool=True, 
                        summary_model:      str=SUMMARY_OLLAMA_MODEL,
                        summary_prompt:     str=PROMPTS["summary_prompt"]) -> Doc:

    # max_n_clusters  = doc.n_sents / 5
    if max_n_clusters < min_n_clusters or doc.n_sents < min_n_clusters or doc.n_sents < max_n_clusters:
        # print("Number of sentences", doc.n_sents)
        # print("Not enough sentences to cluster")
        return doc
    # print("Number of max clusters: ", max_n_clusters)
    # print("Number of min clusters: ", min_n_clusters)
    # print("Computing embeddings")
    embeddings      = compute_embeddings(doc.sents)
    optimal_n, final_clusters, silhouette_scores = get_optimal_n_clusters(embeddings, 
                                                                        min_n_clusters=min_n_clusters, 
                                                                        max_n_clusters=max_n_clusters)
    # print("Optimal number of clusters obtained")    
    clusters            = {f"cluster_{n}": {"sents": []}   for n in range(optimal_n)}
    final_clusters_str  = [f"cluster_{n}"       for n in final_clusters]
    
    for cluster_str, sent in zip(final_clusters_str, doc.sents):
        clusters[cluster_str]["sents"].append(sent)
    # print("Clusters sentences added")
    
    for cluster_str in clusters.keys():
        clusters[cluster_str]["txt"] = " ".join(clusters[cluster_str]["sents"])
    # print("Clusters sentences joined")
    
        if summarize_cluster:
            # print("Generating the summary for cluster {}".format(cluster_str))
            formatted_prompt = summary_prompt.format(clusters[cluster_str]["txt"])
            summary, message_history = ollama_generate_text(model=summary_model, formatted_prompt=formatted_prompt)
            clusters[cluster_str]["summary"] = summary
            clusters[cluster_str]["message_history"] = message_history
    
    doc.clusters    = clusters
    doc.n_clusters  = optimal_n

    return doc

# if __name__=="__main__":
    
#     texts = ["Sentence 1", "Sentence 1", "Sentence 1", "Sentence 1", "Sentence 1", "Sentence 1"
#             "Hello My name is", "Hello whats up", "Hello hi hey sup", "Hey yo yo yo yo",
#             "This is a sentence","This is not a sentence", "This is a word", "This is a uhhhhh" ]
    
#     embeddings = compute_embeddings(texts)
#     n_texts = len(texts)
#     print(embeddings) # [n_sents, embd_dim]
#     print(type(embeddings))
#     print(len(embeddings))
#     print(embeddings.shape[0])
    
#     tensor([[-0.1071,  0.0207,  0.0037,  ...,  0.0160,  0.0010,  0.0059],
#         [-0.1071,  0.0207,  0.0037,  ...,  0.0160,  0.0010,  0.0059],
#         [-0.1071,  0.0207,  0.0037,  ...,  0.0160,  0.0010,  0.0059],
#         ...,
#         [-0.1270, -0.0087,  0.0076,  ...,  0.0026, -0.0064, -0.0136],
#         [-0.1273,  0.0177, -0.0210,  ..., -0.0058,  0.0042,  0.0107],
#         [-0.1159,  0.0027,  0.0207,  ..., -0.0221,  0.0291, -0.0234]])
#        <class 'torch.Tensor'>
#        13
#        13
    
#   optimal_n, final_clusters, silhouette_scores = get_optimal_n_clusters(embeddings, 
#                                                                        min_n_clusters=2, 
#                                                                        max_n_clusters=12)
    
    # print(optimal_n)
    # print(type(optimal_n))
    
    # print(final_clusters)
    # print(type(final_clusters))
    # print(type(final_clusters[0]))

    # print(silhouette_scores)
    # print(type(silhouette_scores))
    # print(type(silhouette_scores[0]))

    # 5
    # <class 'numpy.int64'>
    # [0 0 0 0 0 2 2 2 2 1 1 3 4]
    # <class 'numpy.ndarray'>
    # <class 'numpy.int64'>
    # [np.float32(0.337017), np.float32(0.38912618), np.float32(0.45581087), np.float32(0.46902663)]
    # <class 'list'>
    # <class 'numpy.float32'>