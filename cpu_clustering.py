import umap
import hdbscan
import copy
import numpy as np

class EmbeddingClustering(object):
    def __init__(self):
        self.clustering = hdbscan.HDBSCAN(allow_single_cluster=True, min_cluster_size=4, approx_min_span_tree=False, core_dist_n_jobs=1)
        self.umap_params = {
            "n_components" : 32,
            "metric" : "cosine",
            "n_neighbors" :  16,
            "min_dist" : 0.05,
            "random_state" : 2023,
            "n_jobs" : 1
        }
    
    def cluster(self, list_embeddings):
        umap_embeddings = self.execute_umap(list_embeddings)
        list_labels = self.execute_hdbscan(umap_embeddings)
        return list_labels

    def execute_umap(self, list_embeddings):
        np_list_embeddings = [embedding.detach().cpu().numpy() for embedding in list_embeddings]
        umap_params = copy.deepcopy(self.umap_params)
        umap_params["n_components"] = min(32, len(np_list_embeddings) - 2)
        umap_obj = umap.UMAP(**umap_params)
        umap_embeddings = umap_obj.fit_transform(np.array(np_list_embeddings))
        return umap_embeddings
    
    def execute_hdbscan(self, umap_embeddings):
        list_labels = self.clustering.fit_predict(umap_embeddings)
        return list_labels
    
