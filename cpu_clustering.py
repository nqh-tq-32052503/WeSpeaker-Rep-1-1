import umap
import hdbscan
import copy
import numpy as np

SCALE = 4

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
        self.min_num_points = SCALE * self.umap_params["n_neighbors"]
    
    def cluster(self, list_embeddings):
        np_list_embeddings = [embedding.detach().cpu().numpy() for embedding in list_embeddings]
        N = len(np_list_embeddings)
        if N < self.min_num_points:
            num_duplicates = int(self.min_num_points / N) + 1
            np_list_embeddings = np_list_embeddings * num_duplicates
        umap_params = copy.deepcopy(self.umap_params)
        umap_params["n_components"] = min(32, len(np_list_embeddings) - 2)
        umap_obj = umap.UMAP(**umap_params)
        umap_embeddings = umap_obj.fit_transform(np.array(np_list_embeddings))
        list_labels = self.clustering.fit_predict(umap_embeddings)
        list_labels = list_labels[:N]
        return list_labels

        
    
