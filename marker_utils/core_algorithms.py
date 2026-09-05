import episcanpy as epi
from sklearn.neighbors import kneighbors_graph
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.api as sm
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
import scipy
from scipy.optimize import minimize
from scipy.stats import nbinom
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import random
import anndata
import numba
from numba import njit,prange,float32
import os
from scipy.stats import poisson
from pathlib import Path
import sys
import math
import scanpy as sc
from copy import copy
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, rbf_kernel
from ._version import SOFTWARE_SHORT_NAME, SOFTWARE_VERSION, __version__


def _as_1d(values):
    """Return dense, one-dimensional numeric values from dense/sparse reductions."""
    if scipy.sparse.issparse(values):
        return values.toarray().reshape(-1)
    return np.asarray(values).reshape(-1)


def _matrix_mean_and_var(matrix):
    """Compute per-feature moments without assuming that AnnData.X is sparse."""
    if scipy.sparse.issparse(matrix):
        numeric = matrix.astype(np.float64, copy=True)
        mean = _as_1d(numeric.mean(axis=0))
        squared = numeric.copy()
        squared.data **= 2
        second_moment = _as_1d(squared.mean(axis=0))
    else:
        numeric = np.asarray(matrix, dtype=np.float64)
        mean = numeric.mean(axis=0)
        second_moment = np.square(numeric).mean(axis=0)
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    return np.asarray(mean).reshape(-1), np.asarray(variance).reshape(-1)


def _normalize_scores_to_unit_max(values):
    """Scale non-negative scores by their finite maximum to the interval [0, 1]."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    maximum = values.max(initial=0.0)
    if maximum <= 0:
        return np.zeros_like(values)
    return values / maximum


def _resolve_pca_component_count(requested, n_obs, n_vars):
    """Return a valid PCA component count for the observed matrix dimensions."""
    if requested is None or int(requested) < 1:
        raise ValueError("n_comps must be a positive integer.")
    upper_bound = min(n_obs, n_vars) - 1
    if upper_bound < 1:
        raise ValueError("PCA requires at least 2 observations and 2 features.")
    return min(int(requested), upper_bound)


def _resolve_neighbor_count(requested, n_obs):
    """Return a valid non-self neighbor count for the observation count."""
    if requested is None or int(requested) < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if n_obs < 2:
        raise ValueError("Neighbor construction requires at least 2 observations.")
    return min(int(requested), n_obs - 1)


def _resolve_inclusive_neighbor_count(requested, n_obs):
    """Return a valid neighbor count for a graph that includes each point itself."""
    if n_obs < 1:
        raise ValueError("Neighbor construction requires at least 1 observation.")
    return min(max(1, int(requested)), n_obs)


def _configured_n_jobs(task_count):
    configured = os.environ.get("CSFEATURES_N_JOBS")
    if configured is not None:
        try:
            configured = int(configured)
        except ValueError as exc:
            raise ValueError("CSFEATURES_N_JOBS must be a positive integer.") from exc
        if configured < 1:
            raise ValueError("CSFEATURES_N_JOBS must be a positive integer.")
        return min(configured, task_count)
    return max(1, min(16, os.cpu_count() or 1, task_count))


def _load_adata(adata_or_path, copy_adata=True):
    if isinstance(adata_or_path, anndata.AnnData):
        return adata_or_path.copy() if copy_adata else adata_or_path
    if isinstance(adata_or_path, (str, os.PathLike, Path)):
        return sc.read_h5ad(adata_or_path)
    raise TypeError("Expected an AnnData object or a path to an .h5ad file.")


def _validate_adata(adata, celltype_key="celltype", spatial_key=None):
    if not isinstance(adata, anndata.AnnData):
        raise TypeError("adata must be an AnnData object.")
    if adata.n_obs < 4:
        raise ValueError("CSFeatures requires at least 4 observations.")
    if adata.n_vars < 2:
        raise ValueError("CSFeatures requires at least 2 features.")
    if celltype_key not in adata.obs:
        raise KeyError(f"AnnData.obs must contain '{celltype_key}'.")

    labels = adata.obs[celltype_key]
    if labels.isna().any():
        raise ValueError(f"AnnData.obs['{celltype_key}'] contains missing labels.")
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2:
        raise ValueError("CSFeatures requires at least 2 cell populations.")
    too_small = counts[counts < 2]
    if not too_small.empty:
        names = ", ".join(map(str, too_small.index.tolist()))
        raise ValueError(f"Each cell population needs at least 2 observations; too small: {names}.")

    matrix_values = adata.X.data if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.issubdtype(matrix_values.dtype, np.number):
        raise TypeError("AnnData.X must contain numeric values.")
    if not np.isfinite(matrix_values).all():
        raise ValueError("AnnData.X contains NaN or infinite values.")
    if matrix_values.size and np.min(matrix_values) < 0:
        raise ValueError(
            "AnnData.X must contain non-negative preprocessed values; "
            "scaled/z-scored matrices are not valid CSFeatures input."
        )

    if spatial_key is not None:
        if spatial_key not in adata.obsm:
            raise KeyError(f"AnnData.obsm must contain '{spatial_key}'.")
        spatial = np.asarray(adata.obsm[spatial_key])
        if spatial.ndim != 2 or spatial.shape[0] != adata.n_obs or spatial.shape[1] < 2:
            raise ValueError(
                f"AnnData.obsm['{spatial_key}'] must have shape (n_obs, at least 2)."
            )
        if not np.isfinite(spatial).all():
            raise ValueError(f"AnnData.obsm['{spatial_key}'] contains NaN or infinite values.")

    return counts

def compute_the_adjacency_matrix(X:np.ndarray,k_neighbors=15):
    '''
    Given cell features, compute the adjacency matrix.
    Input.
    X:cell vector, shape:(number of cells,cell features).
    k_neighbors:number of neighbors.
    Output.
    adj_matrix:adjacency matrix, shape (number of cells, number of cells).
    '''
    k_neighbors = _resolve_inclusive_neighbor_count(k_neighbors, X.shape[0])
    adj_matrix = kneighbors_graph(X, k_neighbors, mode='connectivity',include_self=True)
    adj_matrix = adj_matrix.toarray()
    return adj_matrix

def compute_the_similarity_matrix(X:np.ndarray,method="euclidean"):
    '''
    Given cell features, compute the similarity matrix.
    Input.
    X:cell vector, shape:(number of cells, cell features).
    METHOD:Method to compute the similarity.
    Optional:
    1.
    Output.
    Similarity matrix, shape (number of cells, number of cells).
    '''
    if method=="euclidean":
        X=euclidean_distances(X)
        X=1/(1+X)
    return X

def expression_matrix_correction(count_matrix:np.ndarray,adj_matrix:np.ndarray):
    '''
    Given an expression matrix and a cell similarity matrix, do a correction for expression.
    Input.
    count_matrix:expression_matrix, shape (cell, expression).
    similarity_matrix: cell similarity matrix, shape (cell, cell).
    Output.
    Corrected expression matrix, shape (cell, expression).
    '''
    return np.matmul(adj_matrix,count_matrix)

def calculate_mean_and_var_adata(adata,cluster_count_matrix:np.ndarray,
    cluster_processed_matrix:np.ndarray,
    gene_name:np.ndarray,
    cluster:str,
    clusters:np.ndarray,
    count_matrix:np.ndarray,
    processed_matrix:np.ndarray,
    ans:dict):
    ans[cluster]["mean"], ans[cluster]["var"] = _matrix_mean_and_var(cluster_count_matrix)
    ans[cluster]["mean_hat"], ans[cluster]["var_hat"] = _matrix_mean_and_var(
        cluster_processed_matrix
    )



def expression_matrix_correction_adata(adata:anndata._core.anndata.AnnData):
    if 'distances' not in adata.obsp:
        raise KeyError("AnnData.obsp must contain 'distances'; run neighbor construction first.")
    D=scipy.sparse.csr_matrix(copy(adata.obsp['distances']),dtype=np.float64)
    D.data=1/(D.data+1)
    D=(D+scipy.sparse.eye(adata.shape[0],format='csr')).tocsr()
    row_sums=_as_1d(D.sum(axis=1))
    if np.any(row_sums <= 0):
        raise ValueError("The neighbor graph contains a row with no positive weight.")
    D=scipy.sparse.diags(1.0/row_sums).dot(D).tocsr()
    adata.obsm['correctX']=D.dot(adata.X)


def calculate_zero_mu_max(adata,cluster_count_matrix:np.ndarray,
    cluster_processed_matrix:np.ndarray,
    gene_name:np.ndarray,
    cluster:str,
    clusters:np.ndarray,
    count_matrix:np.ndarray,
    processed_matrix:np.ndarray,
    ans:dict):
    '''
    Calculate the maximum mean of the other clusters.
    '''
    mu_max=np.zeros_like(ans[cluster['mean']])
    for index in range(len(ans[cluster]['mean'])):
        other_cluster_max=-1e10
        for j in ans.keys():
            if j==cluster: continue
            other_cluster_max=max(other_cluster_max,ans[j]['mean_hat'][index])
        mu_max[index]=other_cluster_max
    ans[cluster]['mu_max']=mu_max

def calculate_mean_and_var(cluster_count_matrix:np.ndarray,
    cluster_processed_matrix:np.ndarray,
    gene_name:np.ndarray,
    cluster:str,
    clusters:np.ndarray,
    count_matrix:np.ndarray,
    processed_matrix:np.ndarray,
    ans:dict,**kwargs):
    ans[cluster]["mean"]=cluster_count_matrix.mean(axis=0)
    ans[cluster]["var"]=cluster_count_matrix.var(axis=0)
    ans[cluster]["mean_hat"]=cluster_processed_matrix.mean(axis=0)
    ans[cluster]["var_hat"]=cluster_processed_matrix.var(axis=0)

def calculate_local_mean_max(cluster_count_matrix:np.ndarray,
    cluster_processed_matrix:np.ndarray,
    gene_name:np.ndarray,
    cluster:str,
    clusters:np.ndarray,
    count_matrix:np.ndarray,
    processed_matrix:np.ndarray,
    ans:dict,**kwargs):
    '''
    Computes the maximum local mean in the other clusters.
    Will add a local_mean_max to each cluster in the ans dictionary.
    '''
    other_clusters=(clusters==cluster)==False
    X_pca=kwargs["X_pca"]
    other_clusters_pca=X_pca[other_clusters] # other cells * genes
    other_clusters_matrix=count_matrix[other_clusters]
    other_clusters_knn=compute_the_adjacency_matrix(other_clusters_pca,k_neighbors=20) # other cells * cells
    local_mean_matrix=np.matmul(other_clusters_knn,other_clusters_matrix)/other_clusters_knn[0].sum()
    local_mean_max=local_mean_matrix.max(axis=0)
    ans[cluster]["other_clusters_local_mean_max"]=local_mean_max


def calculate_V(cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,**kwargs):
    '''
    Define a method to compute the metrics for a cluster.
    After calculating the metrics, record the metrics using ans[cluster]["metric name"] = metrics array.
    Input.
    cluster_processed_matrix: matrix corresponding to the current cluster. Shape (number of cells, genes).
    cluster_processed_matrix: the processed matrix corresponding to the current cluster. Shape (number of cells, genes).
    gene_name:name of the gene, length is an np array of the number of genes.
    cluster:name of current cluster.
    clusters:list of clusters, length is an np array of cells.
    count_matrix: matrix of expression values, shape (number of cells, genes).
    processed_matrix: matrix of processed values, shape (number of cells, genes).
    ans:Dictionary to record the computed metrics.
    '''
    ans[cluster]["gene_name"]=gene_name
    var=ans[cluster]["var"]
    mean_hat=ans[cluster]["mean_hat"]**2
    # mean_hat=mean_hat/mean_hat.max()
    mean=ans[cluster]["mean"]**2
    # V=mean_hat/((ans[cluster]["smoothness"]))
    denominator=np.maximum(np.asarray(ans[cluster]["smoothness"],dtype=np.float64),np.finfo(np.float64).eps)
    V=mean/denominator
    ans[cluster]["V"]=V

def calculate_prop(cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,**kwargs):
    '''
    Calculate the percentage of cells with non-zero expression values in the current cluster.
    Input: original expression matrix cluster_count_matrix
    Output: proportion of non-zero clusters other than the current cluster (square and add one to increase their influence weights)
    '''
    other_clusters=list(Counter(clusters).keys())
    other_clusters.remove(cluster)
    prop=np.zeros(shape=[len(other_clusters),len(gene_name)])
    for k,i in enumerate(other_clusters):
        tmp_cluster=clusters==i
        tmp_cluster_matrix=count_matrix[tmp_cluster]
        tmp_cluster_matrix=_as_1d((tmp_cluster_matrix>0).mean(axis=0))
        prop[k,:]=tmp_cluster_matrix
    # prop_sum = np.sum(np.exp(2*prop*100), axis=0)
    prop_sum =np.sum((np.e+1)**(prop*100),axis=0)
    # prop_sum = prop.sum(axis=0)
    # prop_sum = (((prop_sum*100)**0.5)+1)
    ans[cluster]["prop_sum"]=prop_sum
    # ans[cluster]["prop_sum"]=(prop_sum**2)+1
    ans[cluster]["prop_max"]=prop.max(axis=0)
    prop = _as_1d((cluster_count_matrix>0).mean(axis=0))
    # prop=(prop*100)**2
    prop=np.exp(prop*100)
    ans[cluster]["prop"]=prop


def calculate_V_adata(adata,cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict):
    '''
    Define a method to compute the metrics for a cluster.
    After calculating the metrics, record the metrics using ans[cluster]["metric name"] = metrics array.
    Input.
    cluster_processed_matrix: matrix corresponding to the current cluster. Shape (number of cells, genes).
    cluster_processed_matrix: the processed matrix corresponding to the current cluster. Shape (number of cells, genes).
    gene_name:name of the gene, length is an np array of the number of genes.
    cluster:name of current cluster.
    clusters:list of clusters, length is an np array of cells.
    count_matrix: matrix of expression values, shape (number of cells, genes).
    processed_matrix: matrix of processed values, shape (number of cells, genes).
    ans:Dictionary to record the computed metrics.
    '''
    ans[cluster]["gene_name"]=gene_name
    var=ans[cluster]["var"]
    mean_hat=ans[cluster]["mean_hat"]**2
    # mean_hat=mean_hat/mean_hat.max()
    mean=ans[cluster]["mean"]**2
    # V=mean_hat/((ans[cluster]["smoothness"]))
    denominator=np.maximum(np.asarray(ans[cluster]["smoothness"],dtype=np.float64),np.finfo(np.float64).eps)
    V=mean/denominator
    ans[cluster]["V"]=V


def calculate_local_mean_max_adata(adata,cluster_count_matrix:np.ndarray,
    cluster_processed_matrix:np.ndarray,
    gene_name:np.ndarray,
    cluster:str,
    clusters:np.ndarray,
    count_matrix:np.ndarray,
    processed_matrix:np.ndarray,
    ans:dict):
    '''
    Computes the maximum local mean in the other clusters.
    Will add a local_mean_max to each cluster in the ans dictionary.
    '''
    other_clusters=(clusters==cluster)==False
    X_pca=adata.obsm["X_pca"]
    other_clusters_pca=X_pca[other_clusters] # other cells * genes
    other_clusters_matrix=count_matrix[other_clusters]
    tmp_anndata=sc.AnnData(other_clusters_matrix)
    tmp_anndata.obsm['X_pca']=other_clusters_pca
    local_neighbors=_resolve_neighbor_count(15,tmp_anndata.n_obs)
    epi.pp.neighbors(tmp_anndata,metric='euclidean',n_neighbors=local_neighbors,use_rep='X_pca')
    knn=(scipy.sparse.csr_matrix(tmp_anndata.obsp['distances'])+scipy.sparse.eye(tmp_anndata.shape[0],format='csr')).tocsr()
    knn.data=np.ones_like(knn.data)
    local_mean_matrix=knn.dot(other_clusters_matrix)/knn[0].sum()
    local_mean_max=local_mean_matrix.max(axis=0)
    ans[cluster]["other_clusters_local_mean_max"]=_as_1d(local_mean_max)


def calculate_prop_adata(adata,cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,**kwargs):
    '''
    Calculate the percentage of cells with non-zero expression values in the current cluster.
    Input: original expression matrix cluster_count_matrix
    Output: proportion of non-zero clusters other than the current cluster (square and add one to increase their influence weights)
    '''
    other_clusters=list(Counter(clusters).keys())
    other_clusters.remove(cluster)
    prop=np.zeros(shape=[len(other_clusters),len(gene_name)])
    for k,i in enumerate(other_clusters):
        tmp_cluster=clusters==i
        tmp_cluster_matrix=count_matrix[tmp_cluster]
        tmp_cluster_matrix=np.array((tmp_cluster_matrix>0).mean(axis=0)).reshape(-1)
        prop[k,:]=tmp_cluster_matrix
    # prop_sum = np.sum(np.exp(2*prop*100), axis=0)
    prop_sum =np.sum((np.e+1)**(prop*100),axis=0)
    # prop_sum = prop.sum(axis=0)
    # prop_sum = (((prop_sum*100)**0.5)+1)
    ans[cluster]["prop_sum"]=prop_sum
    # ans[cluster]["prop_sum"]=(prop_sum**2)+1
    ans[cluster]["prop_max"]=prop.max(axis=0)
    prop = (cluster_count_matrix>0).mean(axis=0)
    # prop=(prop*100)**2
    prop=np.exp(prop*100)
    ans[cluster]["prop"]=np.array(prop).reshape(-1)


@njit(parallel=True,nogil=True)
def _smoothness(cluster_count_matrix,similarity_matrix):
    '''
    cluster_count_matrix: expression matrix, shape (cells, genes).
    similarity_matrix: similarity matrix, shape (cell, cell).
    '''
    ans=np.zeros_like(cluster_count_matrix[0])
    cell_dim=cluster_count_matrix.shape[0]
    gene_dim=cluster_count_matrix.shape[1]
    for i in prange(gene_dim):
        for j in prange(cell_dim):
            for k in prange(j+1,cell_dim):
                tmp1=cluster_count_matrix[j][i]
                tmp2=cluster_count_matrix[k][i]
                if ((tmp1==0 and tmp2==0)): 
                    ans[i]+=2
                else:
                    ans[i]+=2*similarity_matrix[j][k]*(np.abs(tmp1-tmp2)) / (tmp1+tmp2+1)
    return ans
def calculate_smoothness(cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,**kwargs):
    '''
    Calculate the smoothness of each cluster.
    '''
    similarity_matrix=kwargs["similarity_matrix"]
    tmp=np.where(clusters==cluster)[0]
    similarity_matrix=similarity_matrix[tmp]
    similarity_matrix=similarity_matrix[:,tmp]
    ans[cluster]["smoothness"]=_smoothness(cluster_count_matrix,similarity_matrix) / similarity_matrix.shape[0]**2
    ans[cluster]["smoothness"]=ans[cluster]["smoothness"]

def calculate_EI(cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,
        **kwargs):
    '''
    Define a method to compute the metrics for a cluster.
    After calculating the metrics, record the metrics using ans[cluster]["metric name"] = metrics array.
    Input:
    cluster_count_matrix: matrix corresponding to the current cluster. Shape (number of cells, genes).
    cluster_processed_matrix: matrix corresponding to current cluster. Shape (number of cells, genes).
    gene_name:name of the gene, length is an np array of the number of genes.
    cluster:name of current cluster.
    clusters:list of clusters, length is an np array of cells.
    count_matrix: matrix of expression values, shape (number of cells, genes).
    processed_matrix: matrix of processed values, shape (number of cells, genes).
    ans:Dictionary to record the computed metrics.
    '''
    
    EI1=np.array(ans[cluster]['V'])/ ((ans[cluster]["other_clusters_local_mean_max"])**2+1)
    EI1=EI1/EI1.max()
    EI2=np.array(ans[cluster]['prop'])/ ((ans[cluster]["prop_sum"]))
    EI2=EI2/EI2.max()
    
    EI=EI1*EI2

    ans[cluster]['EI']=EI




@njit(parallel=True,nogil=True)
def __smoothness_single_gene(gene_count,cell2cell,similarity_matrix):
    cell_dim=cell2cell.shape[0]
    ans=0
    for j in prange(cell_dim):
        for k in prange(similarity_matrix.shape[1]):
            cell1_index=j
            cell2_index=cell2cell[j,k]
            cell1_count=gene_count[cell1_index]
            cell2_count=gene_count[cell2_index]
            similarity_1and2=similarity_matrix[j,k]
            if ((cell1_count==0 and cell2_count==0)): 
                ans+=1
            else:
                ans+=similarity_1and2*(np.abs(cell1_count-cell2_count)) / (cell1_count+cell2_count+1)
    return ans

from joblib import Parallel, delayed
import numpy as np

def __smoothness_adata(cluster_count_matrix, similarity_matrix):
    '''
    cluster_count_matrix: expression matrix, shape (cells, genes).
    similarity_matrix: similarity matrix, shape (cell, cell).
    '''
    cell_dim = cluster_count_matrix.shape[0]
    gene_dim = cluster_count_matrix.shape[1]

    cluster_count_matrix=scipy.sparse.csr_matrix(cluster_count_matrix)
    similarity_matrix=scipy.sparse.csr_matrix(similarity_matrix)
    neighbor_slots = int(np.max((similarity_matrix > 0).sum(axis=1)))
    cell2cell = np.zeros([cell_dim, neighbor_slots], dtype=np.int32)
    for row_index in range(cell_dim):
        start = similarity_matrix.indptr[row_index]
        end = similarity_matrix.indptr[row_index + 1]
        cell2cell[row_index][:end - start] = similarity_matrix.indices[start:end]
    neighbor_similarity = np.array(
        [
            similarity_matrix[row_index, cell2cell[row_index]].toarray().reshape(-1)
            for row_index in range(cell_dim)
        ]
    )

    def process_gene(i):
        gene_count = cluster_count_matrix[:, i].toarray().reshape(-1)
        return __smoothness_single_gene(gene_count, cell2cell, neighbor_similarity)

    ans = Parallel(
        n_jobs=_configured_n_jobs(gene_dim),
        prefer="threads",
    )(delayed(process_gene)(i) for i in range(gene_dim))
    return np.array(ans)

import scipy.sparse

def calculate_smoothness_adata(adata,cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict):
    '''
    Calculate the smoothness of each cluster.
    '''
    X_pca=adata.obsm["X_pca"]
    clusters_pca=X_pca[clusters==cluster] # other cells * genes
    tmp_anndata=sc.AnnData(cluster_processed_matrix)
    tmp_anndata.obsm['X_pca']=clusters_pca
    local_neighbors=_resolve_neighbor_count(15,tmp_anndata.n_obs)
    epi.pp.neighbors(tmp_anndata,metric='euclidean',n_neighbors=local_neighbors,use_rep='X_pca')
    similarity_matrix=scipy.sparse.csr_matrix(tmp_anndata.obsp['distances'])
    similarity_matrix.data=1/(similarity_matrix.data+1)
    similarity_matrix=(similarity_matrix+scipy.sparse.eye(similarity_matrix.shape[0],format='csr')).tocsr()
    ans[cluster]["smoothness"]=__smoothness_adata(cluster_count_matrix,similarity_matrix) / similarity_matrix.shape[0]**2
    ans[cluster]["smoothness"]=ans[cluster]["smoothness"]


def calculate_EI_adata(adata,cluster_count_matrix:np.ndarray,
        cluster_processed_matrix:np.ndarray,
        gene_name:np.ndarray,
        cluster:str,
        clusters:np.ndarray,
        count_matrix:np.ndarray,
        processed_matrix:np.ndarray,
        ans:dict,
        **kwargs):
    '''
    Define a method to compute the metrics for a cluster.
    After calculating the metrics, record the metrics using ans[cluster]["metric name"] = metrics array.
    Input:
    cluster_count_matrix: matrix corresponding to the current cluster. Shape (number of cells, genes).
    cluster_processed_matrix: matrix corresponding to current cluster. Shape (number of cells, genes).
    gene_name:name of the gene, length is an np array of the number of genes.
    cluster:name of current cluster.
    clusters:list of clusters, length is an np array of cells.
    count_matrix: matrix of expression values, shape (number of cells, genes).
    processed_matrix: matrix of processed values, shape (number of cells, genes).
    ans:Dictionary to record the computed metrics.
    '''
    
    EI1=np.array(ans[cluster]['V'])/ ((ans[cluster]["other_clusters_local_mean_max"])**2+1)
    EI1=_normalize_scores_to_unit_max(EI1)
    EI2=np.divide(
        np.array(ans[cluster]['prop'],dtype=np.float64),
        np.array(ans[cluster]["prop_sum"],dtype=np.float64),
        out=np.zeros_like(np.array(ans[cluster]['prop'],dtype=np.float64)),
        where=np.array(ans[cluster]["prop_sum"]) != 0,
    )
    EI2=_normalize_scores_to_unit_max(EI2)
    
    EI=EI1*EI2

    ans[cluster]['EI']=EI


import time
is_calculate_time=True
if is_calculate_time:
    time_info_dict=defaultdict(int)
def calculate_gene_info(count_matrix:np.ndarray,processed_matrix:np.ndarray,clusters:np.ndarray,gene_name:np.ndarray,methods_list,**kwargs):
    ans=defaultdict(dict)
    tqdm0=tqdm(Counter(clusters).keys())
    for cluster in tqdm0:
        for k,method in enumerate(methods_list):
            tqdm0.set_description_str(f"cluster:{cluster}running{method.__name__}({k+1}/{len(methods_list)})")
            if is_calculate_time:
                start_time = time.time()
            method(count_matrix[clusters==cluster],
                processed_matrix[clusters==cluster],
                gene_name,
                cluster,
                clusters,
                count_matrix,
                processed_matrix,
                ans,**kwargs)
            if is_calculate_time:    
                end_time = time.time()
                time_info_dict[method.__name__]+=end_time-start_time
    if is_calculate_time:
        for i in time_info_dict.keys():
            print(f"{time_info_dict[i]:.2f} seconds used for {i}.")
    return ans



import time
is_calculate_time=True
if is_calculate_time:
    time_info_dict=defaultdict(int)
def calculate_gene_info_adata(adata,methods_list=[
    calculate_mean_and_var_adata,
    calculate_smoothness_adata,
    calculate_V_adata,
    calculate_prop_adata,
    calculate_local_mean_max_adata,
    calculate_EI_adata
],celltype_key="celltype"):
    ans=defaultdict(dict)
    genes=np.array(adata.var.index.tolist())
    barcodes=np.array(adata.obs.index.tolist())
    clusters=np.array(adata.obs[celltype_key].tolist())
    count_matrix=adata.X
    processed_matrix=adata.obsm['correctX']
    tqdm0=tqdm(Counter(clusters).keys())
    for cluster in tqdm0:
        for k,method in enumerate(methods_list):
            tqdm0.set_description_str(f"cluster:{cluster}running{method.__name__}({k+1}/{len(methods_list)})")
            if is_calculate_time:
                start_time = time.time()
            method(adata,count_matrix[clusters==cluster],
                processed_matrix[clusters==cluster],
                genes,
                cluster,
                clusters,
                count_matrix,
                processed_matrix,
                ans)
            if is_calculate_time:    
                end_time = time.time()
                time_info_dict[method.__name__]+=end_time-start_time
    if is_calculate_time:
        for i in time_info_dict.keys():
            print(f"{time_info_dict[i]:.2f} seconds used for {i}.")
    return ans

def dict_to_dataframe(data):
    # 提取并按照顺序保存指定的字段
    ordered_data = {
        'Features': data['gene_name'],
        'Mean': data['mean'],
        'Smoothness': data['smoothness'],
        'Local_max': data['other_clusters_local_mean_max'],  # 使用下标表示 max
        'V': data['V'],
        'Prop': data['prop'],
        'Prop_sum': data['prop_sum'],
        'P': data['prop_max'],
        'EI': data['EI']
    }

    # 转换为DataFrame
    df = pd.DataFrame(ordered_data)
    return df


def _marker_result_to_dataframe(result):
    """Return a marker result as a validated DataFrame for rank aggregation."""
    if isinstance(result, pd.DataFrame):
        frame = result.copy()
    elif isinstance(result, dict):
        frame = dict_to_dataframe(result)
    else:
        raise TypeError("Each batch marker result must be a DataFrame or result dictionary.")
    required = {"Features", "EI"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Batch marker result is missing required columns: {sorted(missing)}.")
    if frame["Features"].isna().any() or frame["Features"].duplicated().any():
        raise ValueError("Batch marker results require unique, non-missing feature identifiers.")
    scores = pd.to_numeric(frame["EI"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("Batch marker EI values must all be finite numeric values.")
    frame["EI"] = scores.astype(np.float64)
    return frame


def aggregate_batch_marker_ranks(batch_results):
    """Aggregate per-batch marker rankings by the median rank of each feature.

    The input is a mapping from batch identifiers to the cell-type marker results
    returned by the independent CSFeatures runs.  For each cell type, EI values
    are ranked within every eligible batch.  The output is one consensus table
    per cell type, ordered by ascending median rank.  Mean rank and mean EI are
    deterministic tie breakers and do not replace the median-rank criterion.
    """
    if not isinstance(batch_results, dict) or not batch_results:
        raise ValueError("batch_results must be a non-empty dictionary.")

    celltypes = []
    for results in batch_results.values():
        if not isinstance(results, dict):
            raise TypeError("Each batch entry must map cell types to marker results.")
        for celltype in results:
            if celltype not in celltypes:
                celltypes.append(celltype)

    consensus_results = {}
    for celltype in celltypes:
        ranked_batches = []
        reference_features = None
        for batch, results in batch_results.items():
            if celltype not in results:
                continue
            frame = _marker_result_to_dataframe(results[celltype])
            feature_set = set(frame["Features"].astype(str))
            if reference_features is None:
                reference_features = feature_set
            elif feature_set != reference_features:
                raise ValueError(
                    f"Feature identifiers differ across batches for cell type '{celltype}'."
                )

            batch_label = str(batch)
            ranked = frame[["Features", "EI"]].copy()
            ranked["Features"] = ranked["Features"].astype(str)
            ranked[f"Rank[{batch_label}]"] = ranked["EI"].rank(
                method="average", ascending=False
            )
            ranked = ranked.rename(columns={"EI": f"EI[{batch_label}]"}).set_index("Features")
            ranked_batches.append(ranked)

        if not ranked_batches:
            continue

        combined = pd.concat(ranked_batches, axis=1, join="outer")
        rank_columns = [column for column in combined if column.startswith("Rank[")]
        score_columns = [column for column in combined if column.startswith("EI[")]
        combined.insert(0, "MedianRank", combined[rank_columns].median(axis=1))
        combined.insert(1, "MeanRank", combined[rank_columns].mean(axis=1))
        combined.insert(2, "MeanEI", combined[score_columns].mean(axis=1))
        combined.insert(3, "BatchesUsed", combined[rank_columns].notna().sum(axis=1).astype(int))
        combined = combined.reset_index().rename(columns={"index": "Features"})
        combined = combined.sort_values(
            ["MedianRank", "MeanRank", "MeanEI", "Features"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        combined.insert(1, "ConsensusRank", np.arange(1, len(combined) + 1, dtype=int))
        consensus_results[celltype] = combined

    if not consensus_results:
        raise ValueError("No cell-type marker results were available for batch aggregation.")
    return consensus_results


def getMarkersEI(
    input_file: str = None, 
    clusters_file: str = None, 
    adata_path: str = None,
    batch_col: str = None,
    n_comps: int = 50,
    n_neighbors: int = 30,
    metric: str = 'euclidean',
    method_list: list = None,
    adata=None,
    celltype_key: str = "celltype",
    copy_adata: bool = True,
    random_state: int = 0
):
    """
    Calculate Expression Index (EI) to find marker genes.
    Automatically handles either CSV file inputs or an AnnData (h5ad) input.

    When ``batch_col`` is provided, each batch is analysed independently on its
    unintegrated normalized values.  Results are then aggregated by feature-wise
    median rank.  AnnData input returns ``(consensus_results, batch_results,
    batch_adatas)``; CSV input returns ``(consensus_results, batch_results)``.
    """
    # Set default methods if not explicitly provided
    # if method_list is None:
    #     method_list = [
    #         calculate_mean_and_var_adata,
    #         calculate_mean_and_var,
    #         calculate_smoothness_adata,
    #         calculate_smoothness,
    #         calculate_V,
    #         calculate_prop,
    #         calculate_local_mean_max_adata,
    #         calculate_local_mean_max,
    #         calculate_EI
    #     ]

    if adata is not None and adata_path is not None:
        raise ValueError("Provide only one of 'adata' or 'adata_path'.")

    adata_source = adata if adata is not None else adata_path
    if adata_source is not None:
        if input_file is not None or clusters_file is not None:
            raise ValueError("AnnData input cannot be combined with CSV inputs.")
        if method_list is None:
            method_list = [
                calculate_mean_and_var_adata,
                calculate_smoothness_adata,
                calculate_V_adata,
                calculate_prop_adata,
                calculate_local_mean_max_adata,
                calculate_EI_adata
            ]
        print("------ Loading AnnData ------")
        adata = _load_adata(adata_source,copy_adata=copy_adata)
        
        def run_adata_pipeline(adata_sub, batch_name="All"):
            _validate_adata(adata_sub,celltype_key=celltype_key)
            print(f"[{batch_name}] {adata_sub.n_obs} cells, {adata_sub.n_vars} genes.")
            
            print(f"[{batch_name}] ------ Step 1: Calculate PCA ------")
            effective_n_comps=_resolve_pca_component_count(
                n_comps,adata_sub.n_obs,adata_sub.n_vars
            )
            sc.pp.pca(adata_sub,n_comps=effective_n_comps,random_state=random_state)
            
            print(f"[{batch_name}] ------ Step 2: Calculate KNN & Similarity ------")
            effective_n_neighbors=_resolve_neighbor_count(n_neighbors,adata_sub.n_obs)
            sc.pp.neighbors(
                adata_sub,
                n_neighbors=effective_n_neighbors,
                use_rep='X_pca',
                metric=metric,
                random_state=random_state,
            )
            
            print(f"[{batch_name}] ------ Step 3: Adjust Expression Value ------")
            expression_matrix_correction_adata(adata_sub)
            
            print(f"[{batch_name}] ------ Step 4: Calculate EI ------")
            info = calculate_gene_info_adata(adata_sub,method_list,celltype_key=celltype_key)
            
            # Format outputs
            for k in info.keys():
                adata_sub.var[f"{k}_EI"] = info[k]['EI']
            adata_sub.uns["csfeatures"] = {
                "software_name": SOFTWARE_SHORT_NAME,
                "software_version": SOFTWARE_VERSION,
                "package_version": __version__,
                "analysis_mode": "single_cell",
                "celltype_key": str(celltype_key),
                "pca_components": int(effective_n_comps),
                "neighbors": int(effective_n_neighbors),
                "metric": str(metric),
                "random_state": int(random_state),
            }
            for k in info.keys():
                info[k] = dict_to_dataframe(info[k])
                
            return info, adata_sub

        # Branch: Single vs Multi-batch
        if batch_col is None:
            print("------ Running Pipeline (No Batch Split) ------")
            info, processed_adata = run_adata_pipeline(adata)
            print("------ Success! ------")
            return info, processed_adata
        else:
            print(f"------ Running Pipeline (Split by Batch: '{batch_col}') ------")
            if batch_col not in adata.obs:
                raise KeyError(f"AnnData.obs must contain batch column '{batch_col}'.")
            unique_batches = adata.obs[batch_col].unique()
            batch_results, batch_adatas = {}, {}
            
            for b in unique_batches:
                print(f"\n====== Processing Batch: {b} ======")
                # Subset adata for specific batch
                adata_sub = adata[adata.obs[batch_col] == b].copy()
                info_b, adata_b = run_adata_pipeline(adata_sub, batch_name=str(b))
                
                batch_results[b] = info_b
                batch_adatas[b] = adata_b
                
            consensus_results = aggregate_batch_marker_ranks(batch_results)
            print("\n------ All batches success; median-rank consensus generated! ------")
            return consensus_results, batch_results, batch_adatas


    elif input_file is not None and clusters_file is not None:
        method_list = [
            calculate_mean_and_var,
            calculate_smoothness,
            calculate_V,
            calculate_prop,
            calculate_local_mean_max,
            calculate_EI
        ]
        print("------ Loading CSV data ------")
        data = pd.read_csv(input_file)
        gene_name = np.array(list(data[data.columns[0]]))
        barcode_name = list(data.columns[1:])
        clusters = pd.read_csv(clusters_file)
        
        print("------ Data Transposition & Metadata Matching ------")
        X = data.to_numpy().T[1:].astype(np.float32)
        
        # Build mapping dictionaries
        barcode_dict = {clusters.iloc[i, 0]: clusters.iloc[i, 1] for i in range(len(clusters))}
        batch_dict = {}
        if batch_col is not None and batch_col in clusters.columns:
            batch_dict = {clusters.iloc[i, 0]: clusters.loc[i, batch_col] for i in range(len(clusters))}
            
        type_list = np.array([barcode_dict.get(i) for i in barcode_name])

        def run_csv_pipeline(X_sub, type_list_sub, batch_name="All"):
            print(f"[{batch_name}] {X_sub.shape[0]} cells, {X_sub.shape[1]} genes.")
            
            print(f"[{batch_name}] ------ Step 1: Calculate PCA ------")
            effective_n_comps = _resolve_pca_component_count(
                n_comps, X_sub.shape[0], X_sub.shape[1]
            )
            pca = PCA(n_components=effective_n_comps,random_state=random_state)
            X_pca = pca.fit_transform(X_sub)
            
            print(f"[{batch_name}] ------ Step 2: Calculate KNN Matrix ------")
            adj_matrix = compute_the_adjacency_matrix(X_pca)
            
            print(f"[{batch_name}] ------ Step 3: Calculate Similarity Matrix ------")
            similarity_matrix = compute_the_similarity_matrix(X_pca)
            
            print(f"[{batch_name}] ------ Step 4: Adjust Expression Value ------")
            matrix = similarity_matrix * adj_matrix
            matrix = matrix / np.reshape(np.sum(matrix, axis=1), [matrix.shape[0], 1])
            correct_X = expression_matrix_correction(X_sub, matrix)
            
            print(f"[{batch_name}] ------ Step 5: Calculate EI ------")
            info = calculate_gene_info(X_sub, correct_X, type_list_sub, gene_name, method_list,
                                       X_pca=X_pca, similarity_matrix=similarity_matrix)
            return info

        # Branch: Single vs Multi-batch
        if batch_col is None or not batch_dict:
            print("------ Running Pipeline (No Batch Split) ------")
            info = run_csv_pipeline(X, type_list)
            print("------ Success! ------")
            return info
        else:
            print(f"------ Running Pipeline (Split by Batch: '{batch_col}') ------")
            batch_list = np.array([batch_dict.get(i) for i in barcode_name])
            unique_batches = np.unique(batch_list)
            batch_results = {}
            
            for b in unique_batches:
                print(f"\n====== Processing Batch: {b} ======")
                # Slice indices for specific batch
                idx = np.where(batch_list == b)[0]
                batch_info = run_csv_pipeline(X[idx], type_list[idx], batch_name=str(b))
                batch_results[b] = batch_info
                
            consensus_results = aggregate_batch_marker_ranks(batch_results)
            print("\n------ All batches success; median-rank consensus generated! ------")
            return consensus_results, batch_results
            
    else:
        raise ValueError(
            "Invalid inputs. Provide 'adata', 'adata_path', or both "
            "'input_file' and 'clusters_file'."
        )


def get_spatial_MarkersEI(adata,n_comps=50,
                        n_neighbors=30,metric='euclidean',
                        spatial_key="spatial",method_list=None,
                        celltype_key="celltype",copy_adata=True,
                        random_state=0):
    # Loading data
    print("------Loading data------")
    adata=_load_adata(adata,copy_adata=copy_adata)
    _validate_adata(adata,celltype_key=celltype_key,spatial_key=spatial_key)
    if method_list is None:
        method_list=[
            calculate_mean_and_var_adata,
            calculate_smoothness_adata,
            calculate_V_adata,
            calculate_prop_adata,
            calculate_local_mean_max_adata,
            calculate_EI_adata
        ]
    print("------Data transposition------")
    print(f"{adata.X.shape[0]} cells, {adata.X.shape[1]} genes.")
    print("------Step1: Calculate PCA------")
    effective_n_comps=_resolve_pca_component_count(n_comps,adata.n_obs,adata.n_vars)
    sc.pp.pca(adata,n_comps=effective_n_comps,random_state=random_state)
    print("------Step2: Calculate the similarity matrix------")
    effective_n_neighbors=_resolve_neighbor_count(n_neighbors,adata.n_obs)
    sc.pp.neighbors(
        adata,
        n_neighbors=effective_n_neighbors,
        use_rep=spatial_key,
        metric=metric,
        random_state=random_state,
    )
    print("------Step3: Adjust the expression value------")
    expression_matrix_correction_adata(adata)
    print("------Step4: Calculate EI------")
    info=calculate_gene_info_adata(adata,method_list,celltype_key=celltype_key)
    print("------Congratulations, success!------") 
    for i in info.keys():
        adata.var[f"{i}_EI"]=info[i]['EI']
    adata.uns["csfeatures"] = {
        "software_name": SOFTWARE_SHORT_NAME,
        "software_version": SOFTWARE_VERSION,
        "package_version": __version__,
        "analysis_mode": "spatial",
        "celltype_key": str(celltype_key),
        "spatial_key": str(spatial_key),
        "pca_components": int(effective_n_comps),
        "neighbors": int(effective_n_neighbors),
        "metric": str(metric),
        "random_state": int(random_state),
    }
    for i in info.keys():
        info[i]=dict_to_dataframe(info[i])
    return info,adata
