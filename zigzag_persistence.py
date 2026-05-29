import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import networkx as nx

def zigzag_persistence_scores(returns, max_dim=1, num_steps=20):
    """
    Compute per-ETF zigzag persistence scores using graph centrality measures
    across forward and backward filtrations.
    Returns dict: ticker -> score.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    corr = returns_clean.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)

    triu_idx = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu_idx]
    if len(flat_dist) == 0:
        return {t: 0.0 for t in returns_clean.columns}
    thresholds = np.linspace(np.min(flat_dist), np.max(flat_dist), num_steps)

    # For each vertex, accumulate a composite score across thresholds
    vertex_scores = np.zeros(n)
    
    for t in thresholds:
        adj = (dist < t).astype(int)
        G = nx.from_numpy_array(adj)
        # Compute per-vertex measures
        if G.number_of_nodes() == 0:
            continue
        # Betweenness centrality (normalized)
        try:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        except:
            betweenness = {i: 0.0 for i in range(n)}
        # Clustering coefficient
        clustering = nx.clustering(G)
        # Degree
        degree = dict(G.degree())
        # Combine: higher score for vertices that are central and have high clustering
        for i in range(n):
            bt = betweenness.get(i, 0.0)
            cl = clustering.get(i, 0.0)
            deg = degree.get(i, 0)
            # Avoid division by zero: add epsilon
            score = bt * (1 + cl) * (1 + np.log(deg+1))
            vertex_scores[i] += score
    
    # Normalize by number of thresholds
    vertex_scores = vertex_scores / len(thresholds)

    # Backward filtration (inverse distances)
    dist_inv = 1.0 / (dist + 1e-12)
    thresholds_inv = np.linspace(np.min(dist_inv[dist_inv < np.inf]), np.max(dist_inv[dist_inv < np.inf]), num_steps)
    vertex_scores_back = np.zeros(n)
    for t in thresholds_inv:
        adj = (dist_inv < t).astype(int)
        G = nx.from_numpy_array(adj)
        if G.number_of_nodes() == 0:
            continue
        try:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        except:
            betweenness = {i: 0.0 for i in range(n)}
        clustering = nx.clustering(G)
        degree = dict(G.degree())
        for i in range(n):
            bt = betweenness.get(i, 0.0)
            cl = clustering.get(i, 0.0)
            deg = degree.get(i, 0)
            score = bt * (1 + cl) * (1 + np.log(deg+1))
            vertex_scores_back[i] += score
    vertex_scores_back = vertex_scores_back / len(thresholds_inv)

    final_score = (vertex_scores + vertex_scores_back) / 2
    tickers = returns_clean.columns
    return {ticker: final_score[i] for i, ticker in enumerate(tickers)}
