import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

def zigzag_persistence_scores(returns, max_dim=1, num_steps=20):
    """
    Compute per-ETF zigzag persistence using harmonic centrality variation across filtrations.
    For each threshold, compute harmonic centrality = sum_{j != i} 1/distance(i,j) for the graph.
    Then compute the standard deviation of this centrality across forward and backward filtrations.
    High std = ETF's topological role changes significantly with scale = high persistence.
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

    # For each vertex, store harmonic centrality at each threshold
    centralities = np.zeros((n, num_steps))
    for step, thresh in enumerate(thresholds):
        # Build graph adjacency: edge if distance < thresh
        adj = (dist < thresh).astype(float)
        # Compute harmonic centrality: sum_{j != i} 1 / distance(i,j) if edge exists, else 0
        harm = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and adj[i,j] > 0:
                    harm[i] += 1.0 / (dist[i,j] + 1e-12)
        centralities[:, step] = harm

    # Forward variation: std across steps
    forward_std = np.std(centralities, axis=1)

    # Backward filtration: invert distances
    dist_inv = 1.0 / (dist + 1e-12)
    thresholds_inv = np.linspace(np.min(dist_inv[dist_inv < np.inf]), np.max(dist_inv[dist_inv < np.inf]), num_steps)
    centralities_back = np.zeros((n, num_steps))
    for step, thresh in enumerate(thresholds_inv):
        adj = (dist_inv < thresh).astype(float)
        harm = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and adj[i,j] > 0:
                    harm[i] += 1.0 / (dist_inv[i,j] + 1e-12)
        centralities_back[:, step] = harm
    backward_std = np.std(centralities_back, axis=1)

    final_scores = (forward_std + backward_std) / 2.0
    tickers = returns_clean.columns
    return {ticker: final_scores[i] for i, ticker in enumerate(tickers)}
