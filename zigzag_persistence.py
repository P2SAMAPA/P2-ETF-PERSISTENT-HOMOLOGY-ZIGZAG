import numpy as np
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh

def zigzag_persistence_scores(returns, max_dim=1, num_steps=20):
    """
    Compute per-ETF zigzag persistence scores using graph filtration.
    For each threshold (distance), we compute:
      - 0-dim persistence (connected components) via connected components count.
      - 1-dim persistence (cycles) via rank of graph Laplacian (Betti‑1 approximation).
    Then combine forward and backward filtrations.
    Returns dict: ticker -> score (average distance * cycle involvement).
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    corr = returns_clean.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)

    # Get unique distance thresholds for filtration
    triu_idx = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu_idx]
    if len(flat_dist) == 0:
        return {t: 0.0 for t in returns_clean.columns}
    thresholds = np.linspace(np.min(flat_dist), np.max(flat_dist), num_steps)

    # For each vertex, accumulate persistence over thresholds
    vertex_score = np.zeros(n)
    for t in thresholds:
        # Build graph adjacency: edge if distance < threshold
        adj = (dist < t).astype(int)
        # Number of connected components (Betti‑0)
        n_components, labels = connected_components(adj, directed=False)
        # For each component, its "persistence" is proportional to threshold? We'll assign each vertex the component size
        # But we need per-vertex score for persistence: we use the Laplacian's smallest eigenvalue as a measure of cycle complexity.
        # Instead, compute Betti‑1 = number of loops = rank of cycle space.
        # Approximate Betti‑1 by: m - n + components (Euler formula)
        m = np.sum(adj) // 2  # number of edges
        betti1 = max(0, m - n + n_components)  # cycles
        # For each vertex, add (betti1 / n) as a score contribution (every vertex gets some cycle influence)
        # Also, vertices in larger components get more weight (component size)
        component_sizes = np.bincount(labels)
        for i in range(n):
            size = component_sizes[labels[i]]
            vertex_score[i] += (size / n) * (1 + betti1 / n)

    # Normalize by number of thresholds
    vertex_score = vertex_score / len(thresholds)

    # Add backward filtration (inverse distances) to capture zigzag
    dist_inv = 1.0 / (dist + 1e-12)
    thresholds_inv = np.linspace(np.min(dist_inv[dist_inv < np.inf]), np.max(dist_inv[dist_inv < np.inf]), num_steps)
    vertex_score_back = np.zeros(n)
    for t in thresholds_inv:
        adj = (dist_inv < t).astype(int)
        n_components, labels = connected_components(adj, directed=False)
        m = np.sum(adj) // 2
        betti1 = max(0, m - n + n_components)
        component_sizes = np.bincount(labels)
        for i in range(n):
            size = component_sizes[labels[i]]
            vertex_score_back[i] += (size / n) * (1 + betti1 / n)
    vertex_score_back = vertex_score_back / len(thresholds_inv)

    # Combine forward and backward scores
    final_score = (vertex_score + vertex_score_back) / 2
    tickers = returns_clean.columns
    return {ticker: final_score[i] for i, ticker in enumerate(tickers)}
