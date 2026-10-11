"""Candidate selection, never using the target photograph.

medoid           masked arms: the candidate closest to the mean SSCD feature
extract_cliques  free-form: SSCD cliques at a fixed threshold, N0 largest kept,
                 one medoid each (Appendix A.4)
"""
import multiprocessing

import networkx as nx
import numpy as np
import torch

import config as C

CLIQUE_TIMEOUT = 30          # seconds per clique search, as in FineXtract


def medoid(features):
    """Index of the feature vector closest to the mean of all of them."""
    f = torch.as_tensor(np.asarray(features)).float()
    return int(torch.argmin(torch.norm(f - f.mean(dim=0), dim=1)).item())


def _greedy_cliques(sims, threshold, min_size):
    """Connect pairs with similarity >= threshold; repeatedly take the largest
    maximal clique and remove it, while it has at least min_size members.
    The cliques are therefore disjoint."""
    n = sims.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                g.add_edge(i, j)
    found = []
    while g.number_of_nodes() > 0:
        cliques = list(nx.find_cliques(g))
        if not cliques:
            break
        best = max(cliques, key=len)
        if len(best) < min_size:
            break
        found.append(best)
        g.remove_nodes_from(best)
    return found


def _with_timeout(fn, args, timeout):
    def target(q):
        q.put(fn(*args))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    return q.get() if not q.empty() else None


def extract_cliques(features, n0, threshold=C.CLIQUE_THRESHOLD):
    """Returns (indices of the extracted images, number of cliques, threshold used).

    Pools with at most n0 images are returned whole. Otherwise cliques of at
    least two images are formed at `threshold`; if fewer than n0 form, the
    threshold is lowered in steps of 0.05 down to 0.2, then the same sweep is
    repeated allowing singletons, and the setting yielding the most cliques is
    used. The n0 largest cliques are kept and each contributes its medoid."""
    f = torch.as_tensor(np.asarray(features)).float()
    n = f.shape[0]
    if n <= n0:
        return list(range(n)), n, None
    fn = torch.nn.functional.normalize(f, dim=1)
    sims = (fn @ fn.T).numpy()
    np.fill_diagonal(sims, 0.0)

    best = None                                   # (cliques, threshold, min_size)
    for min_size in (2, 1):
        th = threshold
        while th >= 0.20 - 1e-9:
            got = _with_timeout(_greedy_cliques, (sims, th, min_size), CLIQUE_TIMEOUT)
            if got is not None:
                if len(got) >= n0:
                    best = (got, th, min_size)
                    break
                if best is None or len(got) > len(best[0]):
                    best = (got, th, min_size)
            th = round(th - 0.05, 2)
        if best and len(best[0]) >= n0:
            break
    if not best or not best[0]:
        return list(range(n)), 0, None
    cliques, th, _ = best
    cliques = sorted(cliques, key=len, reverse=True)[:n0]
    picks = []
    for cl in cliques:
        sub = f[cl]
        picks.append(cl[int(torch.argmin(torch.norm(sub - sub.mean(0), dim=1)).item())])
    return picks, len(cliques), round(th, 2)
