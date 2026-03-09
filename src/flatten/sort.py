import networkx as nx
from collections import defaultdict, deque
from tqdm import tqdm   # optional progress bar for large graphs

def intersection_nodes(G: nx.DiGraph) -> set:
    """
    Return the set of nodes whose total degree (in + out) exceeds 2.
    """
    return {n for n in G.nodes()
            if (G.in_degree(n) + G.out_degree(n)) > 2}

def dfs_paths_limited(G, source, depth_limit=10):
    """
    Yield simple directed paths starting at `source`,
    truncated at `depth_limit` edges.
    """
    stack = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if len(path) - 1 == depth_limit:   # reached limit
            yield path
            continue
        for succ in G.successors(node):
            if succ not in path:            # keep it simple
                stack.append((succ, path + [succ]))
        # also yield the current partial path
        yield path

def shortest_path_candidates(G):
    """
    Produce a list of shortest directed paths for every reachable pair.
    """
    candidates = []
    for u in G.nodes():
        sp = nx.single_source_shortest_path(G, u)
        for v, p in sp.items():
            if len(p) > 1:          # ignore trivial single‑node paths
                candidates.append(p)
    return candidates

def path_score(path, intersections):
    """
    Number of distinct intersection nodes visited by the path.
    """
    # Exclude the first and last node if you don’t want to count
    # endpoints that happen to be intersections; adjust as needed.
    interior = set(path[1:-1])
    return len(interior & intersections)

def rank_paths(G, path_iter, intersections):
    """
    Returns a list of (score, path) tuples sorted descending by score.
    """
    scored = []
    for p in tqdm(path_iter, desc="Scoring paths"):
        sc = path_score(p, intersections)
        scored.append((sc, p))
    # Highest‑score first; tie‑breaker: longer path first (optional)
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return scored

def flatten_ranked_paths(ranked_paths):
    """
    Produce a linear ordering of edges.
    Duplicate edges are emitted only once, at the first occurrence
    (i.e., the highest‑scoring path that contains them).
    """
    seen = set()                 # (u, v) pairs already placed
    ordered_edges = []

    for score, path in ranked_paths:
        # Convert the node list into edge tuples
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        for e in edges:
            if e not in seen:
                ordered_edges.append(e)
                seen.add(e)

    # Append any leftover edges that never appeared in the candidate paths
    # (this can happen if you used a depth‑limited search that missed some)
    for u, v in G.edges():
        if (u, v) not in seen:
            ordered_edges.append((u, v))

    return ordered_edges

def sort_edges_by_max_intersections(G,
                                    method="dfs",
                                    depth_limit=10):
    """
    Main entry point.

    Parameters
    ----------
    G : nx.DiGraph
        Your directed graph.
    method : {"dfs", "shortest"}
        Choose how to generate candidate paths.
    depth_limit : int
        Used only when method=="dfs". Controls how deep the DFS explores.

    Returns
    -------
    List[Tuple[node, node]]
        Edges ordered according to the “max‑intersection‑path first” rule.
    """
    # 1️⃣  Intersection nodes
    inters = intersection_nodes(G)

    # 2️⃣  Candidate paths
    if method == "dfs":
        # generate paths from every possible source node
        path_gen = (p for src in G.nodes()
                       for p in dfs_paths_limited(G, src, depth_limit))
    elif method == "shortest":
        path_gen = shortest_path_candidates(G)
    else:
        raise ValueError("method must be 'dfs' or 'shortest'")

    # 3️⃣  Rank them
    ranked = rank_paths(G, path_gen, inters)

    # 4️⃣  Flatten to edge order
    ordered_edges = flatten_ranked_paths(ranked)

    return ordered_edges

# Build a toy graph
G = nx.DiGraph()
edges = [
    (1, 2), (2, 3), (3, 4), (4, 5),   # a straight chain
    (2, 6), (6, 7), (7, 4),           # creates a junction at 4 (deg>2)
    (5, 8), (8, 9), (9, 10)           # another branch
]
G.add_edges_from(edges)