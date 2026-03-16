import networkx as nx
from collections import defaultdict, deque
from tqdm import tqdm  # optional progress bar for large graphs


def intersection_nodes(G: nx.DiGraph) -> set:
    """
    Return the set of nodes whose total degree (in + out) exceeds 2.
    """
    return {n for n in G.nodes() if (G.in_degree(n) + G.out_degree(n)) > 2}


def dfs_paths_limited(G, source, depth_limit=10):
    """
    Yield simple directed paths starting at `source`,
    truncated at `depth_limit` edges.
    """
    stack = [(source, [source], [])]
    while stack:
        node, path, edges = stack.pop()
        if len(path) - 1 == depth_limit:  # reached limit
            yield (path, edges)
            continue
        for _, succ, key in G.out_edges(node,keys=True):
            if succ not in path:  # keep it simple
                stack.append((succ, path + [succ], edges + [key]))
        # also yield the current partial path
        yield (path, edges)

def path_score(path, intersections):
    """
    Number of distinct intersection nodes visited by the path.
    """
    # Exclude the first and last node
    interior = set(path[1:-1])
    return len(interior & intersections)

def rank_paths(path_iter, intersections):
    """
    Returns a list of (score, path) tuples sorted descending by score.
    """
    scored = []
    for (p,k) in path_iter:#tqdm(path_iter, desc="Scoring paths"):
        sc = path_score(p, intersections)
        scored.append((sc, (p,k)))
    # longer path first
    scored.sort(key=lambda x: (x[0], len(x[1][0])), reverse=True)
    return scored


def flatten_ranked_paths(G, ranked_paths):
    """
    Produce a linear ordering of edges.
    Duplicate edges are emitted only once, at the first occurrence
    (i.e., the highest-scoring path that contains them).
    """
    seen = set()  # (u, v, k) pairs already placed
    ordered_edges = []

    for _, (path, keys) in ranked_paths:
        # Convert the node list into edge tuples
        edges = [(path[i], path[i + 1], keys[i]) for i in range(len(path) - 1)]
        for e in edges:
            if e not in seen:
                ordered_edges.append(e)
                seen.add(e)

    # Append any leftover edges that never appeared in the candidate paths
    # (this can happen if you used a depth-limited search that missed some)
    for u, v, k in G.edges(keys=True):
        if (u, v, k) not in seen:
            ordered_edges.append((u, v, k))

    return ordered_edges


def sort_edges_by_max_intersections(G, depth_limit=100):
    """
    Main entry point.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Your directed graph.
    method : {"dfs", "shortest"}
        Choose how to generate candidate paths.
    depth_limit : int
        Used only when method=="dfs". Controls how deep the DFS explores.

    Returns
    -------
    List[Tuple[node, node]]
        Edges ordered according to the “max-intersection-path first” rule.
    """
    # Intersection nodes
    inters = intersection_nodes(G)

    # generate paths from every possible source node
    path_gen = (
        p for src in G.nodes() for p in dfs_paths_limited(G, src, depth_limit)
    )

    # Rank them
    ranked = rank_paths(path_gen, inters)

    # Flatten to edge order
    ordered_edges = flatten_ranked_paths(G, ranked)

    return ordered_edges
