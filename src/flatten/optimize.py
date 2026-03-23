from datetime import datetime
import geopandas as gpd
import cvxpy as cp
import scipy.sparse as sp
import numpy as np
from itertools import chain
from shapely import Point
from tqdm import tqdm
import math
import logging
import networkx as nx

from flatten.triangle_graph import get_triangle_graph_as_nx

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.addHandler(logging.StreamHandler())

def build_pairwise_difference_matrix(n, pairs):
    """
    Build sparse matrix D where D @ z gives all pairwise differences.
    Each row corresponds to one pair (i, j) with coefficients [1, -1].
    """
    num_pairs = len(pairs)
    rows = []
    cols = []
    data = []
    
    for row_idx, (i, j) in enumerate(pairs):
        rows.extend([row_idx, row_idx])
        cols.extend([i, j])
        data.extend([1.0, -1.0])
    
    return sp.coo_matrix((data, (rows, cols)), shape=(num_pairs, n)).tocsr()

def build_second_difference_matrix(n, consecutive_points):
    """
    Build sparse matrix D where D @ z gives all second differences.
    Each row corresponds to one triplet (i, j, k) with coefficients [1, -2, 1].
    """
    rows = []
    cols = []
    data = []

    for consecutive in consecutive_points:
        if len(consecutive) < 3:
            # not enough points, no second difference
            continue
        # iterate through points 3 by 3
        for idx in range(len(consecutive) - 2):
            i, j, k = consecutive[idx], consecutive[idx + 1], consecutive[idx + 2]
            # Row for this triplet
            row_idx = len(rows) // 3  # Each triplet adds 3 entries
            rows.extend([row_idx, row_idx, row_idx])
            cols.extend([i, j, k])
            data.extend([1.0, -2.0, 1.0])

    return sp.coo_matrix((data, (rows, cols)), shape=(len(rows)//3, n)).tocsr()

def build_normal_alignment_matrix(n, triangles):
    """
    Build matrix where each row represents a triangle's normal constraint.
    We use a simplified version using the second difference between points.
    """
    rows = []
    cols = []
    data = []
    
    for tri_idx, (i, j, k) in enumerate(triangles):
        # Simplified: penalize deviation from planar surface
        rows.extend([tri_idx, tri_idx, tri_idx])
        cols.extend([i, j, k])
        data.extend([1.0, -2.0, 1.0])  # Second difference approximation
    
    return sp.coo_matrix((data, (rows, cols)), shape=(len(rows)//3, n)).tocsr()

def build_normal_consistency_matrix(n, triangles, adjacency):
    """
    Penalize differences between normals of adjacent triangles.
    """
    num_edges = len(adjacency)
    rows = []
    cols = []
    data = []
    
    for edge_idx, (tri1, tri2) in enumerate(adjacency):
        # Add terms that encourage similar z gradients
        # This approximates normal similarity
        for idx in range(3):
            rows.extend([edge_idx, edge_idx])
            cols.extend([triangles[tri1][idx], triangles[tri2][idx]])
            data.extend([1.0, -1.0])
    # for r, c, d in zip(rows, cols, data):
    #     logger.debug(f"{r} {c} {d}")
    return sp.coo_matrix((data, (rows, cols)), shape=(num_edges, n)).tocsr()

def optimize_all_segments(
    points: gpd.GeoDataFrame,
    triangles: gpd.GeoDataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    delta: float = 0.0,
    epsilon: float = 10.0
) -> gpd.GeoDataFrame | None:
    """
    Optimize all segments with segment_order.

    :param alpha: weight of the fidelity to the original heights
    :type alpha: float
    :param beta: weight of the pairwise consistency
    :type beta: float
    :param gamma: weight of the smoothness (gamma > 0 activates smoothness)
    :type gamma: float
    """
    logger.info(f"{datetime.now()} - optimize_all_segments - building data structure")
    points["unique_id"] = points.groupby("geometry", sort=False).ngroup()
    unique_points = points.drop_duplicates(subset=["geometry"], keep="first")
    assert list(unique_points["unique_id"]) == list(range(0, len(unique_points)))

    consecutive_points: list[list[int]] = []
    pairs: list[tuple[int, int]] = []
    segment_orders = sorted(set(points["segment_order"].tolist()))
    for segment_order in segment_orders:
        segment_left = points[
            (points["segment_order"] == segment_order)
            & (points["point_position"] == "left")
        ].sort_values(by=["point_order"])
        segment_right = points[
            (points["segment_order"] == segment_order)
            & (points["point_position"] == "right")
        ].sort_values(by=["point_order"])
        # build consecutive points from each segment
        consecutive_points.append(segment_left["unique_id"].tolist())
        consecutive_points.append(segment_right["unique_id"].tolist())
        # handling bottlenecks (as pairs)
        bottlenecks_left = segment_left["point_bottlenecks"].tolist()
        bottlenecks_right = segment_right["point_bottlenecks"].tolist()
        bottlenecks = sorted(set(list(chain.from_iterable(bottlenecks_left))))
        for b in bottlenecks:
            for i, b_l in enumerate(bottlenecks_left):
                if b in b_l:
                    left = segment_left.iloc[i]["unique_id"]
            for i, r_l in enumerate(bottlenecks_right):
                if b in r_l:
                    right = segment_right.iloc[i]["unique_id"]
            pairs.append((left, right))
    logger.info(f"{datetime.now()} - optimize_all_segments - triangles")
    triangles = triangles.reset_index(drop=True)
    triangles['split_triangle_id'] = triangles.index
    triangles_idx = []
    triangle_map = {}
    tol = 1e-9
    for idx, triangle_geom in enumerate(triangles.geometry):
        coords = triangle_geom.exterior.coords # type: ignore
        triangle_indices = []
        for coord in coords[:-1]:
            mask = (np.abs(unique_points.geometry.x - coord[0]) < tol) & (
                np.abs(unique_points.geometry.y - coord[1]) < tol
            )
            matches = unique_points[mask]
            if any(matches):
                # logger.info(f"matches = {matches}")
                if len(matches) > 0:
                    # logger.info(f'matches["unique_id"] = {matches["unique_id"]}')
                    # logger.info(f'matches.index[0] = {matches.index[0]}')
                    # logger.info(f'matches.at[0,"unique_id"] = {matches.at[0,"unique_id"]}')
                    triangle_indices.append(
                        # matches.index[0]
                        matches["unique_id"].tolist()[0]
                    )
        if len(triangle_indices) == 3:
            triangle_map[idx] = len(triangles_idx)
            triangles_idx.append(triangle_indices)
            # logger.info(f"Triangle {idx} => {len(triangles_idx)}: {triangle_indices}")
        # else:
        #     logger.error(f"{len(triangle_indices)} indices found for {triangle_geom}")
    logger.info(f"{datetime.now()} - triangles_idx {len(triangles_idx)}")
    logger.info(f"{datetime.now()} - optimize_all_segments - adjacency")
    graph = get_triangle_graph_as_nx(triangles,'split_triangle_id')
    # for u, v in graph.edges:
    #     tri1 = triangles.iloc[u]
    #     tri2 = triangles.iloc[v]
    #     logger.debug(f"{u} {v} {tri1["geometry"]} {tri2["geometry"]}")
    adjacency = [(triangle_map[u],triangle_map[v]) for u, v in graph.edges if (u in triangle_map) and (v in triangle_map)]
    logger.info(f"{datetime.now()} - optimize_all_segments - creating problem")
    # original heights
    z0 = np.array(unique_points["geometry"].z)
    n = len(z0)
    # variables
    z = cp.Variable(n)
    # objectives
    obj = 0
    # fidelity to the original heights
    obj += alpha * cp.sum_squares(z - z0)
    if beta > 0:
        # Build difference matrix for pairwise consistency (bottlenecks)
        D_pairs = build_pairwise_difference_matrix(n, pairs)
        obj += beta * cp.sum_squares(D_pairs @ z)
    if gamma > 0:
        # second‑difference smoothness
        D_smoothness = build_second_difference_matrix(n, consecutive_points)
        obj += gamma * cp.sum_squares(D_smoothness @ z)

    if delta > 0:
        # penalize non flat triangles
        N_matrix = build_normal_alignment_matrix(n, triangles_idx)
        if (N_matrix.shape is not None) and (N_matrix.shape[0] > 0):
            obj += delta * cp.sum_squares(N_matrix @ z)

    if epsilon > 0:
        # ajacent triangles
        D_normal = build_normal_consistency_matrix(n, triangles_idx, adjacency)
        if (D_normal.shape is not None) and (D_normal.shape[0] > 0):
            obj += epsilon * cp.sum_squares(D_normal @ z)

    # monotonicity constraints
    constraints = []
    for consecutive in consecutive_points:
        constraints += [
            z[consecutive[idx + 1]] >= z[consecutive[idx]]
            for idx in range(len(consecutive) - 1)
        ]
    # solve
    prob = cp.Problem(cp.Minimize(obj), constraints)
    # Try OSQP first (fast for QP), fallback to ECOS if needed
    try:
        logger.info(f"{datetime.now()} - optimize_all_segments - solve")
        res = prob.solve(solver=cp.OSQP, max_iter=100_000, eps_abs=1e-6, eps_rel=1e-6)
    except Exception as e:
        logger.error(f"OSQP failed: {e}. Trying ECOS...")
        res = prob.solve(solver=cp.ECOS, max_iter=100_000, eps_abs=1e-6, eps_rel=1e-6)

    # Check Status
    if prob.status in ["optimal", "optimal_inaccurate"]:
        logger.info(f"{datetime.now()} - Optimization successful. Status: {prob.status}")
        logger.info(f"{datetime.now()} - Objective value: {prob.value:.4f}")
        # results
        z_opt = z.value
        points["old_z"] = points["unique_id"].map(lambda x: z0.tolist()[x])
        points["new_z"] = points["unique_id"].map(lambda x: z_opt.tolist()[x])  # type: ignore
        xs = points.geometry.x
        ys = points.geometry.y
        zs_new = points["new_z"]

        # Reconstruct geometries
        points["geometry"] = [Point(x, y, z) for x, y, z in zip(xs, ys, zs_new)]  # type: ignore
        verify_pairwise_consistency(z_opt, pairs)
        verify_smoothness(z_opt, consecutive_points)
        # TODO verify all terms?
        return points

    else:
        logger.error(f"{datetime.now()} - Optimization failed. Status: {prob.status}")
        logger.error(f"{datetime.now()} - Reason: {prob.status}")
        logger.error(f"{datetime.now()} - res = {res}")
        return None

def verify_smoothness(z_vals, consecutive_points):
    """Verify the matrix formulation matches the loop version."""
    # Loop version
    loop_sum = 0
    for consecutive in consecutive_points:
        if len(consecutive) < 3:
            continue
        for idx in range(len(consecutive) - 2):
            i, j, k = consecutive[idx], consecutive[idx + 1], consecutive[idx + 2]
            loop_sum += (z_vals[i] - 2 * z_vals[j] + z_vals[k]) ** 2

    # Matrix version
    D = build_second_difference_matrix(len(z_vals), consecutive_points)
    matrix_sum = np.sum((D @ z_vals) ** 2)

    assert abs(loop_sum - matrix_sum) < 1e-6, "Formulations don't match!"
    logger.info(f"{datetime.now()} - Smoothness terms match: {loop_sum:.6f}")

def verify_pairwise_consistency(z_vals, pairs):
    """Verify that the vectorized and loop versions give same result."""
    # Loop version
    loop_sum = sum((z_vals[i] - z_vals[j])**2 for i, j in pairs)
    
    # Vectorized version
    D = build_pairwise_difference_matrix(len(z_vals), pairs)
    vec_sum = np.sum((D @ z_vals)**2)
    
    assert abs(loop_sum - vec_sum) < 1e-6, "Results don't match!"
    logger.info(f"{datetime.now()} - Pairwise terms match: {loop_sum:.6f}")

