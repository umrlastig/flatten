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

logger = logging.getLogger(__name__)

# def optimize_segment(
#     segment_left: gpd.GeoDataFrame,
#     segment_right: gpd.GeoDataFrame,
#     fixed_heights: dict[Point, float | None],
#     alpha: float = 1.0,
#     beta: float = 10.0,
#     gamma: float = 0.1,
#     bottlenecks_as_string: bool = False
# ):
#     """
#     Optimize a segment with segment_order.

#     :param alpha: weight of the fidelity to the original heights
#     :type alpha: float
#     :param beta: weight of the pairwise consistency
#     :type beta: float
#     :param gamma: weight of the smoothness (gamma > 0 activates smoothness)
#     :type gamma: float
#     :param bottlenecks_as_string: true if reading from file (bottleneck info is converted to string), false otherwise
#     """
#     tol = 1e-9

#     # find the bottleneck pairs
#     def values_from_string(s: str):
#         return [int(x.strip()) for x in (s[1:-1].split(",")) if len(x.strip()) > 0]

#     bottlenecks_left = list(
#         map(values_from_string, segment_left["point_bottlenecks"].tolist())
#     ) if bottlenecks_as_string else segment_left["point_bottlenecks"].tolist()
#     bottlenecks_right = list(
#         map(values_from_string, segment_right["point_bottlenecks"].tolist())
#     ) if bottlenecks_as_string else segment_right["point_bottlenecks"].tolist()
#     bottlenecks = np.array(list(chain.from_iterable(bottlenecks_left)))
#     pairs = []
#     if len(bottlenecks) > 0:
#         # the last index of the bottleneck pairs
#         max_bottleneck_order = max(bottlenecks)
#         # zero‑based indices of linked points
#         for b in range(0, max_bottleneck_order + 1):
#             for i, b_l in enumerate(bottlenecks_left):
#                 if b in b_l:
#                     left = i
#             for i, r_l in enumerate(bottlenecks_right):
#                 if b in r_l:
#                     right = i
#             pairs.append((left, right))

#     zL0 = segment_left["geometry"].z  # original left heights, length n
#     zR0 = segment_right["geometry"].z  # original right heights, length m
#     # Fixed‑height specifications (list of (index, height))
#     fixed_left = []
#     fixed_right = []
#     for key, value in fixed_heights.items():
#         if value is not None:
#             # print("Fixed height", key, value)
#             matches = segment_left[
#                 (np.abs(segment_left["geometry"].x - key.x) < tol)
#                 & (np.abs(segment_left["geometry"].y - key.y) < tol)
#             ]
#             if len(matches) > 0:
#                 for i in matches["point_order"]:
#                     # print("Found left", i)
#                     fixed_left.append((i, value))
#             matches = segment_right[
#                 (np.abs(segment_right["geometry"].x - key.x) < tol)
#                 & (np.abs(segment_right["geometry"].y - key.y) < tol)
#             ]
#             if len(matches) > 0:
#                 for i in matches["point_order"]:
#                     # print("Found right", i)
#                     fixed_right.append((i, value))
#     n = len(zL0)
#     m = len(zR0)
#     # variables
#     zL = cp.Variable(n)
#     zR = cp.Variable(m)
#     # objectives
#     obj = 0
#     obj += alpha * cp.sum_squares(
#         zL - zL0
#     )  # fidelity to the original heights, left profile
#     obj += alpha * cp.sum_squares(
#         zR - zR0
#     )  # fidelity to the original heights, right profile
#     obj += beta * cp.sum(
#         [cp.square(zL[i] - zR[j]) for i, j in pairs]
#     )  # pairwise consistency (bottlenecks)
#     if gamma > 0:
#         # second‑difference smoothness
#         obj += gamma * cp.sum_squares(zL[2:] - 2 * zL[1:-1] + zL[:-2])
#         obj += gamma * cp.sum_squares(zR[2:] - 2 * zR[1:-1] + zR[:-2])
#     # monotonicity constraints
#     constraints = []
#     constraints += [zL[i + 1] >= zL[i] for i in range(n - 1)]
#     constraints += [zR[j + 1] >= zR[j] for j in range(m - 1)]
#     # fixed‑height equalities
#     for idx, height in fixed_left:
#         constraints.append(zL[idx] == height)
#     for idx, height in fixed_right:
#         constraints.append(zR[idx] == height)
#     # solve
#     prob = cp.Problem(cp.Minimize(obj), constraints)
#     res = prob.solve(solver=cp.OSQP, max_iter=100_000)

#     # results
#     zL_opt = zL.value
#     zR_opt = zR.value

#     if isinstance(res, float) and (math.isinf(res)):
#         logger.error(f"res = {res}")
#         logger.error(f"segment_order = {segment_left["segment_order"].tolist()[0]}")
#         logger.error(f"left = {segment_left["geometry"].tolist()}")
#         logger.error(f"right = {segment_right["geometry"].tolist()}")
#         logger.error(f"fixed_left = {fixed_left}")
#         logger.error(f"fixed_right = {fixed_right}")
#         logger.error(f"zL0 = {zL0.tolist()}")
#         logger.error(f"zR0 = {zR0.tolist()}")
#         logger.debug(f"zL_opt = {zL_opt}")
#         logger.debug(f"zR_opt = {zR_opt}")
#     if (zL_opt is None) or (zR_opt is None):
#         return None, None

#     # logger.debug(f"outshape = {np.shape(zL_opt)[0]}, {np.shape(zR_opt)[0]}") # type: ignore
#     # add new fixed height
#     for key, value in fixed_heights.items():
#         if value is None:
#             matches = segment_left[
#                 (np.abs(segment_left["geometry"].x - key.x) < tol)
#                 & (np.abs(segment_left["geometry"].y - key.y) < tol)
#             ]
#             if len(matches) > 0:
#                 for i in matches["point_order"]:
#                     fixed_heights[key] = zL_opt[i]  # type: ignore
#             matches = segment_right[
#                 (np.abs(segment_right["geometry"].x - key.x) < tol)
#                 & (np.abs(segment_right["geometry"].y - key.y) < tol)
#             ]
#             if len(matches) > 0:
#                 for i in matches["point_order"]:
#                     # print(key, "Found right", i, zR_opt[i]) # type: ignore
#                     fixed_heights[key] = zR_opt[i]  # type: ignore
#     return zL_opt, zR_opt

# def sliding_window_generator(lst, n=3):
#     for i in range(len(lst) - n + 1):
#         yield lst[i:i+n]


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
            continue

        for idx in range(len(consecutive) - 2):
            i, j, k = consecutive[idx], consecutive[idx + 1], consecutive[idx + 2]

            # Row for this triplet
            row_idx = len(rows) // 3  # Each triplet adds 3 entries

            rows.extend([row_idx, row_idx, row_idx])
            cols.extend([i, j, k])
            data.extend([1.0, -2.0, 1.0])

    num_triplets = len(rows) // 3
    return sp.coo_matrix((data, (rows, cols)), shape=(num_triplets, n)).tocsr()


def optimize_all_segments(
    points: gpd.GeoDataFrame,
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 0.1,
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
    points["unique_id"] = points.groupby("geometry", sort=False).ngroup()
    unique_points = points.drop_duplicates(subset=["geometry"], keep="first")
    assert list(unique_points["unique_id"]) == list(range(0, len(unique_points)))

    consecutive_points: list[list[int]] = []
    pairs: list[tuple[int, int]] = []
    segment_orders = sorted(set(points["segment_order"].tolist()))
    for segment_order in segment_orders:
        # print("segment_order",segment_order)
        segment_left = points[
            (points["segment_order"] == segment_order)
            & (points["point_position"] == "left")
        ].sort_values(by=["point_order"])
        segment_right = points[
            (points["segment_order"] == segment_order)
            & (points["point_position"] == "right")
        ].sort_values(by=["point_order"])
        # print("left",segment_left["unique_id"].tolist())
        # print("right",segment_right["unique_id"].tolist())
        consecutive_points.append(segment_left["unique_id"].tolist())
        consecutive_points.append(segment_right["unique_id"].tolist())
        # handling bottlenecks
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
            # print("pair",left, right)

    # original heights
    z0 = np.array(unique_points["geometry"].z)
    n = len(z0)
    # variables
    z = cp.Variable(n)
    # objectives
    obj = 0
    # fidelity to the original heights
    obj += alpha * cp.sum_squares(z - z0)
    # pairwise consistency (bottlenecks)
    obj += beta * cp.sum([cp.square(z[i] - z[j]) for i, j in pairs])
    if gamma > 0:
        # second‑difference smoothness
        # for consecutive in consecutive_points:
        #     obj += gamma * cp.sum([cp.square(z[i] - 2 * z[j] + z[k]) for i,j,k in sliding_window_generator(consecutive)])
        D = build_second_difference_matrix(n, consecutive_points)
        obj += gamma * cp.sum_squares(D @ z)
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
        res = prob.solve(solver=cp.OSQP, max_iter=50_000, eps_abs=1e-6, eps_rel=1e-6)
    except Exception as e:
        print(f"OSQP failed: {e}. Trying ECOS...")
        res = prob.solve(solver=cp.ECOS)

    # Check Status
    if prob.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimization successful. Status: {prob.status}")
        print(f"Objective value: {prob.value:.4f}")
        # results
        z_opt = z.value
        points["old_z"] = points["unique_id"].map(lambda x: z0.tolist()[x])
        points["new_z"] = points["unique_id"].map(lambda x: z_opt.tolist()[x])  # type: ignore
        xs = points.geometry.x
        ys = points.geometry.y
        zs_new = points["new_z"]

        # Reconstruct geometries
        points["geometry"] = [Point(x, y, z) for x, y, z in zip(xs, ys, zs_new)]  # type: ignore
        verify_smoothness(z_opt, consecutive_points)
        return points

    else:
        print(f"Optimization failed. Status: {prob.status}")
        print(f"Reason: {prob.status}")
        # if isinstance(res, float) and (math.isinf(res)):
        logger.error(f"res = {res}")
        logger.error(f"segment_order = {segment_left['segment_order'].tolist()[0]}")
        logger.error(f"left = {segment_left['geometry'].tolist()}")
        logger.error(f"right = {segment_right['geometry'].tolist()}")
        logger.error(f"z0 = {z0.tolist()}")
        # logger.debug(f"z_opt = {z_opt}")
        # if (z_opt is None):
        return None


def verify_smoothness(z_vals, consecutive_points):
    """Verify the matrix formulation matches the loop version"""
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
    print(f"Smoothness terms match: {loop_sum:.6f}")


# def modify_heights(
#     points: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, heights: list[float]
# ):
#     for p, z in zip(segments["geometry"], heights):
#         tol = 1e-9
#         mask = (np.abs(points["geometry"].x - p.x) < tol) & (
#             np.abs(points["geometry"].y - p.y) < tol
#         )
#         points.loc[mask, "geometry"] = Point(p.x, p.y, z)  # type: ignore

# def optimize(points: gpd.GeoDataFrame, bottlenecks_as_string: bool = False):
#     segment_orders = sorted(points["segment_order"].unique().tolist())

#     counts = points.groupby("geometry")[["segment_order"]].count()
#     shared_points = counts[counts["segment_order"] > 1]
#     fixed_heights = dict[Point, float | None]()
#     for p in shared_points.index.tolist():
#         fixed_heights[p] = None

#     temp = dict()
#     for segment_order in tqdm(segment_orders):
#         # get left and right points
#         segment_left = points[
#             (points["segment_order"] == segment_order)
#             & (points["point_position"] == "left")
#         ].sort_values(by=["point_order"])
#         segment_right = points[
#             (points["segment_order"] == segment_order)
#             & (points["point_position"] == "right")
#         ].sort_values(by=["point_order"])
#         # logger.debug(f"segment {segment_order}")
#         zL_opt, zR_opt = optimize_segment(
#             segment_left, segment_right, fixed_heights, beta=100.0, bottlenecks_as_string = bottlenecks_as_string
#         )
#         if zL_opt is not None:
#             modify_heights(points, segment_left, zL_opt.tolist())
#         if zR_opt is not None:
#             modify_heights(points, segment_right, zR_opt.tolist())
#         opt_left = points[
#             (points["segment_order"] == segment_order)
#             & (points["point_position"] == "left")
#         ].sort_values(by=["point_order"])
#         opt_right = points[
#             (points["segment_order"] == segment_order)
#             & (points["point_position"] == "right")
#         ].sort_values(by=["point_order"])
#         temp[segment_order] = (
#             segment_left["geometry"].z,
#             segment_right["geometry"].z,
#             opt_left["geometry"].z,
#             opt_right["geometry"].z,
#         )
#     return points, temp

# def main():
#     logger.info(f"{datetime.now()} Start")
#     # data
#     points = gpd.read_file("triangle_graph.gpkg", layer="points")

#     points, temp = optimize(points, True)

#     points.to_file("triangle_graph.gpkg", layer="points_optimised")

#     print(f"{datetime.now()} All done!")

#     def show_plot(show: int):
#         zL0, zR0, zL_opt, zR_opt = temp[show]
#         n = len(zL0)
#         m = len(zR0)
#         import matplotlib.pyplot as plt

#         fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
#         ax0.plot(np.arange(n), np.array(zL0), "C0.-", markersize=12)
#         ax0.plot(np.arange(n), zL_opt, "C1.-", markersize=12)
#         ax0.legend((f"Left profile for {show}"), loc="lower right")
#         ax0.set_title("Left")

#         ax1.plot(np.arange(m), np.array(zR0), "C0.-", markersize=12)
#         ax1.plot(np.arange(m), zR_opt, "C1.-", markersize=12)
#         ax1.legend((f"Right profile for {show}"), loc="lower right")
#         ax1.set_title("Right")

#         plt.show()

#     # show_plot(25)
#     # show_plot(16)

# if __name__ == "__main__":
#     logger.setLevel("DEBUG")
#     logger.addHandler(logging.StreamHandler())
#     main()
