from datetime import datetime
import geopandas as gpd
import cvxpy as cp
import numpy as np
from itertools import chain
from shapely import Point
from tqdm import tqdm


print(datetime.now(), "Start")
# data
points  = gpd.read_file("temp.gpkg", layer="points")

segment_orders = sorted(points["segment_order"].unique().tolist())

counts = points.groupby("geometry")[["segment_order"]].count()
shared_points = counts[counts["segment_order"] > 1]
fixed_heights = dict[Point, float|None]()
for p in shared_points.index.tolist():
    fixed_heights[p] = None

def optimize_segment(
        segment_left: gpd.GeoDataFrame, segment_right: gpd.GeoDataFrame, 
        fixed_heights:dict[Point, float|None],
        alpha: float = 1.0, beta: float = 10.0, gamma: float = 0.1
        ):
    """
    Optimize a segment with segment_order.
    
    :param alpha: weight of the fidelity to the original heights
    :type alpha: float
    :param beta: weight of the pairwise consistency 
    :type beta: float
    :param gamma: weight of the smoothness (gamma > 0 activates smoothness)
    :type gamma: float
    """
    # find the bottleneck pairs
    def values_from_string(s: str):
        return [int(x.strip()) for x in (s[1:-1].split(",")) if len(x.strip()) > 0]
    bottlenecks_left = list(map(values_from_string,segment_left["point_bottlenecks"].tolist()))
    bottlenecks_right = list(map(values_from_string,segment_right["point_bottlenecks"].tolist()))
    bottlenecks = np.array(list(chain.from_iterable(bottlenecks_left)))
    pairs = []
    if len(bottlenecks) > 0:
        # the last index of the bottleneck pairs
        max_bottleneck_order = max(bottlenecks)
        # zero‑based indices of linked points
        for b in range(0, max_bottleneck_order + 1):
            for i, b_l in enumerate(bottlenecks_left):
                if b in b_l:
                    left = i
            for i, r_l in enumerate(bottlenecks_right):
                if b in r_l:
                    right = i
            pairs.append((left, right))

    zL0 = segment_left["geometry"].z  # original left heights, length n
    zR0 = segment_right["geometry"].z # original right heights, length m
    # Fixed‑height specifications (list of (index, height))
    fixed_left  = []#[(0, 0.0)]
    fixed_right = []#[(0, 0.0)]
    for key, value in fixed_heights.items():
        if value is not None:
            # print("Fixed height", key, value)
            matches = segment_left[segment_left['geometry'] == key]
            if len(matches) > 0:
                for i in matches["point_order"]:
                    # print("Found left", i)
                    fixed_left.append((i, value))
            matches = segment_right[segment_right['geometry'] == key]
            if len(matches) > 0:
                for i in matches["point_order"]:
                    # print("Found right", i)
                    fixed_right.append((i, value))
    n = len(zL0)
    m = len(zR0)
    # variables
    zL = cp.Variable(n)
    zR = cp.Variable(m)
    # objectives
    obj = 0
    obj += alpha * cp.sum_squares(zL - zL0) # fidelity to the original heights, left profile
    obj += alpha * cp.sum_squares(zR - zR0) # fidelity to the original heights, right profile
    obj += beta * cp.sum([cp.square(zL[i] - zR[j]) for i, j in pairs]) # pairwise consistency
    if gamma > 0:
        # second‑difference smoothness
        obj += gamma * cp.sum_squares(zL[2:] - 2*zL[1:-1] + zL[:-2])
        obj += gamma * cp.sum_squares(zR[2:] - 2*zR[1:-1] + zR[:-2])
    # monotonicity constraints
    constraints = []
    constraints += [zL[i+1] >= zL[i] for i in range(n-1)]
    constraints += [zR[j+1] >= zR[j] for j in range(m-1)]
    # fixed‑height equalities
    for idx, height in fixed_left:
        constraints.append(zL[idx] == height)
    for idx, height in fixed_right:
        constraints.append(zR[idx] == height)
    # solve
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.OSQP, max_iter = 100_000)
    # results
    zL_opt = zL.value
    zR_opt = zR.value
    # print("Initial left heights:", np.array(zL0))
    # print("Optimised left heights:", zL_opt)
    # print("Initial right heights:", np.array(zR0))
    # print("Optimised right heights:", zR_opt)
    # add new fixed height
    for key, value in fixed_heights.items():
        if value is None:
            matches = segment_left[segment_left['geometry'] == key]
            if len(matches) > 0:
                for i in matches["point_order"]:
                    # print(key, "Found left", i, zL_opt[i]) # type: ignore
                    fixed_heights[key] = zL_opt[i] # type: ignore
            matches = segment_right[segment_right['geometry'] == key]
            if len(matches) > 0:
                for i in matches["point_order"]:
                    # print(key, "Found right", i, zR_opt[i]) # type: ignore
                    fixed_heights[key] = zR_opt[i] # type: ignore
    return zL_opt, zR_opt

# fixed_heights[Point(1038743.49974799994379282, 6292847.30473599955439568, 2.3120273444715651)] = 0.0
# fixed_heights[Point(1038740.18830669217277318, 6292857.84114016033709049, 2.20507867031959792)] = 0.0
# print("fixed_heights",fixed_heights)

def modify_heights(segments: gpd.GeoDataFrame, heights: list[float]):
    print("init",segments["geometry"].z)
    points = [Point(p.x, p.y, z) for p, z in zip(segments["geometry"], heights)]
    return points
    # segments["geometry"] = segments["geometry"].apply(lambda p: Point(p.x, p.y, ))
    # segments["geometry"] = points

for segment_order in tqdm(segment_orders):
    # segment_order: int = first_segment_order
    # get left and right points
    # print(datetime.now(), "segment_order", segment_order)
    segment_left = points[(points["segment_order"] == segment_order) & (points["point_position"] == "left")].sort_values(by=["point_order"])
    segment_right = points[(points["segment_order"] == segment_order) & (points["point_position"] == "right")].sort_values(by=["point_order"])
    zL_opt, zR_opt = optimize_segment(segment_left, segment_right, fixed_heights = fixed_heights, beta = 100.0)
    if zL_opt is not None:
        # new_points = modify_heights(segment_left, zL_opt.tolist())
        for p, z in zip(segment_left["geometry"], zL_opt.tolist()):
            points.loc[points['geometry'] == p, 'geometry'] = Point(p.x, p.y, z) # type: ignore
        # print("modif",segments["geometry"].z)
    if zR_opt is not None:
        # modify_heights(segment_right, zR_opt.tolist())
        for p, z in zip(segment_right["geometry"], zR_opt.tolist()):
            points.loc[points['geometry'] == p, 'geometry'] = Point(p.x, p.y, z) # type: ignore


points.to_file("temp.gpkg", layer="points_optimised")

print(datetime.now(), "All done!")
# zL0 = segment_left["geometry"].z  # original left heights, length n
# zR0 = segment_right["geometry"].z # original right heights, length m
# n = len(zL0)
# m = len(zR0)
# for key, value in fixed_heights.items():
#     if value is not None:
#         print("Final Fixed height", key, value)

# import matplotlib.pyplot as plt

# fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
# ax0.plot(np.arange(n), np.array(zL0), "C0.-", markersize=12)
# ax0.plot(np.arange(n), zL_opt, "C1.-", markersize=12)
# ax0.legend(("Left profile"), loc="lower right")
# ax0.set_title("Left")

# ax1.plot(np.arange(m), np.array(zR0), "C0.-", markersize=12)
# ax1.plot(np.arange(m), zR_opt, "C1.-", markersize=12)
# ax1.legend(("Right profile"), loc="lower right")
# ax1.set_title("Right")

# plt.show()