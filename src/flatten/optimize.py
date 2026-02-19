import geopandas as gpd
import cvxpy as cp
import numpy as np
from itertools import chain
points  = gpd.read_file("temp.gpkg", layer="points")

first_segment_order = points["segment_order"].min()

first_segment_left = points[(points["segment_order"] == first_segment_order) & (points["point_position"] == "left")].sort_values(by=["point_order"])
first_segment_right = points[(points["segment_order"] == first_segment_order) & (points["point_position"] == "right")].sort_values(by=["point_order"])

def values_from_string(s: str):
    return [int(x.strip()) for x in (s[1:-1].split(",")) if len(x.strip()) > 0]
bottlenecks_left = list(map(values_from_string,first_segment_left["point_bottlenecks"].tolist()))
print("bottlenecks_left",bottlenecks_left)
bottlenecks_right = list(map(values_from_string,first_segment_right["point_bottlenecks"].tolist()))
bottlenecks = np.array(list(chain.from_iterable(bottlenecks_left)))
print("bottlenecks",bottlenecks)
max = max(bottlenecks)
print("max",max)
pairs = []
for b in range(0, max + 1):
    for i, b_l in enumerate(bottlenecks_left):
        if b in b_l:
            left = i
    for i, r_l in enumerate(bottlenecks_right):
        if b in r_l:
            right = i
    # left = np.where(b in bottlenecks_left, bottlenecks_left)
    # right = np.where(b in bottlenecks_right)
    pairs.append((left, right))
print("pairs",pairs)

# ----- data ---------------------------------------------------------
zL0 = first_segment_left["geometry"].z  # original left heights, length n
zR0 = first_segment_right["geometry"].z # original right heights, length m
# pairs = [(i1, j1), (i2, j2), ...]   # zero‑based indices of linked points
alpha = 1.0
beta  = 10.0
gamma = 0.1   # set > 0 if you want smoothness

n = len(zL0)
m = len(zR0)

# ----- variables -----------------------------------------------------
zL = cp.Variable(n)
zR = cp.Variable(m)

# ----- objective -----------------------------------------------------
obj = 0
obj += alpha * cp.sum_squares(zL - zL0)
obj += alpha * cp.sum_squares(zR - zR0)

obj += beta * cp.sum([cp.square(zL[i] - zR[j]) for i, j in pairs])

if gamma > 0:
    # second‑difference smoothness
    obj += gamma * cp.sum_squares(zL[2:] - 2*zL[1:-1] + zL[:-2])
    obj += gamma * cp.sum_squares(zR[2:] - 2*zR[1:-1] + zR[:-2])

# ----- monotonicity constraints --------------------------------------
constraints = []
constraints += [zL[i+1] >= zL[i] for i in range(n-1)]
constraints += [zR[j+1] >= zR[j] for j in range(m-1)]

# ----- solve ---------------------------------------------------------
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.OSQP)   # or any QP solver you have

# ----- results -------------------------------------------------------
zL_opt = zL.value
zR_opt = zR.value

print("Initial left heights:", np.array(zL0))
print("Optimised left heights:", zL_opt)
print("Initial right heights:", np.array(zR0))
print("Optimised right heights:", zR_opt)

import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
ax0.plot(np.arange(n), np.array(zL0), "C0.", markersize=12)
ax0.plot(np.arange(n), zL_opt, "C1.", markersize=12)
ax0.legend(("Left profile"), loc="lower right")
ax0.set_title("Left")

ax1.plot(np.arange(m), np.array(zR0), "C0.", markersize=12)
ax1.plot(np.arange(m), zR_opt, "C1.", markersize=12)
ax1.legend(("Right profile"), loc="lower right")
ax1.set_title("Right")

plt.show()