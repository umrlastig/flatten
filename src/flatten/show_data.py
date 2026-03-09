import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

points = gpd.read_file("temp.gpkg", layer="points")

first_segment_order = points["segment_order"].min()

first_segment_left = points[
    (points["segment_order"] == first_segment_order)
    & (points["point_position"] == "left")
].sort_values(by=["point_order"])
first_segment_right = points[
    (points["segment_order"] == first_segment_order)
    & (points["point_position"] == "right")
].sort_values(by=["point_order"])

ax0.plot(
    first_segment_left["point_order"],
    first_segment_left["geometry"].z,
    "C0.",
    markersize=12,
)
ax0.legend(("Left profile"), loc="lower right")
ax0.set_title("Left")

ax1.plot(
    first_segment_right["point_order"],
    first_segment_right["geometry"].z,
    "C0.",
    markersize=12,
)
ax1.legend(("Right profile"), loc="lower right")
ax1.set_title("Right")

plt.show()
