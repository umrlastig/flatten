import pytest
from shapely import Polygon, LineString
from flatten.utils import (
    find_projection,
    get_bottlenecks,
    split_by_first_false,
    merge_profile_points,
    get_profiles,
)
import geopandas as gpd


def test_projection():
    triangle = Polygon(((0.0, 0.0), (0.0, 2.0), (1.0, 2.0), (0.0, 0.0)))
    segment = LineString(((0.0, 0.0), (0.0, 2.0)))
    line = find_projection(triangle, segment)
    print(line)
    assert not line
    triangle = Polygon(((0.0, 0.0), (0.0, 2.0), (2.0, 1.0), (0.0, 0.0)))
    line = find_projection(triangle, segment)
    print(line)
    assert line
    assert line == LineString(((0.0, 1.0), (2.0, 1.0)))


def test_get_bottlenecks():
    t1 = Polygon(((0.0, 0.0), (0.0, 2.0), (2.0, 1.0), (0.0, 0.0)))
    t2 = Polygon(((0.0, 0.0), (2.0, 1.0), (2.0, 0.0), (0.0, 0.0)))
    l1 = LineString(((0.0, 0.0), (0.0, 2.0)))
    b1 = LineString(((0, 1), (2, 1)))
    triangles = gpd.GeoDataFrame(
        {"triangle_id": [0, 1], "geometry": [t1, t2]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {"segment_id": [0], "constraint": [True], "geometry": [l1]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 1
    assert bottlenecks.at[0, "geometry"] == b1
    l2 = LineString(((2.0, 0.0), (2.0, 1.0)))
    segments = gpd.GeoDataFrame(
        {"segment_id": [0, 1], "constraint": [True, True], "geometry": [l1, l2]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 1
    t3 = Polygon(((0.0, 2.0), (2.0, 3.0), (2.0, 1.0), (0.0, 2.0)))
    l3 = LineString(((2.0, 1.0), (2.0, 3.0)))
    b2 = LineString(((2, 2), (0, 2)))
    triangles = gpd.GeoDataFrame(
        {"triangle_id": [0, 1, 2], "geometry": [t1, t2, t3]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {
            "segment_id": [0, 1, 2],
            "constraint": [True, True, True],
            "geometry": [l1, l2, l3],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 2
    assert bottlenecks.at[0, "geometry"] == b1
    assert bottlenecks.at[2, "geometry"] == b2


def test_split_by_first_false():
    seq = [("a", True, 1), ("b", True, 2), ("c", False, 3)]
    res = split_by_first_false(seq, False)
    print(res)
    assert res == ([("a", True, 1), ("b", True, 2)], [("c", False, 3)])
    res = split_by_first_false(seq, True)
    print(res)
    assert res == ([], [("a", True, 1), ("b", True, 2), ("c", False, 3)])
    seq = [("a", True, 1), ("b", True, 2), ("c", False, 3), ("d", True, 4)]
    res = split_by_first_false(seq, True)
    print(res)
    assert res == ([("d", True, 4)], [("a", True, 1), ("b", True, 2), ("c", False, 3)])


def test_merge_profile_points():
    points: list[tuple[tuple[float, float], bool, int | None]] = [
        ((0.0, 0.0), True, 1),
        ((0.0, 0.0), False, 2),
        ((0.0, 0.0), False, 3),
    ]
    merged = merge_profile_points(points)
    print(merged)
    assert merged == [((0.0, 0.0), True, [1, 2, 3])]
    points: list[tuple[tuple[float, float], bool, int | None]] = [
        ((0.0, 0.0), False, 1),
        ((0.0, 0.0), False, 2),
        ((0.0, 0.0), False, 3),
        ((0.0, 1.0), False, 4),
    ]
    merged = merge_profile_points(points)
    print(merged)
    assert merged == [((0.0, 0.0), False, [1, 2, 3]), ((0.0, 1.0), False, [4])]


def test_get_profiles():
    s1 = LineString(((0.0, 0.0), (0.0, 2.0)))
    s2 = LineString(((2.0, 0.0), (0.0, 3.0)))
    s3 = LineString(((0.0, 4.0), (3.0, 0.0)))
    l1 = LineString(((-1.0, 1.0), (4.0, 1.0)))
    l2 = LineString(((-1.0, 0.0), (-1.0, 1.0)))
    segments = gpd.GeoDataFrame(
        {
            "segment_id": [0, 1, 2],
            "constraint": [False, False, False],
            "type": ["unconstrained", "bottleneck", "unconstrained"],
            "n_segments": [1, 1, 1],
            "geometry": [s1, s2, s3],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    left, right = get_profiles(segments, l1)  # type: ignore
    print(left)
    assert left == [
        ((0.0, 2.0), False, None),
        ((0.0, 3.0), False, 0),
        ((0.0, 4.0), False, None),
    ]
    print(right)
    assert right == [
        ((0.0, 0.0), False, None),
        ((2.0, 0.0), False, 0),
        ((3.0, 0.0), False, None),
    ]

    profiles = get_profiles(segments, l2)  # type: ignore
    assert profiles is None
