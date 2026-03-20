import pytest
from shapely import Polygon, LineString
from flatten.orient_triangle_graph import (
    split_cycle,
    get_oriented_graph,
    get_nodes_accessible_from,
)
import geopandas as gpd


def test_split_cycle():
    cycle = ["a", "b", "c", "d", "e"]
    split_nodes = set(["b", "d"])
    split = split_cycle(cycle, split_nodes)
    print(split)
    assert len(split) == 2
    assert split == [["b", "c", "d"], ["d", "e", "a", "b"]]


def test_get_oriented_graph():
    triangles_data = [
        {
            "triangle_id": 0,
            "geometry": Polygon([(0, 0, 10), (10, 0, 20), (10, 10, 30)]),
        },
        {
            "triangle_id": 1,
            "geometry": Polygon([(0, 0, 10), (10, 10, 30), (0, 10, 30)]),
        },
        {
            "triangle_id": 2,
            "geometry": Polygon([(0, 10, 30), (10, 10, 30), (5, 20, 40)]),
        },
        {
            "triangle_id": 3,
            "geometry": Polygon([(0, 10, 30), (5, 20, 40), (0, 20, 40)]),
        },
        {
            "triangle_id": 4,
            "geometry": Polygon([(10, 10, 30), (10, 20, 40), (5, 20, 40)]),
        },
        {
            "triangle_id": 5,
            "geometry": Polygon([(0, 20, 40), (5, 20, 40), (5, 30, 50)]),
        },
        {
            "triangle_id": 6,
            "geometry": Polygon([(5, 20, 40), (10, 20, 40), (5, 30, 50)]),
        },
    ]
    triangles_gdf = gpd.GeoDataFrame(triangles_data, crs="EPSG:4326")
    segments_data = [
        {
            "segment_id": 0,
            "cleabs": "A",
            "geometry": LineString([(0, 30, 40), (5, 10, 20), (5, 0, 0)]),
        },
        {
            "segment_id": 1,
            "cleabs": "B",
            "geometry": LineString([(10, 20, 40), (5, 10, 20)]),
        },
    ]
    segments_gdf = gpd.GeoDataFrame(segments_data, crs="EPSG:4326")
    graph, edge_gdf, has_no_cycle = get_oriented_graph(triangles_gdf, segments_gdf)
    print(graph)
    assert graph.has_edge(1, 0)
    assert graph.has_edge(2, 1)
    assert graph.has_edge(3, 2)
    assert graph.has_edge(4, 2)
    assert graph.has_edge(5, 3)
    assert graph.has_edge(6, 4)
    print(edge_gdf)
    print(has_no_cycle)
    assert has_no_cycle
    assert get_nodes_accessible_from(graph, 0, set()) == 0
    assert get_nodes_accessible_from(graph, 1, set()) == 1
    assert get_nodes_accessible_from(graph, 2, set()) == 2
    assert get_nodes_accessible_from(graph, 3, set()) == 3
    assert get_nodes_accessible_from(graph, 4, set()) == 3
    assert get_nodes_accessible_from(graph, 5, set()) == 4
    assert get_nodes_accessible_from(graph, 6, set()) == 6
    # TODO somehow, it is arbitrary that the link is from 6 to 5 and not the other way around...
