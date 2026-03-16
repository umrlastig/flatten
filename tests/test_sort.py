import pytest
from shapely import Polygon, LineString
import geopandas as gpd
import networkx as nx

from flatten.sort import sort_edges_by_max_intersections

def test_sort():
    G = nx.MultiDiGraph()
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),  # a straight chain
        (2, 6),
        (6, 7),
        (7, 8),
        (8, 4),  # creates a junction at 4 (deg>2)
        (5, 9),
        (9, 10),
        (10, 11),  # another branch
    ]
    G.add_edges_from(edges)
    sorted = sort_edges_by_max_intersections(G)
    print(sorted)
    assert sorted == [(1, 2, 0), (2, 6, 0), (6, 7, 0), (7, 8, 0), (8, 4, 0), (4, 5, 0), (5, 9, 0), (9, 10, 0), (10, 11, 0), (2, 3, 0), (3, 4, 0)]
    G = nx.MultiDiGraph()
    edges = [
        (1, 2, {"attr":"A"}),
        (2, 3, {"attr":"B"}),
        (3, 4, {"attr":"C"}),
        (4, 5, {"attr":"D"}),  # a straight chain
        (2, 6, {"attr":"E"}),
        (6, 7, {"attr":"F"}),
        (7, 8, {"attr":"G"}),
        (8, 4, {"attr":"H"}),  # creates a junction at 4 (deg>2)
        (5, 9, {"attr":"I"}),   
        (9, 10, {"attr":"J"}),
        (10, 11, {"attr":"K"}),  # another branch
    ]
    G.add_edges_from(edges)
    sorted = sort_edges_by_max_intersections(G)
    print(sorted)
    assert sorted == [(1, 2, 0), (2, 6, 0), (6, 7, 0), (7, 8, 0), (8, 4, 0), (4, 5, 0), (5, 9, 0), (9, 10, 0), (10, 11, 0), (2, 3, 0), (3, 4, 0)]
    assert G.get_edge_data(1, 2, 0)["attr"] == "A"
    assert G.get_edge_data(2, 6, 0)["attr"] == "E"
    assert G.get_edge_data(6, 7, 0)["attr"] == "F"
    assert G.get_edge_data(7, 8, 0)["attr"] == "G"
    assert G.get_edge_data(8, 4, 0)["attr"] == "H"
    assert G.get_edge_data(4, 5, 0)["attr"] == "D"
    assert G.get_edge_data(5, 9, 0)["attr"] == "I"
    assert G.get_edge_data(9, 10, 0)["attr"] == "J"
    assert G.get_edge_data(10, 11, 0)["attr"] == "K"
    assert G.get_edge_data(2, 3, 0)["attr"] == "B"
    assert G.get_edge_data(3, 4, 0)["attr"] == "C"
