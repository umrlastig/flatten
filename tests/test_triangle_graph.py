import pytest
from shapely import Polygon, LineString
import geopandas as gpd
import networkx as nx

from flatten.triangle_graph import remove_interstitial_nodes, reverse

def test_reverse():
    graph = nx.MultiDiGraph()
    graph.add_edge(0,1, geometry=LineString([[0,0],[1,0]]), edge_ids=["A"])
    graph.add_edge(1,2, geometry=LineString([[1,0],[2,0]]), edge_ids=["B"])
    graph.add_edge(2,3, geometry=LineString([[2,0],[3,0]]), edge_ids=["C"])
    graph.add_edge(3,4, geometry=LineString([[3,0],[4,0]]), edge_ids=["D"])
    graph.add_edge(4,5, geometry=LineString([[4,0],[5,0]]), edge_ids=["E"])
    graph.add_edge(2,6, geometry=LineString([[2,0],[3,1]]), edge_ids=["F"])
    graph.add_edge(6,7, geometry=LineString([[3,1],[4,1]]), edge_ids=["G"])
    graph.add_edge(7,4, geometry=LineString([[4,1],[4,0]]), edge_ids=["H"])
    reversed = reverse(graph)
    print(reversed)
    for edge in reversed.edges:
        attributes = reversed.get_edge_data(*edge)
        print(edge, attributes)
    assert reversed.has_edge(1,0)
    assert reversed.has_edge(2,1)
    assert reversed.has_edge(3,2)
    assert reversed.has_edge(4,3)
    assert reversed.has_edge(5,4)
    assert reversed.has_edge(6,2)
    assert reversed.has_edge(7,6)
    assert reversed.has_edge(4,7)

def test_remove_interstitial_nodes():
    graph = nx.MultiDiGraph()
    graph.add_edge(0,1, geometry=LineString([[0,0],[1,0]]), edge_ids=["A"])
    graph.add_edge(1,2, geometry=LineString([[1,0],[2,0]]), edge_ids=["B"])
    graph.add_edge(2,3, geometry=LineString([[2,0],[3,0]]), edge_ids=["C"])
    graph.add_edge(3,4, geometry=LineString([[3,0],[4,0]]), edge_ids=["D"])
    graph.add_edge(4,5, geometry=LineString([[4,0],[5,0]]), edge_ids=["E"])
    graph.add_edge(2,6, geometry=LineString([[2,0],[3,1]]), edge_ids=["F"])
    graph.add_edge(6,7, geometry=LineString([[3,1],[4,1]]), edge_ids=["G"])
    graph.add_edge(7,4, geometry=LineString([[4,1],[4,0]]), edge_ids=["H"])
    simplified = remove_interstitial_nodes(graph)
    print(simplified)
    assert simplified.number_of_edges() == 4
    assert simplified.number_of_nodes() == 4
    for (u,v,_,d) in simplified.edges(keys=True, data=True):
        assert (u,v,d) in [
            (0,2,{"geometry":LineString([[0, 0], [1, 0], [2, 0]]),"edge_ids":["A", "B"]}),
            (2,4,{"geometry":LineString([[2, 0], [3, 0], [4, 0]]),"edge_ids":["C", "D"]}),
            (2,4,{"geometry":LineString([[2, 0], [3, 1], [4, 1], [4, 0]]),"edge_ids":["F", "G", "H"]}),
            (4,5,{"geometry":LineString([[4, 0], [5, 0]]),"edge_ids":["E"]})
        ]
