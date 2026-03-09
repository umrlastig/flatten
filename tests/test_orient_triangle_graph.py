import pytest
from shapely import Polygon, LineString
from flatten.orient_triangle_graph import split_cycle
import geopandas as gpd

def test_split_cycle():
    cycle = ['a', 'b', 'c', 'd', 'e']
    split_nodes = set(['b', 'd'])
    split = split_cycle(cycle, split_nodes)
    print(split)
    assert len(split) == 2
    assert split == [['b', 'c', 'd'], ['d', 'e', 'a', 'b']]

