import pytest
from shapely import Point, Polygon, LineString
import geopandas as gpd
import networkx as nx

from flatten.optimize import optimize_all_segments

def test_unique():
    # Create sample points
    points_data = [
        # segment 1
        {'point_id': 1, 'segment_order': 1, 'point_position': "left", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((-1, 0, 0))},
        {'point_id': 3, 'segment_order': 1, 'point_position': "left", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((-1, 1, 0))},
        {'point_id': 2, 'segment_order': 1, 'point_position': "right", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((1, 0, 0))},
        {'point_id': 4, 'segment_order': 1, 'point_position': "right", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((1, 1, 0))},
        # segment 2
        {'point_id': 3, 'segment_order': 2, 'point_position': "left", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((-1, 1, 0))},
        {'point_id': 6, 'segment_order': 2, 'point_position': "left", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((-2, 3, 2))},
        {'point_id': 11, 'segment_order': 2, 'point_position': "left", 'point_order':2, 'point_bottlenecks': [2], 'geometry': Point((-1, 5, 4))},
        {'point_id': 4, 'segment_order': 2, 'point_position': "right", 'point_order':0, 'point_bottlenecks': [], 'geometry': Point((1, 1, 0))},
        {'point_id': 5, 'segment_order': 2, 'point_position': "right", 'point_order':1, 'point_bottlenecks': [0], 'geometry': Point((0, 2, 1))},
        {'point_id': 7, 'segment_order': 2, 'point_position': "right", 'point_order':2, 'point_bottlenecks': [1], 'geometry': Point((-1, 3, 2))},
        {'point_id': 10, 'segment_order': 2, 'point_position': "right", 'point_order':3, 'point_bottlenecks': [2], 'geometry': Point((0, 4, 3))},
        {'point_id': 12, 'segment_order': 2, 'point_position': "right", 'point_order':4, 'point_bottlenecks': [], 'geometry': Point((1, 5, 5))},
        # segment 3
        {'point_id': 3, 'segment_order': 3, 'point_position': "left", 'point_order':0, 'point_bottlenecks': [], 'geometry': Point((-1, 1, 0))},
        {'point_id': 5, 'segment_order': 3, 'point_position': "left", 'point_order':1, 'point_bottlenecks': [0], 'geometry': Point((0, 2, 1))},
        {'point_id': 8, 'segment_order': 3, 'point_position': "left", 'point_order':2, 'point_bottlenecks': [1], 'geometry': Point((1, 3, 2))},
        {'point_id': 10, 'segment_order': 3, 'point_position': "left", 'point_order':3, 'point_bottlenecks': [2], 'geometry': Point((0, 4, 3))},
        {'point_id': 11, 'segment_order': 3, 'point_position': "left", 'point_order':4, 'point_bottlenecks': [], 'geometry': Point((-1, 5, 4))},
        {'point_id': 4, 'segment_order': 3, 'point_position': "right", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((1, 1, 0))},
        {'point_id': 9, 'segment_order': 3, 'point_position': "right", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((2, 3, 2))},
        {'point_id': 12, 'segment_order': 3, 'point_position': "right", 'point_order':2, 'point_bottlenecks': [2], 'geometry': Point((1, 5, 5))},
        # segment 4
        {'point_id': 11, 'segment_order': 4, 'point_position': "left", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((-1, 5, 4))},
        {'point_id': 13, 'segment_order': 4, 'point_position': "left", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((-1, 6, 0))},
        {'point_id': 12, 'segment_order': 4, 'point_position': "right", 'point_order':0, 'point_bottlenecks': [0], 'geometry': Point((1, 5, 5))},
        {'point_id': 14, 'segment_order': 4, 'point_position': "right", 'point_order':1, 'point_bottlenecks': [1], 'geometry': Point((1, 6, 0))},
    ]
    points = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
    test_file = "test.gpkg"
    points.to_file(test_file, layer='init')

    points['unique_id'] = points.groupby('geometry', sort=False).ngroup()
    unique_points = points.drop_duplicates(subset=['geometry'], keep='first')
    assert list(unique_points['unique_id']) == list(range(0, len(unique_points)))
    assert len(unique_points) == 14
    print(points)
    print(unique_points)
    res = optimize_all_segments(points, alpha=1, beta=10, gamma=0.1)
    assert res is not None
    if res is not None:
        points_final = res
        print(points_final)
        points_final.to_file(test_file, layer='final')
