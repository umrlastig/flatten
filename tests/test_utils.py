import pytest
from shapely import Polygon, LineString
from flatten.utils import find_projection, get_bottlenecks
import geopandas as gpd

def test_projection():
    triangle = Polygon(((0., 0.), (0., 2.), (1., 2.), (0., 0.)))
    segment = LineString(((0., 0.), (0., 2.)))
    line = find_projection(triangle, segment)
    print(line)
    assert not line
    triangle = Polygon(((0., 0.), (0., 2.), (2., 1.), (0., 0.)))
    line = find_projection(triangle, segment)
    print(line)
    assert line
    assert line == LineString(((0., 1.), (2., 1.)))

def test_get_bottlenecks():
    t1 = Polygon(((0., 0.), (0., 2.), (2., 1.), (0., 0.)))
    t2 = Polygon(((0., 0.), (2., 1.), (2., 0.), (0., 0.)))
    l1 = LineString(((0., 0.), (0., 2.)))
    b1 = LineString(((0, 1), (2, 1)))
    triangles = gpd.GeoDataFrame({'triangle_id':[0, 1], 'geometry':[t1, t2]}, geometry='geometry', crs="EPSG:4326")
    segments = gpd.GeoDataFrame({'segment_id':[0], 'constraint':[True], 'geometry':[l1]}, geometry='geometry', crs="EPSG:4326")
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 1
    assert bottlenecks.at[0,"geometry"] == b1
    l2 = LineString(((2., 0.), (2., 1.)))
    segments = gpd.GeoDataFrame({'segment_id':[0, 1], 'constraint':[True,True], 'geometry':[l1, l2]}, geometry='geometry', crs="EPSG:4326")
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 1
    t3 = Polygon(((0., 2.), (2., 3.), (2., 1.), (0., 2.)))
    l3 = LineString(((2., 1.), (2., 3.)))
    b2 = LineString(((2, 2), (0, 2)))
    triangles = gpd.GeoDataFrame({'triangle_id':[0, 1, 2], 'geometry':[t1, t2, t3]}, geometry='geometry', crs="EPSG:4326")
    segments = gpd.GeoDataFrame({'segment_id':[0, 1, 2], 'constraint':[True,True,True], 'geometry':[l1, l2, l3]}, geometry='geometry', crs="EPSG:4326")
    bottlenecks = get_bottlenecks(triangles, segments, "triangle_id")
    print(bottlenecks)
    assert len(bottlenecks) == 2
    assert bottlenecks.at[0,"geometry"] == b1
    assert bottlenecks.at[2,"geometry"] == b2
