from typing import Callable
import geopandas as gpd
import numpy as np
from shapely import Point

def write_obj(vertices: gpd.GeoSeries, triangles: gpd.GeoSeries, rescale:Callable[[Point], Point], output: str):
    obj_file = open(output, 'w')
    # create vertices
    for _, vertex in vertices.items():
        rescaled_vertex = rescale(vertex) # type: ignore
        obj_file.write("v {0} {1} {2}\n".format(rescaled_vertex.x, rescaled_vertex.y, rescaled_vertex.z))# type: ignore
    # create faces (triangles)
    for triangle in triangles:
        coords = triangle.exterior.coords # type: ignore
        triangle_indices = []
        for coord in coords:
            tol = 1e-9
            mask = (np.abs(vertices.x - coord[0]) < tol) & (np.abs(vertices.y - coord[1]) < tol)
            matches = vertices[mask]
            if any(matches):
                triangle_indices.append(matches.index[0]+1)#index start at 1 for obj
        if len(triangle_indices) == 4: # if one of the points was not found (4 since the first is repeated as last)
            obj_file.write("f {0} {1} {2}\n".format(triangle_indices[0], triangle_indices[1], triangle_indices[2]))
    obj_file.close()

def export(point_geom: gpd.GeoSeries, triangles: gpd.GeoSeries, output: str, output_range_x: float|None = None, z_rescale: float = 1.0, rescale: bool = True):
    min_x, min_y, max_x, max_y = triangles.total_bounds
    # min_x = triangles.x.min()
    # max_x = triangles.x.max()
    # min_y = triangles.y.min()
    # max_y = triangles.y.max()
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_out_x = output_range_x if output_range_x is not None else range_x
    range_out_y = range_y * range_out_x / range_x
    # rescale function
    def rescale_point(point: Point) -> Point:
        return Point(
            (point.x - min_x) * range_out_x / range_x - range_out_x / 2.0, 
            (point.y - min_y) * range_out_y / range_y - range_out_y / 2.0, 
            point.z * z_rescale
        )
    write_obj(point_geom, triangles, rescale_point if rescale else lambda p: p, output) # type: ignore
    # thefile = open(output, 'w')
    # for _, vertex in point_geom.items():
    #     point: Point = vertex # type: ignore
    #     thefile.write("v {0} {1} {2}\n".format(
    #         (point.x - min_x) * range_out_x / range_x - range_out_x / 2.0, 
    #         (point.y - min_y) * range_out_y / range_y - range_out_y / 2.0, 
    #         point.z
    #     ))
    # create faces (triangles)
    # for triangle in triangles["geometry"]:
    #     coords = triangle.exterior.coords
    #     triangle_indices = []
    #     for coord in coords:
    #         tol = 1e-9
    #         mask = (np.abs(point_geom.x - coord[0]) < tol) & (np.abs(point_geom.y - coord[1]) < tol)
    #         matches = point_geom[mask]
    #         if any(matches):
    #             triangle_indices.append(matches.index[0]+1)#index start at 1 for obj
    #     if len(triangle_indices) == 4: # if one of the points was not found (4 since the first is repeated as last)
    #         thefile.write("f {0} {1} {2}\n".format(triangle_indices[0], triangle_indices[1], triangle_indices[2]))
    # thefile.close()

input = "temp.gpkg"
triangle_layer = "triangle"
output_range_x = 10000.0
z_rescale = 10.0
triangles = gpd.read_file(input, layer=triangle_layer)

unique_points: gpd.GeoSeries = triangles.geometry.extract_unique_points()
unique_points: gpd.GeoSeries = unique_points.explode(ignore_index=True)
unique_points: gpd.GeoSeries = unique_points.drop_duplicates().reset_index(drop=True)# type: ignore
print("original")
export(unique_points, triangles.geometry, f'original.obj', output_range_x, z_rescale)
point_layer = "points"
points = gpd.read_file(input, layer=point_layer)
point_geom: gpd.GeoSeries = points.geometry.drop_duplicates().reset_index(drop=True)# type: ignore
print(point_layer)
export(point_geom, triangles.geometry, f'{point_layer}.obj', output_range_x, z_rescale)
point_layer = "points_optimised"
points = gpd.read_file(input, layer=point_layer)
point_geom: gpd.GeoSeries = points.geometry.drop_duplicates().reset_index(drop=True)# type: ignore
print(point_layer)
export(point_geom, triangles.geometry, f'{point_layer}.obj', output_range_x, z_rescale)
point_layer = f"{point_layer}_no_rescaling"
print(point_layer)
export(point_geom, triangles.geometry, f'{point_layer}.obj', rescale=False)
print("All done!")