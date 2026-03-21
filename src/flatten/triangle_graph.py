from typing import Optional

import geopandas as gpd
import numpy
import pandas as pd
from pyproj import Transformer
import shapely
from shapely import LineString
import networkx as nx
from pyinterpolate import inverse_distance_weighting

from flatten.elevation import throttle_requests
from flatten.split import get_segments
from flatten.utils import get_bottlenecks, split_triangles_with_bottlenecks


def get_triangles_with_height(
    union, crs, sample_points, elevations
) -> gpd.GeoDataFrame:
    triangles = shapely.constrained_delaunay_triangles(union)

    def triangle_height(geom: shapely.Polygon) -> float:
        # keep only 3 points and select the z value
        z_values = shapely.get_coordinates(geom, include_z=True)[:3, 2]
        z_values = z_values[~numpy.isnan(z_values)]
        return z_values.mean()

    # use crs from input file
    triangle_list = [p for p in triangles.geoms]
    triangle_heights = [triangle_height(p) for p in triangles.geoms]  # type: ignore
    triangle_centroids = [[p.centroid.x, p.centroid.y] for p in triangles.geoms]
    triangle_elevations = []
    for c in triangle_centroids:
        triangle_elevations.append(
            inverse_distance_weighting(
                unknown_location=c,
                known_values=elevations,
                known_geometries=sample_points,
                power=2.0,
            )
        )
    gdf_triangle = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "triangle_id": list(range(0, len(triangle_list))),
                "triangle_height": triangle_heights,
                "triangle_elevation_rge": triangle_elevations,
            }
        ),
        geometry=triangle_list,
        crs=crs,
    )
    return gdf_triangle


def get_graph(
    surfaces: gpd.GeoDataFrame, max_segment_length: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # union = shapely.union_all(surfaces.geometry).simplify(0.1, preserve_topology=True)
    union = shapely.union_all(surfaces.geometry).segmentize(max_segment_length)
    # sample points
    number_of_points = int(
        union.area / 10000
    )  # computed according to the surface of union (1 point per hectare)
    sample_multipoint: shapely.MultiPoint = uniform(union, number_of_points, 42)  # type: ignore
    sample_points = [(p.x, p.y) for p in sample_multipoint.geoms]

    transformer = Transformer.from_crs(surfaces.crs, 4326)
    elevations = throttle_requests(list(transformer.itransform(sample_points)))
    assert len(elevations) == len(sample_points)

    gdf_triangle = get_triangles_with_height(
        union, surfaces.crs, sample_points, elevations
    )
    # triangle_graph: gpd.GeoDataFrame = get_triangle_graph(gdf_triangle)
    # triangle_graph = neatnet.remove_interstitial_nodes(triangle_graph) # type: ignore
    # print("triangle_graph",len(triangle_graph))
    # oriented_triangle_graph = orient_triangle_graph(triangle_graph, gdf_triangle)

    triangle_segment = list(set(get_segments(gdf_triangle.geometry)))
    # determine if segments belong to the boundary: if they do, they must be constraints
    triangle_segment_constraint = list(
        map(lambda s: union.boundary.contains(s), triangle_segment)
    )
    gdf_triangle_segment = gpd.GeoDataFrame(
        {
            "triangle_segment_id": list(range(0, len(triangle_segment))),
            "constraint": triangle_segment_constraint,
        },
        geometry=triangle_segment,
        crs=surfaces.crs,
    )
    # merge with bottlenecks
    gdf_triangle_segment["type"] = gdf_triangle_segment["constraint"].apply(
        lambda b: "constrained" if b else "unconstrained"
    )
    gdf_triangle_segment = gdf_triangle_segment.rename(
        columns={"triangle_segment_id": "tmp_id"}
    )
    gdf_bottlenecks = get_bottlenecks(gdf_triangle, gdf_triangle_segment)

    gdf_split = split_triangles_with_bottlenecks(gdf_triangle, gdf_bottlenecks)

    def add_height(row) -> Optional[LineString]:
        line = row["geometry"]
        if not line:
            return None
        coordinates: list[list[float]] = shapely.get_coordinates(
            line, include_z=False
        ).tolist()
        height0 = inverse_distance_weighting(
            unknown_location=coordinates[0],
            known_values=elevations,
            known_geometries=sample_points,
            power=2.0,
        )
        height1 = inverse_distance_weighting(
            unknown_location=coordinates[-1],
            known_values=elevations,
            known_geometries=sample_points,
            power=2.0,
        )
        return LineString([(*coordinates[0], height0), (*coordinates[-1], height1)])

    # gdf_bottlenecks["geometry"] = gdf_bottlenecks.apply(add_height, axis=1)
    gdf_bottlenecks["constraint"] = False
    gdf_bottlenecks["type"] = "bottleneck"
    gdf_bottlenecks = gdf_bottlenecks.rename(columns={"triangle_id": "tmp_id"})
    gdf_bottlenecks = gdf_bottlenecks.set_crs(
        gdf_triangle_segment.crs,  # type: ignore
        allow_override=True,
    )  # type: ignore

    unified: gpd.GeoDataFrame = pd.concat(
        [gdf_triangle_segment, gdf_bottlenecks], ignore_index=True, sort=False
    )  # type: ignore
    # Cast the constraint column to a proper boolean dtype (in case concat made it object)
    unified["constraint"] = unified["constraint"].astype(bool)
    # Re‑order columns for readability (optional)
    unified.insert(0, "triangle_segment_id", range(len(unified)))  # 0‑based integer IDs
    unified = unified[["triangle_segment_id", "constraint", "type", "geometry"]]
    # recompute height for all segments
    unified["geometry"] = unified.apply(add_height, axis=1)
    unified = unified.set_crs(gdf_triangle_segment.crs, allow_override=True)  # type: ignore
    return gdf_triangle, unified, gdf_split  # type: ignore #, oriented_triangle_graph#gdf_edges

def get_triangle_graph_as_nx(gdf_triangle: gpd.GeoDataFrame) -> nx.Graph:
    touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    graph = nx.Graph()
    for triangle_id_left, triangle_id_right in zip(
        touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"]
    ):
        if triangle_id_left < triangle_id_right:
            geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"]  # type: ignore
            geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"]  # type: ignore
            if geom_left.intersection(geom_right).length > 0:
                graph.add_edge(
                    triangle_id_left,
                    triangle_id_right,
                    geometry=LineString([geom_left.centroid, geom_right.centroid]),
                )
    return graph


# def get_triangle_graph_as_gdf(gdf_triangle: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
#     edges = []
#     for triangle_id_left, triangle_id_right in zip(
#         touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"]
#     ):
#         if triangle_id_left < triangle_id_right:
#             geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"]  # type: ignore
#             geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"]  # type: ignore
#             if geom_left.intersection(geom_right).length > 0:
#                 edges.append(LineString([geom_left.centroid, geom_right.centroid]))
#     gdf_edges = gpd.GeoDataFrame(
#         {"edge_id": list(range(0, len(edges)))}, geometry=edges, crs=gdf_triangle.crs
#     )
#     print("gdf_edges", len(gdf_edges))
#     return gdf_edges  # type: ignore


def reverse(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """reverses the directed graph."""
    directed_graph = nx.MultiDiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        line: LineString = data["geometry"]
        edge_ids = data.get("edge_ids", [])
        directed_graph.add_edge(
            v, u, key=key, geometry=line.reverse(), edge_ids=edge_ids
        )
    return directed_graph


def remove_interstitial_nodes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Removes nodes of in degree 1 and out_degree 1 from the directed graph."""
    directed_graph = graph.copy()
    nodes_to_remove = []
    for node in directed_graph.nodes:
        if (directed_graph.in_degree(node) == 1) & (  # type: ignore
            directed_graph.out_degree(node) == 1  # type: ignore
        ):
            (pred_node, _, in_key, pred_data) = list(
                directed_graph.in_edges(node, keys=True, data=True)  # type: ignore
            )[0]
            (_, succ_node, out_key, succ_data) = list(
                directed_graph.out_edges(node, keys=True, data=True)  # type: ignore
            )[0]
            geom = shapely.line_merge(
                shapely.GeometryCollection(
                    [pred_data["geometry"], succ_data["geometry"]]
                )
            )
            ids = sorted(list(set(pred_data["edge_ids"] + succ_data["edge_ids"])))
            directed_graph.remove_edges_from(
                [(pred_node, node, in_key), (node, succ_node, out_key)]
            )
            nodes_to_remove.append(node)
            directed_graph.add_edge(pred_node, succ_node, geometry=geom, edge_ids=ids)
    directed_graph.remove_nodes_from(nodes_to_remove)
    return directed_graph  # type: ignore
