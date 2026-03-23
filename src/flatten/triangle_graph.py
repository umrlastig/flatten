from typing import Callable, Optional

import geopandas as gpd
import numpy
import pandas as pd
from pyproj import CRS
import shapely
from shapely import Geometry, LineString, Polygon
import networkx as nx

from flatten.split import get_segments
from flatten.utils import get_bottlenecks, split_triangles_with_bottlenecks


def get_triangles_with_height(
    union: Geometry, crs: CRS|None, get_elevation: Callable[[list[float]],float]
) -> gpd.GeoDataFrame:
    triangles = shapely.constrained_delaunay_triangles(union)

    def triangle_height(geom: Polygon) -> float:
        # keep only 3 points and select the z value
        z_values = shapely.get_coordinates(geom, include_z=True)[:3, 2]
        z_values = z_values[~numpy.isnan(z_values)]
        return z_values.mean()

    # use crs from input file
    triangle_list = [p for p in triangles.geoms]
    triangle_heights = [triangle_height(p) for p in triangles.geoms]  # type: ignore
    triangle_centroids = [[p.centroid.x, p.centroid.y] for p in triangles.geoms]
    triangle_elevations = [get_elevation(c) for c in triangle_centroids]
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


def get_triangles_and_segments(
    union: Geometry, get_elevation: Callable[[list[float]],float], crs: CRS|None, max_segment_length: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build the triangles and triangle segments from the input hydro surfaces.    
    """
    gdf_triangle = get_triangles_with_height(union, crs, get_elevation)
    # triangle_graph: gpd.GeoDataFrame = get_triangle_graph(gdf_triangle)
    # triangle_graph = neatnet.remove_interstitial_nodes(triangle_graph) # type: ignore
    # print("triangle_graph",len(triangle_graph))
    # oriented_triangle_graph = orient_triangle_graph(triangle_graph, gdf_triangle)

    triangle_segment = list(set(get_segments(gdf_triangle.geometry)))
    # determine if segments belong to the boundary: if they do, they must be constraints
    triangle_segment_constraint = list(
        map(lambda s: union.boundary.contains(s), triangle_segment) # type: ignore
    )
    gdf_triangle_segment = gpd.GeoDataFrame(
        {
            "triangle_segment_id": list(range(0, len(triangle_segment))),
            "constraint": triangle_segment_constraint,
        },
        geometry=triangle_segment,
        crs=crs,
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
        height0 = get_elevation(coordinates[0])
        height1 = get_elevation(coordinates[-1])
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

def get_triangle_graph_as_nx(gdf_triangle: gpd.GeoDataFrame, id_column: str = "triangle_id") -> nx.Graph:
    touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    graph = nx.Graph()
    for triangle_id_left, triangle_id_right in zip(
        touching_triangles[f"{id_column}_left"], touching_triangles[f"{id_column}_right"]
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
