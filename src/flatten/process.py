from datetime import datetime
from typing import Union
from functools import reduce
from itertools import groupby

import pandas as pd
from shapely import LineString, LinearRing, Point
import geopandas as gpd
import shapely
import networkx as nx

from flatten.get_data import get_graph
from flatten.optimize import optimize
from flatten.orient_triangle_graph import get_oriented_graph
from flatten.triangle_graph import remove_interstitial_nodes, reverse
from flatten.wfs import get_hydro_data

import logging
logger = logging.getLogger(__name__)

def get_profiles(
    gdf_segments: gpd.GeoDataFrame, line: LineString, direct: bool = True
) -> Union[
    tuple[
        list[tuple[tuple[float, float], int | None]],
        list[tuple[tuple[float, float], int | None]],
    ],
    None,
]:
    """
    Computes the left and right profiles for the input line using the given triangle segments.
    """
    intersecting_segments = gdf_segments.loc[gdf_segments.intersects(line)]
    intersecting_segments_geometry = intersecting_segments["geometry"]
    intersecting_segments_type = intersecting_segments["type"]
    intersections = [
        line.intersection(segment) for segment in intersecting_segments_geometry
    ]
    intersection_zipped = zip(
        intersections,
        intersecting_segments_geometry,
        intersecting_segments_type,
    )

    def is_not_on_one_end(
        point: Point, segment: LineString, _: str
    ) -> bool:
        """
        True in the intersection is a point and it is not one of the endpoints of the segment.
        """
        return point.geom_type == "Point" and point.distance(segment.boundary) > 0

    intersection_zipped = [
        intersection
        for intersection in intersection_zipped
        if is_not_on_one_end(*intersection)
    ]
    # TODO handle MultiPoints?
    if not intersection_zipped:
        return None
    distances = [shapely.line_locate_point(line, p[0]) for p in intersection_zipped]
    # unzip
    (
        intersections,
        intersecting_segments_geometry,
        intersecting_segments_type,
    ) = zip(*intersection_zipped)
    # zip again with distance
    zipped: tuple[list[LineString], list[str], list[Point], list[float]] = (
        zip(
            intersecting_segments_geometry,
            intersecting_segments_type,
            intersections,
            distances,
        )
    )  # type: ignore
    # sort by distance
    sorted_intersections = sorted(
        zipped, key=lambda intersection: intersection[3], reverse=not direct
    )

    def two_d_point(p) -> tuple[float,float]:
        return tuple(shapely.get_coordinates(p,include_z=False).tolist()[0])
    point: Point = Point(line.coords[0] if direct else line.coords[-1])
    def update(x, y: tuple[LineString, str, Point, float]):
        # the accumulator x: (the previous point, the index of the current bottleneck, left accumulator, right accumulator)
        previous, index, left, right = x
        segment, type, intersection, _ = y
        is_bottleneck = type == "bottleneck"
        edge_index = index if is_bottleneck else None
        # logger.debug(f"x={x}")
        # logger.debug(f"y={y}")
        # logger.debug(f"is_bottleneck={is_bottleneck}")
        # logger.debug(f"edge_index={edge_index}")
        # logger.debug(f"ring={[two_d_point(previous), two_d_point(intersection), two_d_point(segment)]}")
        if LinearRing([two_d_point(previous), two_d_point(intersection), two_d_point(segment)]).is_ccw:
            left.append((segment.coords[0], edge_index))
            right.append((segment.coords[1], edge_index))
        else:
            left.append((segment.coords[1], edge_index))
            right.append((segment.coords[0], edge_index))
        return (intersection, index + 1 if is_bottleneck else index, left, right)

    _, _, left, right = reduce(update, sorted_intersections, (point, 0, [], []))  # type: ignore
    return left, right
def merge_profile_points(
    points: list[tuple[tuple[float, float], int | None]],
) -> list[tuple[tuple[float, float], list[int]]]:
    # removing consecutive identical points
    def get_b(list: list[tuple[tuple[float, float], int | None]]) -> list[int]:
        """
        Merge bottleneck information. e[2] is a bottleck index (int) or None.
        """
        return [e[1] for e in list if e[1] is not None]

    def build(
        point: tuple[float, float],
        list: list[tuple[tuple[float, float], int | None]],
    ):
        return (point, get_b(list))

    return [build(key, list(g)) for key, g in groupby(points, key=lambda l: l[0])]

def main(srs: str, in_box: tuple[float, float, float, float], max_segment_length: float, output_file: str | None) -> gpd.GeoDataFrame | None:
    logger.info(f"{datetime.now()} - start")
    box = (in_box[0], in_box[1], in_box[2], in_box[3], srs)
    r = get_hydro_data(box, srs)
    if not r:
        logger.error("no data")
        return None
    (surfaces, segments, _) = r
    # triangles = get_triangles(surfaces, max_segment_length)
    gdf_triangle, gdf_triangle_segment, gdf_split = get_graph(surfaces, max_segment_length)

    graph, edge_gdf, has_no_cycle = get_oriented_graph(gdf_triangle, segments)

    for cycle in list(nx.simple_cycles(graph)):
        logger.debug(f"Cycle: {cycle}")
    if output_file is not None:
        edge_gdf.to_file(output_file, layer="edges")
    assert(has_no_cycle) # make sure there is no cycle

    # simplify graph by removing simple nodes (1 incoming edge, 1 outgoing edge)
    graph = remove_interstitial_nodes(graph)
    line_graph = nx.line_graph(graph)
    sorted_edges = list(nx.lexicographical_topological_sort(line_graph, key=lambda x: line_graph.out_degree(x)))
    graph = reverse(graph)
    sorted_edges.reverse()
    sorted_edges = [(v, u, k) for (u, v, k) in sorted_edges]
    # reverse and simplify graph by removing simple nodes (1 incoming edge, 1 outgoing edge)
    # graph = remove_interstitial_nodes(reverse(graph))
    # sorted_edges = list(nx.topological_sort(nx.line_graph(graph)))
    # edges = sort_edges_by_max_intersections(graph)

    segment_order = []
    segment_id = []
    point_order = []
    point_position = []
    point_geom = []
    point_bottlenecks = []
    edge_profiles = dict()

    for edge_order, edge in enumerate(sorted_edges):
        attributes = graph.get_edge_data(*edge)
        segment_key = attributes["edge_ids"] or []
        segment_geometry = attributes["geometry"]
        profiles = get_profiles(
            gdf_triangle_segment,
            line=segment_geometry,
        )
        if profiles:
            left, right = profiles
            edge_profiles[edge] = (left, right)
    
    print(len(graph.edges), "edges")
    # remove edges without any profile
    to_remove = []
    for graph_edge in graph.edges:
        if graph_edge not in edge_profiles:
            to_remove.append(graph_edge)
    for graph_edge in to_remove:
        graph.remove_edge(*graph_edge)
    print(len(graph.edges), "edges after cleanup")
    for graph_node in graph.nodes:
        in_edges = list(graph.in_edges(graph_node, keys=True))
        out_edges = list(graph.out_edges(graph_node, keys=True))
        if (len(in_edges) == 1) & (len(out_edges) == 1):
            # FIXME should allow for more complex situations but it creates unsolvable optimisations
            # FIXME we should switch to a global optimisation!
            logger.debug(f"graph_node {graph_node} with {len(in_edges)} and {len(out_edges)}")
            # make sur the last points of in_edge belong to out_edge
            in_edge = in_edges[0]
            if in_edge in edge_profiles:
                in_left, in_right = edge_profiles[in_edge]
                last_left = in_left[-1]
                last_right = in_right[-1]
                logger.debug(f"last_left={last_left} last_right={last_right}")
                # out_edge = out_edges[0]
                for out_edge in out_edges:
                    if out_edge in edge_profiles:
                        out_left, out_right = edge_profiles[out_edge]
                        modif = False
                        logger.debug(f"{out_edge} with {out_left[0][0]} and {out_right[0][0]} and {len(out_left)} {len(out_right)}")
                        if last_left[0] != out_left[0][0]:
                            out_left.insert(
                                0, (last_left[0], None)
                            )  # we don't keep bottleneck info
                            modif = True
                        if last_right[0] != out_right[0][0]:
                            out_right.insert(
                                0, (last_right[0], None)
                            )  # we don't keep bottleneck info
                            modif = True
                        if modif:
                            logger.debug(f"modif {len(out_left)} {len(out_right)}")
                            edge_profiles[out_edge] = out_left, out_right

    for edge_order, (edge, (left, right)) in enumerate(edge_profiles.items()):
        attributes = graph.get_edge_data(*edge)
        segment_key = attributes["edge_ids"] or []
        segment_geometry = attributes["geometry"]
        merged_left = merge_profile_points(left)
        merged_right = merge_profile_points(right)

        def add_points(profile, position):
            for i, p in enumerate(profile):
                segment_order.append(edge_order)
                segment_id.append(segment_key)
                point_order.append(i)
                point_position.append(position)
                point_geom.append(Point(p[0]))
                point_bottlenecks.append(p[1])

        add_points(merged_left, "left")
        add_points(merged_right, "right")

    gdf_points = gpd.GeoDataFrame(
        {
            "point_id": list(range(0, len(point_geom))),
            "segment_id": segment_id,
            "segment_order": segment_order,
            "point_order": point_order,
            "point_position": point_position,
            "point_bottlenecks": point_bottlenecks,
        },
        geometry=point_geom,
        crs=surfaces.crs,
    )
    if output_file is not None:
        gdf_points.to_file(output_file, layer="points", driver="GPKG")

    edge_ids = []
    edge_geometries = []
    edge_left_profiles = []
    edge_right_profiles = []
    for edge_order, edge in enumerate(sorted_edges):
        attributes = graph.get_edge_data(*edge)
        edge_ids.append(attributes["edge_ids"] or [])
        edge_geometries.append(attributes["geometry"])
        (left, right) = edge_profiles[edge]
        edge_left_profiles.append(len(left))
        edge_right_profiles.append(len(right))

    gdf_edges_final = gpd.GeoDataFrame(
        {
            "edge_id": list(range(0, len(sorted_edges))),
            "edge_ids": edge_ids,
            "left_profiles": edge_left_profiles,
            "right_profiles": edge_right_profiles,
        },
        geometry=edge_geometries,
        crs=surfaces.crs,
    )
    if output_file is not None:
        gdf_edges_final.to_file(output_file, layer="edges_final", driver="GPKG")
        surfaces.to_file(output_file, layer="surface")
        segments.to_file(output_file, layer="hydro_segment")
        gdf_triangle.to_file(output_file, layer="triangle", driver="GPKG")
        gdf_split.to_file(output_file, layer="triangle_split", driver="GPKG")
        gdf_triangle_segment.to_file(
            output_file, layer="triangle_segment", driver="GPKG"
        )
    logger.info(f"{datetime.now()} - all done!")
    return gdf_points

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    logger.addHandler(logging.StreamHandler())
    srs = "urn:ogc:def:crs:EPSG::2154"
    box = (1036535, 6289927, 1042268, 6305786)
    output_file = "triangle_graph.gpkg"
    max_segment_length = 20.0
    res = main(srs, box, max_segment_length, output_file)
    if res is not None:
        points, temp = optimize(res)
        points.to_file(output_file, layer="points_optimised")
