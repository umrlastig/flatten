from datetime import datetime
from typing import Callable, Union
from functools import reduce
from itertools import groupby

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import Geometry, LineString, LinearRing, Point, Polygon
import geopandas as gpd
from geopandas.tools._random import uniform
import shapely
import networkx as nx

from flatten.elevation import interpolate, throttle_requests
from flatten.optimize import optimize_all_segments
from flatten.orient_triangle_graph import get_oriented_graph
from flatten.triangle_graph import get_triangles_and_segments, remove_interstitial_nodes, reverse
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

    def is_not_on_one_end(point: Point, segment: LineString, _: str) -> bool:
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
    zipped: tuple[list[LineString], list[str], list[Point], list[float]] = zip(
        intersecting_segments_geometry,
        intersecting_segments_type,
        intersections,
        distances,
    )  # type: ignore
    # sort by distance
    sorted_intersections = sorted(
        zipped, key=lambda intersection: intersection[3], reverse=not direct
    )

    def two_d_point(p) -> tuple[float, float]:
        return tuple(shapely.get_coordinates(p, include_z=False).tolist()[0])

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
        if LinearRing(
            [two_d_point(previous), two_d_point(intersection), two_d_point(segment)]
        ).is_ccw:
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

def build_triangles_and_points(
    union: Geometry,
    segments: gpd.GeoDataFrame,
    max_segment_length: float,
    get_elevation: Callable[[list[float]],float],
    output_file: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    logger.info(f"{datetime.now()} - start")
    crs = segments.crs
    gdf_triangle, gdf_triangle_segment, gdf_split = get_triangles_and_segments(
        union, get_elevation, crs, max_segment_length
    )
    logger.info(f"{len(gdf_triangle)} triangles, {len(gdf_triangle_segment)} triangle segments")
    simple_graph, edge_gdf, has_no_cycle = get_oriented_graph(gdf_triangle, segments)

    for cycle in list(nx.simple_cycles(simple_graph)):
        logger.debug(f"Cycle: {cycle}")
    if output_file is not None:
        edge_gdf.to_file(output_file, layer="edges")
    assert has_no_cycle  # make sure there is no cycle

    # simplify graph by removing simple nodes (1 incoming edge, 1 outgoing edge)
    graph = remove_interstitial_nodes(simple_graph)
    line_graph = nx.line_graph(graph)
    sorted_edges = list(
        nx.lexicographical_topological_sort(
            line_graph, key=lambda x: line_graph.out_degree(x)
        )
    )
    # reverse the graph (keep the keys) and the sorting too
    graph = reverse(graph)
    sorted_edges.reverse()
    # reverse the edges in the sorted edges (keep the keys)
    sorted_edges = [(v, u, k) for (u, v, k) in sorted_edges]

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

    logger.info(f"{len(graph.edges)} edges")
    # remove edges without any profile
    to_remove = []
    for graph_edge in graph.edges:
        if graph_edge not in edge_profiles:
            to_remove.append(graph_edge)
    for graph_edge in to_remove:
        graph.remove_edge(*graph_edge)
    logger.info(f"{len(graph.edges)} edges after cleanup")
    for graph_node in graph.nodes:
        in_edges = list(graph.in_edges(graph_node, keys=True))
        out_edges = list(graph.out_edges(graph_node, keys=True))
        if (len(in_edges) == 1) & (len(out_edges) >= 1):
            # logger.debug(f"graph_node {graph_node} with {len(in_edges)} and {len(out_edges)}")
            # make sur the last points of in_edge belong to out_edge
            in_edge = in_edges[0]
            if in_edge in edge_profiles:
                in_left, in_right = edge_profiles[in_edge]
                last_left = in_left[-1]
                last_right = in_right[-1]
                # logger.debug(f"last_left={last_left} last_right={last_right}")
                for out_edge in out_edges:
                    if out_edge in edge_profiles:
                        out_left, out_right = edge_profiles[out_edge]
                        modif = False
                        # logger.debug(f"{out_edge} with {out_left[0][0]} and {out_right[0][0]} and {len(out_left)} {len(out_right)}")
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
                            # logger.debug(f"modif {len(out_left)} {len(out_right)}")
                            edge_profiles[out_edge] = out_left, out_right
        elif (len(in_edges) > 1) & (len(out_edges) == 1):
            # logger.debug(f"graph_node {graph_node} with {len(in_edges)} and {len(out_edges)}")
            # make sur the first points of out_edge belong to out_edge
            out_edge = out_edges[0]
            if out_edge in edge_profiles:
                out_left, out_right = edge_profiles[out_edge]
                first_left = out_left[0]
                first_right = out_right[0]
                # logger.debug(f"first_left={first_left} first_right={first_right}")
                for in_edge in in_edges:
                    if in_edge in edge_profiles:
                        in_left, in_right = edge_profiles[in_edge]
                        modif = False
                        # logger.debug(f"{in_edge} with {in_left[0][0]} and {in_right[0][0]} and {len(in_left)} {len(in_right)}")
                        if first_left[0] != in_left[0][0]:
                            in_left.append(
                                (first_left[0], None)
                            )  # we don't keep bottleneck info
                            modif = True
                        if first_right[0] != in_right[0][0]:
                            in_right.append(
                                (first_right[0], None)
                            )  # we don't keep bottleneck info
                            modif = True
                        if modif:
                            # logger.debug(f"modif {len(in_left)} {len(in_right)}")
                            edge_profiles[in_edge] = in_left, in_right

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
        crs=crs,
    )
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

    if output_file is not None:
        gdf_points.to_file(output_file, layer="points", driver="GPKG")
        # we only need to build the gdf if we export it
        gdf_edges_final = gpd.GeoDataFrame(
            {
                "edge_id": list(range(0, len(sorted_edges))),
                "edge_ids": edge_ids,
                "left_profiles": edge_left_profiles,
                "right_profiles": edge_right_profiles,
            },
            geometry=edge_geometries,
            crs=crs,
        )
        gdf_edges_final.to_file(output_file, layer="edges_final", driver="GPKG")
        segments.to_file(output_file, layer="hydro_segment")
        gdf_triangle.to_file(output_file, layer="triangle", driver="GPKG")
        gdf_split.to_file(output_file, layer="triangle_split", driver="GPKG")
        gdf_triangle_segment.to_file(
            output_file, layer="triangle_segment", driver="GPKG"
        )
    logger.info(f"{datetime.now()} - all done!")
    return gdf_points, gdf_split


def process_all(
    surfaces: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    max_segment_length: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
    get_elevation: Callable[[list[float]],float]|None = None,
    output_file: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
        # union = shapely.union_all(surfaces.geometry).simplify(0.1, preserve_topology=True)
    union = shapely.union_all(surfaces.geometry).segmentize(max_segment_length)
    if get_elevation is None:
        # not elevation function given, we create one using the IGN API
        # sample points
        number_of_points = int(
            union.area / 10000
        )  # computed according to the surface of union (1 point per hectare)
        sample_multipoint: shapely.MultiPoint = uniform(union, number_of_points, 42) # type: ignore
        sample_points = [(p.x, p.y) for p in sample_multipoint.geoms]
        transformer = Transformer.from_crs(surfaces.crs, 4326)
        elevations = throttle_requests(list(transformer.itransform(sample_points)))
        assert len(elevations) == len(sample_points)
        get_elevation = interpolate(sample_points, elevations)

    res = build_triangles_and_points(union, segments, max_segment_length, get_elevation, output_file)
    if res is None:
        logger.error("build_triangles_and_points Failed")
        return None
    points, triangles = res
    logger.info(f"{datetime.now()} - optimize_all_segments ({len(points)} points, {len(triangles)} triangles)")
    points = optimize_all_segments(points, triangles, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
    if points is None:
        logger.error("optimize_all_segments Failed")
        return None
    logger.info(f"{datetime.now()} - optimize triangles ({len(triangles)})")
    final_triangles_gdf = optimize_triangles(points, triangles)
    logger.info(f"{datetime.now()} - optimized triangles ({len(final_triangles_gdf)})")
    if output_file:
        surfaces.to_file(output_file, layer="surface")
        points.to_file(output_file, layer="points_optimised")
        final_triangles_gdf.to_file(output_file, layer="triangles_optimised")
    logger.info(f"{datetime.now()} - All done!")
    return points, final_triangles_gdf

def main(
    srs: str,
    in_box: tuple[float, float, float, float],
    max_segment_length: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
    output_file: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    logger.info(f"{datetime.now()} - start")
    box = (in_box[0], in_box[1], in_box[2], in_box[3], srs)
    (surfaces, segments, _) = get_hydro_data(box, srs)
    if (surfaces is None) or (segments is None):
        logger.error("no surface data or no segment data")
        return None
    return process_all(surfaces, segments, max_segment_length, alpha, beta, gamma, delta, epsilon, None, output_file)

def optimal_triangle_height(p1: Point, p2: Point, p3: Point) -> Point:
    """
    Calculate optimal p3 to make triangle as horizontal as possible.

    Args:
        p1: (x1, y1, z1) - first point (fixed)
        p2: (x2, y2, z2) - second point (fixed)
        p3: (x3, y3, z3) - third point (optimal z3 will be calculated)

    Returns:
        p3: third point with optimal height
    """
    # 2D vectors
    v12 = np.array([p2.x - p1.x, p2.y - p1.y])
    v13 = np.array([p3.x - p1.x, p3.y - p1.y])
    # Slope from P1 to P2
    dz12 = p2.z - p1.z
    # Optimal z3
    z3 = p1.z + dz12 * np.dot(v12, v13) / np.dot(v12, v12)
    return Point(p3.x, p3.y, z3)

def optimize_triangles(points: gpd.GeoDataFrame, triangles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply the optimized points to the triangles.
    """
    final_triangles = []
    for _, triangle in triangles.iterrows():
        new_triangle = triangle.copy()
        coords = triangle["geometry"].exterior.coords
        triangle_indices = []
        missing_indices = []
        for coord_index, coord in enumerate(coords):
            tol = 1e-9
            mask = (np.abs(points.geometry.x - coord[0]) < tol) & (
                np.abs(points.geometry.y - coord[1]) < tol
            )
            matches = points.geometry[mask]
            if any(matches):
                triangle_indices.append(matches.index[0])
            else:
                missing_indices.append(coord_index)
                triangle_indices.append(-1)
        if len(missing_indices) == 0:
            new_triangle["geometry"] = Polygon(
                [
                    points.geometry.iat[triangle_indices[0]],  # type: ignore
                    points.geometry.iat[triangle_indices[1]],  # type: ignore
                    points.geometry.iat[triangle_indices[2]],  # type: ignore
                    points.geometry.iat[triangle_indices[0]],  # type: ignore
                ]
            )
            final_triangles.append(new_triangle)
        elif len(missing_indices) == 1:
            missing_index = missing_indices[0]
            # there is one point missing, let's guess its height
            index_1 = triangle_indices[(missing_index - 1) % 4]
            index_2 = triangle_indices[(missing_index + 1) % 4]
            p1: Point = points.geometry.iat[index_1]  # type: ignore
            p2: Point = points.geometry.iat[index_2]  # type: ignore
            p3 = Point(coords[missing_index])
            p3_opt = optimal_triangle_height(p1, p2, p3)
            new_triangle["geometry"] = Polygon([p1, p2, p3_opt, p1])
            final_triangles.append(new_triangle)
    return gpd.GeoDataFrame(final_triangles, crs=points.crs)

if __name__ == "__main__":
    logger.setLevel("DEBUG")
    logger.addHandler(logging.StreamHandler())
    srs = "urn:ogc:def:crs:EPSG::2154"
    # box = (1030000, 6280000, 1040000, 6291100)
    box = (1036535, 6289927, 1042268, 6305786)
    # box = (1030000, 6280000, 1040000, 6310000)
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 0.0
    epsilon = 10.0
    output_file = f"triangle_graph_{alpha}_{beta}_{gamma}_{delta}_{epsilon}.gpkg"
    max_segment_length = 20.0
    res = main(srs, box, max_segment_length, alpha, beta, gamma, delta, epsilon, output_file)