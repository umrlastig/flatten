from typing import Union
import pandas as pd
import geopandas as gpd
from shapely import Point, MultiPoint
from shapely.geometry.geo import shape
import networkx as nx
from flatten.triangle_graph import get_graph
from flatten.utils import (
    get_profiles,
    split_by_first_false,
    merge_profile_points,
)
from flatten.wfs import get_hydro_data
import logging

logger = logging.getLogger(__name__)


def get_directed_graph(segment: gpd.GeoDataFrame, reverse=False) -> nx.DiGraph:
    directed_graph = nx.DiGraph()
    for s in segment.iterfeatures():
        geom = shape(s["geometry"])
        nodes = [
            str(s["properties"]["lien_vers_noeud_hydrographique_ini"]),
            str(s["properties"]["lien_vers_noeud_hydrographique_fin"]),
        ]
        # if reverse XOR already inversed linestring
        if reverse ^ (s["properties"]["sens_de_l_ecoulement"] != "Sens direct"):
            geom = geom.reverse()
            nodes.reverse()
        directed_graph.add_edge(
            *nodes, edge_key=str(s["properties"]["cleabs"]), edge_geometry=geom
        )
    return directed_graph


def get_sources_and_targets(segment: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    directed_graph = get_directed_graph(segment)
    sources = [
        x
        for x in directed_graph.nodes()
        if directed_graph.out_degree(x) == 1 and directed_graph.in_degree(x) == 0
    ]
    targets = [
        x
        for x in directed_graph.nodes()
        if directed_graph.out_degree(x) == 0 and directed_graph.in_degree(x) == 1
    ]
    # print(directed_graph.in_edges)
    node_id = []
    node_edge_id = []
    node_geom = []
    node_type = []
    print("IN", len(sources))
    for source in sources:
        u, v = list(directed_graph.out_edges(source))[0]
        attributes = directed_graph.get_edge_data(u, v)
        key = attributes["edge_key"]
        seg = segment.loc[segment["cleabs"] == key]
        line = seg.geometry.item()
        first = Point(line.coords[0])
        node_id.append(source)
        node_edge_id.append(seg["cleabs"].item())
        node_geom.append(first)
        node_type.append("IN")
    # print(directed_graph.out_edges)
    print("OUT", len(targets))
    for target in targets:
        u, v = list(directed_graph.in_edges(target))[0]
        attributes = directed_graph.get_edge_data(u, v)
        key = attributes["edge_key"]
        seg = segment.loc[segment["cleabs"] == key]
        line = seg.geometry.item()
        last = Point(line.coords[-1])
        node_id.append(target)
        node_edge_id.append(seg["cleabs"].item())
        node_geom.append(last)
        node_type.append("OUT")
    return gpd.GeoDataFrame(
        pd.DataFrame(
            {"node_id": node_id, "node_edge_id": node_edge_id, "node_type": node_type}
        ),
        geometry=node_geom,
        crs=segment.crs,
    )

def add_intersection_column(
    gdf: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame, id_name: str, column_name: str
) -> gpd.GeoDataFrame:
    intersection_counts = (
        overlay.groupby(id_name)
        .size()  # gives the number of rows per group
        .rename(column_name)  # name of the new column
        .reset_index()
    )
    gdf = gdf.merge(
        intersection_counts,
        on=id_name,
        how="left",  # we keep all segments, even those without intersection
    )
    # fill missing values
    gdf[column_name] = gdf[column_name].fillna(0).astype(int)
    return gdf


def choose_id(ids: list[str], orders: list[int], type: list[str]) -> Union[str, None]:
    """
    Helper used inside the group-by aggregation.
    * If ``ids`` is empty → return None.
    * Else, return the ID whose corresponding (order, type == principal) is maximal
    """
    if not ids:  # no intersecting segment
        return None
    # max by order and type == principal
    best_record = max(zip(ids, orders, type), key=lambda r: (r[1], r[2] == "Principal"))
    return best_record[0]


def filter_intersections_on_endpoints(
    gdf: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    def line_endpoints(line):
        """Return a MultiPoint with the start‑ and end‑point of a LineString."""
        # line.coords returns a sequence of (x, y) tuples
        start, end = line.coords[0], line.coords[-1]
        return MultiPoint([Point(start), Point(end)])

    # print("overlay\n", overlay.head())
    # Create a GeoSeries whose index aligns with gdf_a
    endpoints = gdf.geometry.apply(line_endpoints)  # type: ignore
    overlay = overlay.join(endpoints.rename("_endpoints"), on="triangle_segment_id")
    mask_not_on_endpoint = ~overlay.geometry.intersects(overlay["_endpoints"])  # type: ignore
    # tol = 1e-6
    # mask = overlay.geometry.distance(overlay["_endpoints"]) > tol
    clean_result = overlay[mask_not_on_endpoint].copy()
    # Drop the helper column
    clean_result = clean_result.drop(columns=["_endpoints"])
    # print("clean_result\n", clean_result.head())
    return clean_result


def main():
    srs = "urn:ogc:def:crs:EPSG::2154"
    box = (1036535, 6289927, 1042268, 6305786, srs)
    output_file = "temp.gpkg"

    r = get_hydro_data(box, srs)
    if r is None:
        logger.error("no data")
        return
    (surface, segment, node) = r
    gdf_nodes = get_sources_and_targets(segment)
    gdf_triangle, gdf_triangle_segment, split = get_graph(surface, 20.0)
    # keep only the unconstrained segments
    unconstrained_triangle_segment = gdf_triangle_segment.query('type != "constrained"')
    # intersect them with the hydro segments
    res_intersection = unconstrained_triangle_segment.overlay(
        segment, how="intersection", keep_geom_type=False
    )
    res_intersection = filter_intersections_on_endpoints(
        unconstrained_triangle_segment, res_intersection
    )
    # count the intersection for the triangle segments
    gdf_triangle_segment = add_intersection_column(
        gdf_triangle_segment, res_intersection, "triangle_segment_id", "n_segments"
    )
    # add the id of the most important segment
    agg = (
        res_intersection.groupby("triangle_segment_id")
        .apply(
            lambda df: choose_id(
                df["cleabs"].tolist(),  # type: ignore
                df["numero_d_ordre"].tolist(),  # type: ignore
                df["type_de_bras"].tolist(),  # type: ignore
            )  # type: ignore
        )  # type: ignore
        .reset_index()
        .rename(columns={0: "hydro_segment_id"})
    )
    gdf_triangle_segment = (
        gdf_triangle_segment.copy()
        .reset_index()
        .merge(agg, how="left", on="triangle_segment_id")
    )
    gdf_triangle_segment["hydro_segment_id"] = gdf_triangle_segment[
        "hydro_segment_id"
    ].astype("str")
    logger.info(f"segment\n{segment}")
    directed_graph = get_directed_graph(segment, True)
    edges = list(nx.topological_sort(nx.line_graph(directed_graph)))
    segment_order = []
    segment_id = []
    point_order = []
    point_position = []
    point_geom = []
    point_bottlenecks = []
    edge_profiles = dict()
    for edge_order, edge in enumerate(edges):
        attributes = directed_graph.get_edge_data(*edge)
        segment_key = attributes["edge_key"]
        segment_geometry = attributes["edge_geometry"]
        profiles = get_profiles(
            gdf_triangle_segment[
                gdf_triangle_segment["hydro_segment_id"] == segment_key
            ],
            line=segment_geometry,
        )
        if profiles:
            left, right = profiles
            edge_profiles[edge] = (left, right)
    logger.info(f"{len(directed_graph.edges)} edges")
    # move shared points up or down
    for graph_edge in directed_graph.edges:
        if graph_edge in edge_profiles:
            left, right = edge_profiles[graph_edge]
            attributes = directed_graph.get_edge_data(*graph_edge)
            segment_key = attributes["edge_key"]
            if (left[0][1] is True) & (right[0][1] is True):
                in_edges = list(directed_graph.in_edges(graph_edge[0]))
                # print(
                #     "shared at start",
                #     segment_key,
                #     len(in_edges),
                #     "with",
                #     len(left),
                #     len(right),
                # )
                if len(in_edges) == 1:
                    in_edge = in_edges[0]
                    sub_left, left = split_by_first_false(left)
                    sub_right, right = split_by_first_false(right)
                    # print("\tleft", len(sub_left), len(left))
                    # print("\tright", len(sub_right), len(right))
                    if in_edge in edge_profiles:
                        in_left, in_right = edge_profiles[in_edge]
                        in_left.extend(sub_left)
                        in_right.extend(sub_right)
                        edge_profiles[in_edge] = (in_left, in_right)
                    else:
                        edge_profiles[in_edge] = (sub_left, sub_right)
                    if (len(left) > 0) & (len(right) > 0):
                        edge_profiles[graph_edge] = (left, right)
                    else:
                        edge_profiles.pop(graph_edge)
            if (len(left) > 0) & (len(right) > 0):
                if (left[-1][1] is True) & (right[-1][1] is True):
                    out_edges = list(directed_graph.out_edges(graph_edge[1]))
                    # print(
                    #     "shared at end",
                    #     segment_key,
                    #     len(out_edges),
                    #     "with",
                    #     len(left),
                    #     len(right),
                    # )
                    if len(out_edges) == 1:
                        out_edge = out_edges[0]
                        sub_left, left = split_by_first_false(left, reverse=True)
                        sub_right, right = split_by_first_false(right, reverse=True)
                        # print("\tleft", len(sub_left), len(left))
                        # print("\tright", len(sub_right), len(right))
                        if out_edge in edge_profiles:
                            out_left, out_right = edge_profiles[out_edge]
                            sub_left.extend(out_left)
                            sub_right.extend(out_right)
                            edge_profiles[out_edge] = (sub_left, sub_right)
                        else:
                            edge_profiles[out_edge] = (sub_left, sub_right)
                        if (len(left) > 0) & (len(right) > 0):
                            edge_profiles[graph_edge] = (left, right)
                        else:
                            if graph_edge in edge_profiles:
                                edge_profiles.pop(graph_edge)
    # remove edges without any profile
    to_remove = []
    for graph_edge in directed_graph.edges:
        if graph_edge not in edge_profiles:
            to_remove.append(graph_edge)
    for graph_edge in to_remove:
        directed_graph.remove_edge(*graph_edge)
    print(len(directed_graph.edges), "edges after cleanup")
    for graph_node in directed_graph.nodes:
        in_edges = list(directed_graph.in_edges(graph_node))
        out_edges = list(directed_graph.out_edges(graph_node))
        if (len(in_edges) == 1) & (len(out_edges) == 1):
            # make sur the last points of in_edge belong to out_edge
            in_edge = in_edges[0]
            if in_edge in edge_profiles:
                in_left, in_right = edge_profiles[in_edge]
                last_left = in_left[-1]
                last_right = in_right[-1]
                out_edge = out_edges[0]
                if out_edge in edge_profiles:
                    out_left, out_right = edge_profiles[out_edge]
                    modif = False
                    print("out_left", out_left)
                    print("out_right", out_right)
                    if last_left[0] != out_left[0][0]:
                        print(
                            "adding a left point",
                            last_left[0],
                            "before",
                            out_left[0][0],
                        )
                        out_left.insert(
                            0, (last_left[0], False, None)
                        )  # we don't keep bottleneck info
                        modif = True
                    if last_right[0] != out_right[0][0]:
                        print(
                            "adding a right point",
                            last_right[0],
                            "before",
                            out_right[0][0],
                        )
                        out_right.insert(
                            0, (last_right[0], False, None)
                        )  # we don't keep bottleneck info
                        modif = True
                    if modif:
                        edge_profiles[out_edge] = out_left, out_right

    for edge_order, (edge, (left, right)) in enumerate(edge_profiles.items()):
        attributes = directed_graph.get_edge_data(*edge)
        segment_key = attributes["edge_key"]
        segment_geometry = attributes["edge_geometry"]
        merged_left = merge_profile_points(left)
        merged_right = merge_profile_points(right)

        def add_points(profile, position):
            for i, p in enumerate(profile):
                segment_order.append(edge_order)
                segment_id.append(segment_key)
                point_order.append(i)
                point_position.append(position)
                point_geom.append(Point(p[0]))
                point_bottlenecks.append(p[2])

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
        crs=surface.crs,
    )
    gdf_points.to_file(output_file, layer="points", driver="GPKG")
    surface.to_file(output_file, layer="surface")
    segment.to_file(output_file, layer="segment")
    node.to_file(output_file, layer="node")
    gdf_triangle.to_file(output_file, layer="triangle", driver="GPKG")
    gdf_triangle_segment.to_file(output_file, layer="triangle_segment", driver="GPKG")
    gdf_nodes.to_file(output_file, layer="graph_node")
    print("All done!")


if __name__ == "__main__":
    main()
