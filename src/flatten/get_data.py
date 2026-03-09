from typing import Optional, Union
import pandas as pd
import geopandas as gpd
from geopandas.tools._random import uniform
import shapely
from shapely import LineString, Point, MultiPoint
from shapely.geometry.geo import shape
from flatten.split import get_segments
from flatten.elevation import throttle_requests
import networkx as nx
import numpy
from pyproj import Transformer
from pyinterpolate import inverse_distance_weighting
from flatten.utils import get_bottlenecks, get_profiles, split_by_first_false, merge_profile_points
from flatten.wfs import get_hydro_data

def get_directed_graph(segment: gpd.GeoDataFrame, reverse = False) -> nx.DiGraph:
    directed_graph = nx.DiGraph()
    for s in segment.iterfeatures():
        geom = shape(s["geometry"])
        nodes = [
            str(s["properties"]["lien_vers_noeud_hydrographique_ini"]),
            str(s["properties"]["lien_vers_noeud_hydrographique_fin"])
        ]
        # if reverse XOR already inversed linestring
        if reverse ^ (s["properties"]["sens_de_l_ecoulement"] != 'Sens direct'):
            geom = geom.reverse()
            nodes.reverse()
        directed_graph.add_edge(
            *nodes,
            edge_key = str(s["properties"]["cleabs"]),
            edge_geometry = geom
        )
    return directed_graph

def get_sources_and_targets(segment: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    directed_graph = get_directed_graph(segment)
    sources = [x for x in directed_graph.nodes() if directed_graph.out_degree(x)==1 and directed_graph.in_degree(x)==0]
    targets = [x for x in directed_graph.nodes() if directed_graph.out_degree(x)==0 and directed_graph.in_degree(x)==1]
    # print(directed_graph.in_edges)
    node_id = []
    node_edge_id = []
    node_geom = []
    node_type = []
    print('IN',len(sources))
    for source in sources:
        u, v = list(directed_graph.out_edges(source))[0]
        attributes = directed_graph.get_edge_data(u,v)
        key = attributes["edge_key"]
        seg = segment.loc[segment['cleabs'] == key]
        line = seg.geometry.item()
        first = Point(line.coords[0])
        node_id.append(source)
        node_edge_id.append(seg["cleabs"].item())
        node_geom.append(first)
        node_type.append("IN")
    # print(directed_graph.out_edges)
    print('OUT',len(targets))
    for target in targets:
        u, v = list(directed_graph.in_edges(target))[0]
        attributes = directed_graph.get_edge_data(u,v)
        key = attributes["edge_key"]
        seg = segment.loc[segment['cleabs'] == key]
        line = seg.geometry.item()
        last = Point(line.coords[-1])
        node_id.append(target)
        node_edge_id.append(seg["cleabs"].item())
        node_geom.append(last)
        node_type.append("OUT")
    return gpd.GeoDataFrame(
        pd.DataFrame({'node_id': node_id, 'node_edge_id': node_edge_id, 'node_type': node_type}),
        geometry=node_geom,
        crs=segment.crs
    )

def get_triangles(union, crs, sample_points, elevations) -> gpd.GeoDataFrame:
    triangles = shapely.constrained_delaunay_triangles(union)
    def triangle_height(geom: shapely.Polygon) -> float:
        # keep only 3 points and select the z value
        z_values = shapely.get_coordinates(geom, include_z=True)[:3, 2]
        z_values = z_values[~numpy.isnan(z_values)]
        return z_values.mean()
    print("Triangles:", len(triangles.geoms))
    # use crs from input file
    triangle_list = [p for p in triangles.geoms]
    triangle_heights = [triangle_height(p) for p in triangles.geoms] # type: ignore
    triangle_centroids = [[p.centroid.x, p.centroid.y] for p in triangles.geoms]
    triangle_elevations = []
    for c in triangle_centroids:
        triangle_elevations.append(
            inverse_distance_weighting(
                unknown_location=c,
                known_values=elevations,
                known_geometries=sample_points,
                power=2.
            )
        )
    gdf_triangle = gpd.GeoDataFrame(
        pd.DataFrame({
            'triangle_id': list(range(0, len(triangle_list))),
            'triangle_height': triangle_heights,
            'triangle_elevation_rge': triangle_elevations,
        }),
        geometry=triangle_list,
        crs=crs
    )
    return gdf_triangle

def get_graph(surfaces: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    union = shapely.union_all(surfaces.geometry).simplify(0.1, preserve_topology=True)
    # sample points
    number_of_points = int(union.area / 10000) # computed according to the surface of union
    print("number_of_points",number_of_points)
    sample_multipoint: shapely.MultiPoint = uniform(union, number_of_points, 42) # type: ignore
    sample_points = [(p.x, p.y) for p in sample_multipoint.geoms]
    
    transformer = Transformer.from_crs(surfaces.crs, 4326)
    elevations = throttle_requests(list(transformer.itransform(sample_points)))
    assert(len(elevations) == len(sample_points))

    gdf_triangle = get_triangles(union, surfaces.crs, sample_points, elevations)
    # triangle_graph: gpd.GeoDataFrame = get_triangle_graph(gdf_triangle)
    # triangle_graph = neatnet.remove_interstitial_nodes(triangle_graph) # type: ignore
    # print("triangle_graph",len(triangle_graph))
    # oriented_triangle_graph = orient_triangle_graph(triangle_graph, gdf_triangle)

    triangle_segment = list(set(get_segments(gdf_triangle.geometry)))
    # determine if segments belong to the boundary: if they do, they must be constraints
    triangle_segment_constraint = list(map(lambda s: union.boundary.contains(s), triangle_segment))
    gdf_triangle_segment = gpd.GeoDataFrame(
        {
            'triangle_segment_id': list(range(0, len(triangle_segment))),
            'constraint': triangle_segment_constraint
        },
        geometry=triangle_segment,
        crs=surfaces.crs
    )
    # merge with bottlenecks
    gdf_triangle_segment["type"] = gdf_triangle_segment["constraint"].apply(lambda b: "constrained" if b else "unconstrained")
    gdf_triangle_segment = gdf_triangle_segment.rename(columns={"triangle_segment_id": "tmp_id"})
    gdf_bottlenecks = get_bottlenecks(gdf_triangle, gdf_triangle_segment)
    def add_height(row) -> Optional[LineString]:
        line = row["geometry"]
        if not line:
            return None
        coordinates: list[list[float]] = shapely.get_coordinates(line, include_z=False).tolist()        
        height0 = inverse_distance_weighting(
            unknown_location=coordinates[0],
            known_values=elevations,
            known_geometries=sample_points,
            power=2.
        )
        height1 = inverse_distance_weighting(
            unknown_location=coordinates[-1],
            known_values=elevations,
            known_geometries=sample_points,
            power=2.
        )
        return LineString([(*coordinates[0], height0), (*coordinates[-1], height1)])
    # gdf_bottlenecks["geometry"] = gdf_bottlenecks.apply(add_height, axis=1)
    gdf_bottlenecks["constraint"] = False
    gdf_bottlenecks["type"] = "bottleneck"
    gdf_bottlenecks = gdf_bottlenecks.rename(columns={"triangle_id": "tmp_id"})
    gdf_bottlenecks = gdf_bottlenecks.set_crs(gdf_triangle_segment.crs, allow_override=True) # type: ignore

    unified: gpd.GeoDataFrame = pd.concat([gdf_triangle_segment, gdf_bottlenecks], ignore_index=True, sort=False) # type: ignore
    # Cast the constraint column to a proper boolean dtype (in case concat made it object)
    unified["constraint"] = unified["constraint"].astype(bool)
    # Re‑order columns for readability (optional)
    unified.insert(0, "triangle_segment_id", range(len(unified)))   # 0‑based integer IDs
    unified = unified[["triangle_segment_id", "constraint", "type", "geometry"]]
    # recompute height for all segments
    unified["geometry"] = unified.apply(add_height, axis=1)
    unified = unified.set_crs(gdf_triangle_segment.crs, allow_override=True) # type: ignore
    return gdf_triangle, unified# type: ignore #, oriented_triangle_graph#gdf_edges

def add_intersection_column(gdf: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame, id_name: str, column_name: str) -> gpd.GeoDataFrame:
    intersection_counts = (
        overlay.groupby(id_name)
        .size()                # gives the number of rows per group
        .rename(column_name)   # name of the new column
        .reset_index()
    )
    gdf = gdf.merge(
        intersection_counts,
        on=id_name,
        how="left",# we keep all segments, even those without intersection
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
    if not ids: # no intersecting segment
        return None
    # max by order and type == principal
    best_record = max(
        zip(ids, orders, type),
        key=lambda r: (r[1], r[2] == "Principal")
    )
    return best_record[0]

def filter_intersections_on_endpoints(gdf: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def line_endpoints(line):
        """Return a MultiPoint with the start‑ and end‑point of a LineString."""
        # line.coords returns a sequence of (x, y) tuples
        start, end = line.coords[0], line.coords[-1]
        return MultiPoint([Point(start), Point(end)])

    print("overlay\n",overlay.head())
    # Create a GeoSeries whose index aligns with gdf_a
    endpoints = gdf.geometry.apply(line_endpoints) # type: ignore
    overlay = overlay.join(
        endpoints.rename("_endpoints"),
        on="triangle_segment_id"
    )
    mask_not_on_endpoint = ~overlay.geometry.intersects(overlay["_endpoints"]) # type: ignore
    # tol = 1e-6
    # mask = overlay.geometry.distance(overlay["_endpoints"]) > tol
    clean_result = overlay[mask_not_on_endpoint].copy()
    # Drop the helper column
    clean_result = clean_result.drop(columns=["_endpoints"])
    print("clean_result\n",clean_result.head())
    return clean_result
def main():
    srs = "urn:ogc:def:crs:EPSG::2154"
    box = (1036535, 6289927, 1042268, 6305786, srs)
    output_file = "temp.gpkg"

    r = get_hydro_data(box, srs)
    if r:
        (surface, segment, node) = r
        gdf_nodes = get_sources_and_targets(segment)
        gdf_triangle, gdf_triangle_segment = get_graph(surface)
        # keep only the unconstrained segments
        unconstrained_triangle_segment = gdf_triangle_segment.query('type != "constrained"')
        # intersect them with the hydro segments
        res_intersection = unconstrained_triangle_segment.overlay(segment, how='intersection', keep_geom_type=False)
        res_intersection = filter_intersections_on_endpoints(unconstrained_triangle_segment, res_intersection)
        # count the intersection for the triangle segments
        gdf_triangle_segment = add_intersection_column(gdf_triangle_segment, res_intersection, "triangle_segment_id", "n_segments")
        # add the id of the most important segment
        agg = (
            res_intersection.groupby("triangle_segment_id")
            .apply(lambda df: choose_id(df["cleabs"].tolist(), df["numero_d_ordre"].tolist(), df["type_de_bras"].tolist())) # type: ignore
            .reset_index()
            .rename(columns={0: "hydro_segment_id"})
        )
        gdf_triangle_segment = gdf_triangle_segment.copy().reset_index().merge(agg, how="left", on="triangle_segment_id")
        gdf_triangle_segment["hydro_segment_id"] = gdf_triangle_segment["hydro_segment_id"].astype("str")
        # segment = add_intersection_column(segment, res_intersection, "cleabs", "n_intersections")
        # gdf_shared_triangle_segment = gdf_triangle_segment.query('n_segments > 1')
        # res_shared_intersection = res_intersection[res_intersection['triangle_segment_id'].isin(gdf_shared_triangle_segment["triangle_segment_id"])]
        # segment = add_intersection_column(segment, res_shared_intersection, "cleabs", "n_shared_intersections")
        print("segment\n",segment)
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
            profiles = get_profiles(gdf_triangle_segment[gdf_triangle_segment["hydro_segment_id"] == segment_key], line=segment_geometry)
            if profiles:
                left, right = profiles
                edge_profiles[edge] = (left, right)
                # for i, l in enumerate(left):
                #     segment_order.append(edge_order)
                #     segment_id.append(segment_key)
                #     point_order.append(i)
                #     point_position.append("left")
                #     point_geom.append(Point(l[0]))
                #     point_bottlenecks.append(l[1])
                # for i, r in enumerate(right):
                #     segment_order.append(edge_order)
                #     segment_id.append(segment_key)
                #     point_order.append(i)
                #     point_position.append("right")
                #     point_geom.append(Point(r[0]))
                #     point_bottlenecks.append(r[1])
        print(len(directed_graph.edges),"edges")
        # move shared points up or down
        for graph_edge in directed_graph.edges:
            if graph_edge in edge_profiles:
                left, right = edge_profiles[graph_edge]
                attributes = directed_graph.get_edge_data(*graph_edge)
                segment_key = attributes["edge_key"]
                if (left[0][1] is True) & (right[0][1] is True):
                    in_edges = list(directed_graph.in_edges(graph_edge[0]))
                    print("shared at start", segment_key, len(in_edges), "with",len(left),len(right))
                    if len(in_edges) == 1:
                        in_edge = in_edges[0]
                        sub_left, left = split_by_first_false(left)
                        sub_right, right = split_by_first_false(right)
                        print("\tleft", len(sub_left), len(left))
                        print("\tright", len(sub_right), len(right))
                        if in_edge in edge_profiles:
                            in_left, in_right = edge_profiles[in_edge]
                            in_left.extend(sub_left)
                            in_right.extend(sub_right)
                            edge_profiles[in_edge] = (in_left, in_right)
                        else:
                            edge_profiles[in_edge] = (sub_left, sub_right)
                        if (len(left) > 0) & (len(right) >0):
                            edge_profiles[graph_edge] = (left, right)
                        else:
                            edge_profiles.pop(graph_edge)
                if (len(left) > 0) & (len(right) > 0):
                    if (left[-1][1] is True) & (right[-1][1] is True):
                        out_edges = list(directed_graph.out_edges(graph_edge[1]))
                        print("shared at end", segment_key, len(out_edges), "with",len(left),len(right))
                        if len(out_edges) == 1:
                            out_edge = out_edges[0]
                            sub_left, left = split_by_first_false(left, reverse=True)
                            sub_right, right = split_by_first_false(right, reverse=True)
                            print("\tleft", len(sub_left), len(left))
                            print("\tright", len(sub_right), len(right))
                            if out_edge in edge_profiles:
                                out_left, out_right = edge_profiles[out_edge]
                                sub_left.extend(out_left)
                                sub_right.extend(out_right)
                                edge_profiles[out_edge] = (sub_left, sub_right)
                            else:
                                edge_profiles[out_edge] = (sub_left, sub_right)
                            if (len(left) > 0) & (len(right) >0):
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
        print(len(directed_graph.edges),"edges after cleanup")
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
                        print("out_left",out_left)
                        print("out_right",out_right)
                        if last_left[0] != out_left[0][0]:
                            print("adding a left point",last_left[0],"before",out_left[0][0])
                            out_left.insert(0, (last_left[0], False, None)) # we don't keep bottleneck info
                            modif = True
                        if last_right[0] != out_right[0][0]:
                            print("adding a right point",last_right[0],"before",out_right[0][0])
                            out_right.insert(0, (last_right[0], False, None)) # we don't keep bottleneck info
                            modif = True
                        if modif:
                            edge_profiles[out_edge] = out_left, out_right

            # if (len(in_edges) == 1) & (len(out_edges) >= 1):
            #     # make sur the last points of in_edge belong to all out_edges
            #     in_edge = in_edges[0]
            #     if in_edge in edge_profiles:
            #         in_left, in_right = edge_profiles[in_edge]
            #         last_left = in_left[-1]
            #         last_right = in_right[-1]
            #         for out_edge in out_edges:
            #             if out_edge in edge_profiles:
            #                 out_left, out_right = edge_profiles[out_edge]
            #                 modif = False
            #                 if last_left[0] != out_left[0][0]:
            #                     print("adding a left point",last_left[0],"before",out_left[0][0])
            #                     out_left.insert(0, (last_left[0], [])) # we don't keep bottleneck info
            #                     modif = True
            #                 if last_right[0] != out_right[0][0]:
            #                     print("adding a right point",last_right[0],"before",out_right[0][0])
            #                     out_right.insert(0, (last_right[0], [])) # we don't keep bottleneck info
            #                     modif = True
            #                 if modif:
            #                     edge_profiles[out_edge] = out_left, out_right
            # if (len(in_edges) > 1) & (len(out_edges) == 1):
            #     # make sur the first points of out_edge belong to all in_edges
            #     out_edge = out_edges[0]
            #     if out_edge in edge_profiles:
            #         out_left, out_right = edge_profiles[out_edge]
            #         first_left = in_left[0]
            #         first_right = in_right[0]
            #         for in_edge in in_edges:
            #             if in_edge in edge_profiles:
            #                 in_left, in_right = edge_profiles[in_edge]
            #                 modif = False
            #                 if first_left[0] != in_left[-1][0]:
            #                     print("adding a left point",first_left[0],"after",in_left[-1][0])
            #                     in_left.append((first_left[0], [])) # we don't keep bottleneck info
            #                     modif = True
            #                 if first_right[0] != in_right[-1][0]:
            #                     print("adding a right point",first_right[0],"after",in_right[-1][0])
            #                     in_right.append((first_right[0], [])) # we don't keep bottleneck info
            #                     modif = True
            #                 if modif:
            #                     edge_profiles[in_edge] = in_left, in_right

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
            # for i, l in enumerate(left):
            #     segment_order.append(edge_order)
            #     segment_id.append(segment_key)
            #     point_order.append(i)
            #     point_position.append("left")
            #     point_geom.append(Point(l[0]))
            #     point_bottlenecks.append(l[2])
            # for i, r in enumerate(right):
            #     segment_order.append(edge_order)
            #     segment_id.append(segment_key)
            #     point_order.append(i)
            #     point_position.append("right")
            #     point_geom.append(Point(r[0]))
            #     point_bottlenecks.append(r[2])

        gdf_points = gpd.GeoDataFrame(
            {
                'point_id': list(range(0,len(point_geom))),
                'segment_id': segment_id,
                'segment_order': segment_order,
                'point_order': point_order,
                'point_position': point_position,
                'point_bottlenecks': point_bottlenecks
            }, 
            geometry=point_geom, 
            crs=surface.crs
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
