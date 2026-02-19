from typing import Optional, Union
import pandas as pd
import geopandas as gpd
from geopandas.tools._random import uniform
import geojson
from owslib.wfs import WebFeatureService
import shapely
from shapely import LineString, Point
from shapely.geometry.geo import shape
from flatten.split import get_segments
from flatten.elevation import throttle_requests
import networkx as nx
import numpy
from pyproj import Transformer
from pyinterpolate import inverse_distance_weighting
from flatten.utils import get_bottlenecks, get_profiles

def get_wfs_data(url: str, type_name: str, box, srs) -> gpd.GeoDataFrame | None:
    # Specify the url for the backend.
    wfs20 = WebFeatureService(url=url, version='2.0.0')
    if wfs20:
        # Specify parameters (read data in json format) and fetch data from WFS using requests
        response = wfs20.getfeature(
            typename=type_name, bbox=box, srsname=srs, outputFormat='application/json')
        # Create GeoDataFrame from geojson and set coordinate reference system
        return gpd.GeoDataFrame.from_features(geojson.loads(response.read()), crs=srs)
    return None

def get_hydro_data(box, srs):
    surface = get_wfs_data("https://data.geopf.fr/wfs",
                            'BDTOPO_V3:surface_hydrographique',
                            box, srs)
    segment = get_wfs_data("https://data.geopf.fr/wfs",
                            'BDTOPO_V3:troncon_hydrographique',
                            box, srs)
    nodes = get_wfs_data("https://data.geopf.fr/wfs",
                         'BDTOPO_V3:noeud_hydrographique',
                            box, srs)
    if (surface is None) or (segment is None) or (nodes is None):
        print("No surface or no segment found.")
        return None
    print("Surfaces:", len(surface))
    print("Segments:", len(segment))
    print("Nodes:", len(nodes))
    surface = surface.query('nature == "Ecoulement naturel"')
    print("mask:", len(surface))
    surface = surface.query('persistance == "Permanent"')
    print("mask:", len(surface))
    # segment = segment.query('nature == "Ecoulement naturel" OR nature == "Conduit buse"')
    segment = segment[(segment["nature"] == "Ecoulement naturel") | (segment["nature"] == "Conduit buse")]
    print("mask:", len(segment))
    segment = segment[segment['liens_vers_surface_hydrographique'].isin(surface["cleabs"]) | (segment["nature"] == "Conduit buse")]
    print("mask:", len(segment))
    return (surface, segment, nodes)

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

# def remove_interstitial_nodes(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     g = nx.Graph()
#     g.add_edges_from([(start, end) for start, end in zip(edges["start"], edges["end"])])
#     from collections import deque
#     q = deque()
#     q.extend(g.nodes)
#     while q:
#         cur = q.popleft()
#         neighbors = g.neighbors(cur)


#     return edges

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
    # touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    # edges = []
    # edges_source = []
    # edges_target = []
    # # create links between triangle centroids
    # for triangle_id_left, triangle_id_right, triangle_elevation_rge_left, triangle_elevation_rge_right in zip(
    #     touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"], 
    #     touching_triangles["triangle_elevation_rge_left"], touching_triangles["triangle_elevation_rge_right"]):
    #     if triangle_id_left != triangle_id_right:
    #         geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"] # type: ignore
    #         geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"] # type: ignore
    #         if geom_left.intersection(geom_right).length > 0:
    #             # print(triangle_id_left, triangle_id_right)
    #             # height_left = triangle_height(geom_left)
    #             # height_right = triangle_height(geom_right)
    #             # if height_left > height_right:
    #             if triangle_elevation_rge_left > triangle_elevation_rge_right:
    #                 edges.append(LineString([geom_left.centroid, geom_right.centroid]))
    #                 edges_source.append(f"TRIANGLE_{triangle_id_left}")
    #                 edges_target.append(f"TRIANGLE_{triangle_id_right}")
    #             else:
    #                 edges.append(LineString([geom_right.centroid, geom_left.centroid]))
    #                 edges_source.append(f"TRIANGLE_{triangle_id_right}")
    #                 edges_target.append(f"TRIANGLE_{triangle_id_left}")
    # def get_point_coordinate(p: Point):
    #     return shapely.get_coordinates(p)[0,:]
    # gdf_triangle_centroids = gdf_triangle.copy()
    # gdf_triangle_centroids.geometry = gdf_triangle_centroids.geometry.centroid
    # def get_triangle(p: Point):
    #     triangles_containing = gdf_triangle[gdf_triangle.geometry.contains(p)]
    #     # print(p,"triangles_containing",triangles_containing)
    #     triangle_id = triangles_containing.iloc[0]["triangle_id"]
    #     triangle_coord = get_point_coordinate(triangles_containing.iloc[0]["geometry"].centroid)
    #     return triangle_id, triangle_coord
    # for node_id, node_type, node_geom in zip(nodes["node_id"], nodes["node_type"], nodes["geometry"]):
    #     node_coord = get_point_coordinate(node_geom)
    #     triangles_containing = gdf_triangle[gdf_triangle.geometry.contains(node_geom)]
    #     if len(triangles_containing) == 1:
    #         print("1 triangle")
    #         triangle_coord = get_point_coordinate(triangles_containing.iloc[0]["geometry"].centroid)
    #         triangle_id = triangles_containing.iloc[0]["triangle_id"]
    #         if node_type == "IN":
    #             edges.append(LineString([node_coord, triangle_coord]))
    #             edges_source.append(f"SOURCE_{node_id}")
    #             edges_target.append(f"TRIANGLE_{triangle_id}")
    #         else:
    #             edges.append(LineString([triangle_coord, node_coord]))
    #             edges_source.append(f"TRIANGLE_{triangle_id}")
    #             edges_target.append(f"TARGET_{node_id}")
    #     else:
    #         print(len(triangles_containing),"triangles")
    #         nearest = gdf_triangle_centroids.sindex.nearest(node_geom)
    #         ind: int = nearest[1][0]
    #         triangle_id, triangle_coord = get_triangle(gdf_triangle_centroids.geometry.iat[ind])
    #         if node_type == "IN":
    #             edges.append(LineString([node_coord, triangle_coord]))
    #             edges_source.append(f"SOURCE_{node_id}")
    #             edges_target.append(f"TRIANGLE_{triangle_id}")
    #         else:
    #             edges.append(LineString([triangle_coord, node_coord]))
    #             edges_source.append(f"TRIANGLE_{triangle_id}")
    #             edges_target.append(f"TARGET_{node_id}")
    # gdf_edges = gpd.GeoDataFrame(
    #     pd.DataFrame({
    #         'edge_id': list(range(0, len(edges))),
    #         'source_id': edges_source,
    #         'target_id': edges_target,
    #     }),
    #     geometry=edges,
    #     crs=surfaces.crs
    # )

    # hydro_graph = nx.Graph()
    # for u,v in zip(edges_source,edges_target):
    #     hydro_graph.add_edge(u,v)
    # directed_graph = orient_hydrograph(hydro_graph, filter(lambda x: x.startswith("SOURCE_"), edges_source), filter(lambda x: x.startswith("TARGET_"), edges_target))
    # print(directed_graph)
    # directed_edge_source = []
    # directed_edge_target = []
    # directed_edge_geom = []
    # def get_coord(id: str):
    #     id_type, id_val = id.split("_")
    #     if id_type == "TRIANGLE":
    #         return get_point_coordinate(gdf_triangle_centroids.iloc[int(id_val)]["geometry"])
    #     node_ = nodes[nodes["node_id"] == id_val]
    #     # print(id,id_val,node_,node_.iloc[0]["geometry"])
    #     return get_point_coordinate(node_.iloc[0]["geometry"])
    # for u,v in directed_graph.edges:
    #     directed_edge_source.append(u)
    #     directed_edge_target.append(v)
    #     print("U=",get_coord(u))
    #     print("V=",get_coord(v))
    #     directed_edge_geom.append(LineString([get_coord(u), get_coord(v)])) # type: ignore
    # gdf_edges = gpd.GeoDataFrame(
    #     pd.DataFrame({
    #         'edge_id': list(range(0, len(directed_edge_geom))),
    #         'source_id': directed_edge_source,
    #         'target_id': directed_edge_target,
    #     }),
    #     geometry=directed_edge_geom,
    #     crs=surfaces.crs
    # )

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
        segment = add_intersection_column(segment, res_intersection, "cleabs", "n_intersections")
        gdf_shared_triangle_segment = gdf_triangle_segment.query('n_segments > 1')
        res_shared_intersection = res_intersection[res_intersection['triangle_segment_id'].isin(gdf_shared_triangle_segment["triangle_segment_id"])]
        segment = add_intersection_column(segment, res_shared_intersection, "cleabs", "n_shared_intersections")
        directed_graph = get_directed_graph(segment, True)
        edges = list(nx.topological_sort(nx.line_graph(directed_graph)))
        segment_order = []
        segment_id = []
        point_order = []
        point_position = []
        point_geom = []
        point_bottlenecks = []
        # TODO add bottleneck information
        for edge_order, edge in enumerate(edges):
            attributes = directed_graph.get_edge_data(*edge)
            segment_key = attributes["edge_key"]
            segment_geometry = attributes["edge_geometry"]
            profiles = get_profiles(gdf_triangle_segment[gdf_triangle_segment["hydro_segment_id"] == segment_key], line=segment_geometry)
            if profiles:
                left, right = profiles
                for i, l in enumerate(left):
                    segment_order.append(edge_order)
                    segment_id.append(segment_key)
                    point_order.append(i)
                    point_position.append("left")
                    point_geom.append(Point(l[0]))
                    point_bottlenecks.append(l[1])
                for i, r in enumerate(right):
                    segment_order.append(edge_order)
                    segment_id.append(segment_key)
                    point_order.append(i)
                    point_position.append("right")
                    point_geom.append(Point(r[0]))
                    point_bottlenecks.append(r[1])

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
        # gdf_edges.to_file(output_file, layer="graph_edges")

    print("All done!")

if __name__ == "__main__":
    main()
