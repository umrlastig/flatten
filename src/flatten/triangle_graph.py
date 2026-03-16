import geopandas as gpd
import shapely
from shapely import LineString
import networkx as nx

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
                if triangle_id_left > triangle_id_right:
                    graph.add_edge(
                        triangle_id_right,
                        triangle_id_left,
                        geometry=LineString([geom_right.centroid, geom_left.centroid]),
                    )
                else:
                    graph.add_edge(
                        triangle_id_left,
                        triangle_id_right,
                        geometry=LineString([geom_left.centroid, geom_right.centroid]),
                    )
    print("graph edges", graph.number_of_edges())
    return graph


def get_triangle_graph_as_gdf(gdf_triangle: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    edges = []
    for triangle_id_left, triangle_id_right in zip(
        touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"]
    ):
        if triangle_id_left < triangle_id_right:
            geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"]  # type: ignore
            geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"]  # type: ignore
            if geom_left.intersection(geom_right).length > 0:
                if triangle_id_left > triangle_id_right:
                    edges.append(LineString([geom_right.centroid, geom_left.centroid]))
                else:
                    edges.append(LineString([geom_left.centroid, geom_right.centroid]))
    gdf_edges = gpd.GeoDataFrame(
        {"edge_id": list(range(0, len(edges)))}, geometry=edges, crs=gdf_triangle.crs
    )
    print("gdf_edges", len(gdf_edges))
    return gdf_edges  # type: ignore

def reverse(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """reverses the directed graph."""
    directed_graph = nx.MultiDiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        line: LineString = data["geometry"]
        edge_ids = data.get("edge_ids", [])
        directed_graph.add_edge(v, u, key=key, geometry = line.reverse(), edge_ids = edge_ids)
    return directed_graph

def remove_interstitial_nodes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Removes nodes of in degree 1 and out_degree 1 from the directed graph."""
    directed_graph = graph.copy()
    nodes_to_remove = []
    for node in directed_graph.nodes:
        if (directed_graph.in_degree(node) == 1) & (directed_graph.out_degree(node) == 1): # type: ignore
            (pred_node, _, in_key, pred_data) = list(directed_graph.in_edges(node,keys=True,data=True))[0] # type: ignore
            (_, succ_node, out_key, succ_data) = list(directed_graph.out_edges(node,keys=True,data=True))[0] # type: ignore
            geom = shapely.line_merge(shapely.GeometryCollection([pred_data["geometry"],succ_data["geometry"]])) # type: ignore
            ids = sorted(list(set(pred_data["edge_ids"] + succ_data["edge_ids"]))) # type: ignore
            directed_graph.remove_edges_from([(pred_node, node, in_key), (node, succ_node, out_key)])
            nodes_to_remove.append(node)
            directed_graph.add_edge(pred_node, succ_node, geometry = geom, edge_ids = ids)
    directed_graph.remove_nodes_from(nodes_to_remove)
    return directed_graph # type: ignore
