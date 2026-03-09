import geopandas as gpd
import shapely
from shapely import Point, LineString
import networkx as nx

def get_triangle_graph_as_nx(gdf_triangle: gpd.GeoDataFrame) -> nx.Graph:
    touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    graph = nx.Graph()
    for triangle_id_left, triangle_id_right in zip(touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"]):
        if triangle_id_left < triangle_id_right:
            geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"] # type: ignore
            geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"] # type: ignore
            if geom_left.intersection(geom_right).length > 0:
                if triangle_id_left > triangle_id_right:
                    graph.add_edge(triangle_id_right, triangle_id_left, geometry = LineString([geom_right.centroid, geom_left.centroid]))
                else:
                    graph.add_edge(triangle_id_left, triangle_id_right, geometry = LineString([geom_left.centroid, geom_right.centroid]))
    print("graph edges",graph.number_of_edges())
    return graph

def get_triangle_graph_as_gdf(gdf_triangle: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    touching_triangles = gdf_triangle.sjoin(gdf_triangle, predicate="touches")
    edges = []
    for triangle_id_left, triangle_id_right in zip(touching_triangles["triangle_id_left"], touching_triangles["triangle_id_right"]):
        if triangle_id_left < triangle_id_right:
            geom_left: shapely.Polygon = gdf_triangle.at[triangle_id_left, "geometry"] # type: ignore
            geom_right: shapely.Polygon = gdf_triangle.at[triangle_id_right, "geometry"] # type: ignore
            if geom_left.intersection(geom_right).length > 0:
                if triangle_id_left > triangle_id_right:
                    edges.append(LineString([geom_right.centroid, geom_left.centroid]))
                else:
                    edges.append(LineString([geom_left.centroid, geom_right.centroid]))
    gdf_edges = gpd.GeoDataFrame({"edge_id": list(range(0, len(edges)))}, geometry=edges, crs=gdf_triangle.crs)
    print("gdf_edges",len(gdf_edges))
    return gdf_edges # type: ignore

def orient_triangle_graph(gdf_edges: gpd.GeoDataFrame, gdf_triangle: gpd.GeoDataFrame):
    def orient(line: LineString) -> LineString:
        coords = shapely.get_coordinates(line, include_z=False).tolist()
        start = coords[0]
        end = coords[-1]
        triangles_containing = gdf_triangle[gdf_triangle.geometry.contains(Point(start))]
        start_height = triangles_containing.iloc[0]["triangle_elevation_rge"]
        triangles_containing = gdf_triangle[gdf_triangle.geometry.contains(Point(end))]
        end_height = triangles_containing.iloc[0]["triangle_elevation_rge"]
        if start_height > end_height:
            return line.reverse()
        return line
    gdf_edges['oriented'] = gdf_edges.apply(lambda row : orient(row.geometry), axis=1)
    gdf_edges = gdf_edges.set_geometry('oriented', crs=gdf_triangle.crs)
    gdf_edges = gdf_edges.drop(columns=["geometry"])
    return gdf_edges

def orient_hydrograph(graph: nx.Graph, sources, targets) -> nx.DiGraph:
    # 1. BFS from targets to compute downstream distance
    from collections import deque
    dist: dict[str,float|None] = {v: None for v in graph.nodes}
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)
    while q:
        cur = q.popleft()
        for nb in graph.neighbors(cur):#neighbors(cur, E):
            if dist[nb] is None:
                dist[nb] = dist[cur] + 1 # type: ignore
                q.append(nb)
    # 2. Orient edges
    directed_graph = nx.DiGraph()
    # directed_edges = []
    for (u, v) in graph.edges:
        du, dv = dist[u], dist[v]
        if du is None or dv is None:
            # component without a target – skip or flag
            continue
        if du > dv:
            # directed_edges.append((u, v))   # u → v
            directed_graph.add_edge(u,v)
        elif dv > du:
            #directed_edges.append((v, u))   # v → u
            directed_graph.add_edge(v,u)
        else:
            # tie – resolve deterministically
            # directed_edges.append((min(u, v), max(u, v)))
            directed_graph.add_edge(min(u, v),max(u,v))
    return directed_graph
