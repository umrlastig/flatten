from flatten.utils import get_triangles
from flatten.wfs import get_hydro_data
from flatten.triangle_graph import get_triangle_graph_as_nx
from shapely import LineString
import geopandas as gpd
import networkx as nx
from networkx.algorithms.dag import descendants
from typing import List, Set, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.addHandler(logging.StreamHandler())


def cycle_sublist(cycle: List[Any], start: int, end: int) -> List[Any]:
    """Create a sublist of cycle from `start` to `end` (included)."""
    n = len(cycle)
    return [cycle[(start + i) % n] for i in range((n + end + 1 - start) % n)]


def split_cycle(cycle: List[Any], split_nodes: Set[Any]) -> List[List[Any]]:
    """Return segments of a circular list split at the `split_nodes`."""
    split_indices = sorted([cycle.index(n) for n in split_nodes])
    return [
        cycle_sublist(cycle, start, end)
        for (start, end) in zip(split_indices, split_indices[1:] + split_indices[:1])
    ]


def add_path(G: nx.MultiDiGraph, sub_path: List[Any], reverse: bool):
    """Add the path (a list of nodes) to the graph. Reverse it if `reverse`."""
    if reverse:
        sub_path.reverse()
    for i in range(len(sub_path) - 1):
        if (not G.has_edge(sub_path[i], sub_path[i + 1])) & (
            not G.has_edge(sub_path[i + 1], sub_path[i])
        ):
            G.add_edge(
                sub_path[i],
                sub_path[i + 1],
                edge_ids=[],
                geometry=LineString(
                    [
                        G.nodes[sub_path[i]]["geometry"].centroid,
                        G.nodes[sub_path[i + 1]]["geometry"].centroid,
                    ]
                ),
            )


def get_nodes_accessible_from(
    G: nx.MultiDiGraph, node: Any, cycle_nodes: Set[Any]
) -> int:
    """Get the number of nodes accessible from `node` outside of the cycle"""
    if len(set(G.successors(node)).difference(cycle_nodes)) == 0:
        return 0
    return len(descendants(G, node).difference(cycle_nodes))


def get_oriented_graph(
    gdf_triangle: gpd.GeoDataFrame, gdf_segment: gpd.GeoDataFrame
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, bool]:
    """
    Create an oriented multi graph from the the hydrographic segments and the surfaces.
    """
    # triangle_graph_gdf = get_triangle_graph_as_gdf(gdf_triangle)
    triangle_graph = get_triangle_graph_as_nx(gdf_triangle)
    gdf_segment["_edge_idx"] = gdf_segment.index  # just in case the index isn’t numeric
    # Perform a spatial intersection.  The result will have one row per
    # (edge, triangle) pair where they overlap, and the geometry will be the
    # *segment of the edge that lies inside the triangle*.
    intersections = gpd.overlay(
        gdf_segment[["cleabs", "geometry", "_edge_idx"]],
        gdf_triangle[["triangle_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    def fragment_position(row):
        # Retrieve the original full edge geometry (we stored it in the overlay)
        edge_geom: LineString = gdf_segment.loc[row["_edge_idx"], "geometry"]  # type: ignore
        # Use the centroid of the fragment as a stable representative point
        pt = row["geometry"].centroid
        # Distance from the start of the edge to the point
        dist_along = edge_geom.project(pt)
        # Normalise by total length (avoid division by zero)
        return dist_along / edge_geom.length if edge_geom.length != 0 else 0.0

    # Apply the function row‑wise
    intersections["pos_along"] = intersections.apply(fragment_position, axis=1)
    # Sort by edge, then by the relative position along that edge
    intersections_sorted = intersections.sort_values(
        ["cleabs", "pos_along"]
    ).reset_index(drop=True)
    tri_lists = (
        intersections_sorted.groupby("cleabs")["triangle_id"]
        .apply(list)
        .reset_index(name="tri_sequence")
    )
    G = nx.MultiDiGraph()
    # Add all triangle IDs as nodes (optional – NetworkX will auto‑add them)
    G.add_nodes_from(gdf_triangle["triangle_id"])
    # attach geometry of each triangle as node attribute
    tri_geom_dict = dict(zip(gdf_triangle["triangle_id"], gdf_triangle["geometry"]))
    nx.set_node_attributes(G, tri_geom_dict, name="geometry")
    # Iterate over each edge’s ordered triangle list
    for _, row in tri_lists.iterrows():
        edge_id = row["cleabs"]
        seq = row["tri_sequence"]
        # Create directed arcs between successive triangles
        for src, dst in zip(seq[:-1], seq[1:]):
            # If you want to keep track of *which* original edge created this arc:
            if G.has_edge(src, dst):
                # Append to a list of edge_ids that share the same arc
                for key in G[src][dst].keys():
                    G[src][dst][key]["edge_ids"].append(edge_id)
            else:
                # check if the opposite edge exists
                # TODO sort principal segments first?
                if not G.has_edge(dst, src):
                    src_geometry = G.nodes[src]["geometry"]
                    dst_geometry = G.nodes[dst]["geometry"]
                    G.add_edge(
                        src,
                        dst,
                        edge_ids=[edge_id],
                        geometry=LineString(
                            [src_geometry.centroid, dst_geometry.centroid]
                        ),
                    )
    logger.info(f"Number of triangle nodes: {G.number_of_nodes()}")
    logger.info(f"Number of directed arcs : {G.number_of_edges()}")
    # find the cycles in the triangle graph and connect them to the directed graph
    connected_nodes = set(filter(lambda n: len(list(G.neighbors(n))) > 0, G.nodes()))
    triangle_graph_cycles = sorted(list(nx.simple_cycles(triangle_graph)), key=len)
    for cycle in triangle_graph_cycles:
        shared = set(cycle).intersection(connected_nodes)
        if len(shared) < len(cycle):
            if len(shared) > 0:
                # we split at the points with 3 edges that belong to both graphs
                split_nodes = set(
                    filter(lambda n: len(list(triangle_graph.neighbors(n))) > 2, cycle)
                ).intersection(connected_nodes)
                sub_paths = split_cycle(cycle, split_nodes)
                for sub_path in sub_paths:
                    sub_path_shared = set(sub_path).intersection(connected_nodes)
                    if len(sub_path_shared) < len(sub_path):
                        # there are unshared nodes
                        # logger.debug(f"sub_path: {sub_path}")
                        start = sub_path[0]
                        # find out if start has successors outside the cycle
                        desc_start = get_nodes_accessible_from(G, start, set(cycle))
                        end = sub_path[-1]
                        # find out if end has successors outside the cycle
                        desc_end = get_nodes_accessible_from(G, end, set(cycle))
                        # logger.debug(f"start: {start} => {desc_start}")
                        # logger.debug(f"end: {end} => {desc_end}")
                        add_path(G, sub_path, desc_start >= desc_end)

    connected_nodes = set(filter(lambda n: len(list(G.neighbors(n))) > 0, G.nodes()))
    # connect the remaining edges from the triangle graph starting from the ones connected to the directed graph
    while len(connected_nodes) < len(G.nodes):
        edges = [
            (a, b)
            for (a, b) in triangle_graph.edges
            if (a in connected_nodes) ^ (b in connected_nodes)
            # a or b has to be connected but not both
        ]
        for a, b in edges:
            start = a if a in connected_nodes else b
            end = b if a in connected_nodes else a
            add_path(G, [start, end], True)
            connected_nodes.add(end)

    records = []
    for u, v, data in G.edges(data=True):
        line = data["geometry"]
        records.append(
            {
                "source": u,
                "target": v,
                "edge_ids": data.get("edge_ids", []),  # list of original edge IDs
                "geometry": line,
            }
        )
    # Create the GeoDataFrame
    edge_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf_triangle.crs)
    cycles = list(nx.simple_cycles(G))
    return G, edge_gdf, len(cycles) == 0


def main():
    srs = "urn:ogc:def:crs:EPSG::2154"
    box = (1036535, 6289927, 1042268, 6305786, srs)
    output_file = "triangle_graph.gpkg"
    (surfaces, gdf_segment, _) = get_hydro_data(box, srs)
    if (surfaces is None) or (gdf_segment is None):
        logger.error("no data")
        return
    triangles = get_triangles(surfaces, 20.0)
    _, edge_gdf, has_no_cycle = get_oriented_graph(triangles, gdf_segment)
    edge_gdf.to_file(output_file, layer="edges")
    assert has_no_cycle  # make sure there is no cycle
    logger.debug("All done!")


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    logger.addHandler(logging.StreamHandler())
    main()
