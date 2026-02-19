from itertools import groupby
from typing import Optional, Union
import shapely
from shapely import LinearRing, Point, LineString, Polygon
import geopandas as gpd
from functools import reduce
from pyinterpolate import inverse_distance_weighting

def find_projection(triangle: shapely.Polygon, segment: shapely.LineString) -> shapely.LineString | None:
    triangle_coordinates: list[list[float]] = shapely.get_coordinates(triangle, include_z=False).tolist()
    segment_coordinates: list[list[float]] = shapely.get_coordinates(segment, include_z=False).tolist()
    other_coordinate = next(filter(lambda c: c not in segment_coordinates, triangle_coordinates))
    dist = segment.project(Point(*other_coordinate), normalized=True)
    if (dist > 0.) and (dist < 1.):
        projection: list[float] = shapely.get_coordinates(segment.interpolate(dist, normalized=True)).tolist()[0]
        return LineString([projection, other_coordinate])
        # proj_height = inverse_distance_weighting(
        #     unknown_location=projection,
        #     known_values=elevations,
        #     known_geometries=sample_points,
        #     power=2.
        # )
        # other_height = inverse_distance_weighting(
        #     unknown_location=other_coordinate,
        #     known_values=elevations,
        #     known_geometries=sample_points,
        #     power=2.
        # )
        # return LineString([(*projection, proj_height), (*other_coordinate, other_height)])
    return None

def get_bottlenecks(gdf_triangle: gpd.GeoDataFrame, gdf_triangle_segments: gpd.GeoDataFrame, id_col:str = "triangle_id"):
    # constrained segments
    constrained_seg = gdf_triangle_segments[gdf_triangle_segments["constraint"]].copy()
    constrained_seg = constrained_seg.rename(columns={"geometry": "segment_geom"})
    constrained_seg = constrained_seg.set_geometry("segment_geom")
    triangles = gdf_triangle[[id_col, "geometry"]].copy()
    triangles["triangle_geom"] = triangles.geometry.boundary
    triangles = triangles.set_geometry("triangle_geom")
    # Associate each constrained segment with its parent triangle
    print("constrained_seg=",constrained_seg)
    print("triangles=",triangles)
    assoc = gpd.sjoin(
        constrained_seg,
        triangles,
        how="inner",
        predicate="within",
    )
    print("assoc=",assoc)
    # After the join we have: index_left (segment), index_right (triangle)
    # assoc = assoc.rename(columns={ "index_right": id_col })
    assoc = assoc[[id_col, "segment_geom"]].copy()
    # Count constrained segments per triangle
    seg_counts = (
        assoc.groupby(id_col)
        .size()
        .rename("constrained_cnt")
    )
    # Attach the count back to the triangle GeoDataFrame
    triangles_cnt = gdf_triangle.merge(seg_counts, how="left", left_on=id_col, right_index=True)
    triangles_cnt["constrained_cnt"] = triangles_cnt["constrained_cnt"].fillna(0).astype(int)
    # Keep only triangles with exactly ONE constrained segment
    mask_one = triangles_cnt["constrained_cnt"] == 1
    eligible_tri = triangles_cnt.loc[mask_one, [id_col, "geometry"]].copy()
    eligible_tri = eligible_tri.rename(columns={"geometry": "triangle_geom"})
    # Pull the *single* constrained segment for each eligible triangle
    eligible_with_seg = eligible_tri.merge(
        assoc,
        how="left",
        on=id_col,
    )  # columns: triangle_id, triangle_geom, segment_geom
    # Compute the bottleneck (projection) for each pair
    def _proj(row) -> Optional[LineString]:
        """Wrapper that safely calls find_projection."""
        try:
            return find_projection(row["triangle_geom"], row["segment_geom"])
        except Exception:
            return None
    eligible_with_seg["geometry"] = eligible_with_seg.apply(_proj, axis=1)
    # Build the final GeoDataFrame
    result = eligible_with_seg[[id_col, "geometry"]].copy()
    return result.dropna(subset=["geometry"])

def get_profiles(gdf_segments: gpd.GeoDataFrame, line: LineString, direct: bool = True) -> Union[tuple[list[tuple[tuple[float, float],list[int]]], list[tuple[tuple[float, float],list[int]]]], None]:
    intersecting_segments = gdf_segments.loc[gdf_segments.intersects(line)]
    intersecting_segments_geometry = intersecting_segments["geometry"]
    intersecting_segments_type = intersecting_segments["type"]
    intersections = [line.intersection(segment) for segment in intersecting_segments_geometry]
    intersection_zipped = zip(intersections, intersecting_segments_geometry, intersecting_segments_type)
    def is_not_on_one_end(point: Point, segment: LineString, _: str) -> bool:
        """
        True in the intersection is a point and it is not one of the endpoints of the segment.
        """
        return point.geom_type == "Point" and point.distance(segment.boundary) > 0
    intersection_zipped = [intersection for intersection in intersection_zipped if is_not_on_one_end(*intersection)]
    # TODO handle MultiPoints?
    if not intersection_zipped:
        return None
    distances = [shapely.line_locate_point(line, p[0]) for p in intersection_zipped]
    # unzip
    intersections, intersecting_segments_geometry, intersecting_segments_type = zip(*intersection_zipped)
    # zip again with distance
    zipped: tuple[list[LineString], list[str], list[Point], list[float]] = zip(intersecting_segments_geometry, intersecting_segments_type, intersections, distances) # type: ignore
    # sort by distance
    sorted_intersections = sorted(zipped, key=lambda intersection: intersection[3], reverse=not direct)
    point: tuple[float, ...] = line.coords[0] if direct else line.coords[-1]
    def update(x, y: tuple[LineString, str, Point, float]):
        previous, index, left, right = x
        segment, type, intersection, _ = y
        is_bottleneck = (type == "bottleneck")
        edge_index = index if is_bottleneck else None
        if LinearRing([previous, intersection, segment.coords[0]]).is_ccw:
            left.append((segment.coords[0], edge_index))
            right.append((segment.coords[1], edge_index))
        else:
            left.append((segment.coords[1], edge_index))
            right.append((segment.coords[0], edge_index))
        return (intersection, index + 1 if is_bottleneck else index, left, right)
    _, _, left, right = reduce(update, sorted_intersections, (point, 0, [], [])) # type: ignore
    # removing consecutive identical points
    def get_b(list: list[tuple[tuple[float, float], int | None]]) -> list[int]:
        res = [e[1] for e in list if e[1] is not None]
        print(list)
        print("\t",res)
        return res
    final_left: list[tuple[tuple[float, float],list[int]]] = [(key, get_b(list(g))) for key, g in groupby(left, key=lambda l: l[0])] # type: ignore
    final_right: list[tuple[tuple[float, float],list[int]]] = [(key, get_b(list(g))) for key, g in groupby(right, key=lambda l: l[0])]
    return final_left, final_right