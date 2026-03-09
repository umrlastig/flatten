from itertools import groupby, takewhile
from typing import Any, Optional, Union
import shapely
from shapely import LinearRing, Point, LineString, Polygon
import geopandas as gpd
from functools import reduce
from pyinterpolate import inverse_distance_weighting


def find_projection(
    triangle: shapely.Polygon, segment: shapely.LineString
) -> shapely.LineString | None:
    """
    Projects the point not on the constrained segment to its projection on the constrained segment if it exists.

    :param triangle: a triangle
    :type triangle: shapely.Polygon
    :param segment: a constrained segment
    :type segment: shapely.LineString
    :return: the linestring connecting the point not on the constrained segment to its projection.
    :rtype: LineString | None
    """
    triangle_coordinates: list[list[float]] = shapely.get_coordinates(
        triangle, include_z=False
    ).tolist()
    segment_coordinates: list[list[float]] = shapely.get_coordinates(
        segment, include_z=False
    ).tolist()
    other_coordinate = next(
        filter(lambda c: c not in segment_coordinates, triangle_coordinates)
    )
    dist = segment.project(Point(*other_coordinate), normalized=True)
    if (dist > 0.0) and (dist < 1.0):
        projection: list[float] = shapely.get_coordinates(
            segment.interpolate(dist, normalized=True)
        ).tolist()[0]
        return LineString([projection, other_coordinate])
    return None


def get_bottlenecks(
    gdf_triangle: gpd.GeoDataFrame,
    gdf_triangle_segments: gpd.GeoDataFrame,
    id_col: str = "triangle_id",
) -> gpd.GeoDataFrame:
    """
    Compute the bottleneck dataframe.
    Project points from the triangles on the constrained segments ("constraint" == True).
    Keeps only the points on the constrained segments.

    :param gdf_triangle: input triangles
    :type gdf_triangle: gpd.GeoDataFrame
    :param gdf_triangle_segments: input triangle segments
    :type gdf_triangle_segments: gpd.GeoDataFrame
    :param id_col: column of the triangle index
    :type id_col: str
    :return: bottleneck dataframe
    :rtype: GeoDataFrame
    """
    # constrained segments
    constrained_seg = gdf_triangle_segments[gdf_triangle_segments["constraint"]].copy()
    constrained_seg = constrained_seg.rename(columns={"geometry": "segment_geom"})
    constrained_seg = constrained_seg.set_geometry("segment_geom")
    triangles = gdf_triangle[[id_col, "geometry"]].copy()
    triangles["triangle_geom"] = triangles.geometry.boundary
    triangles = triangles.set_geometry("triangle_geom")
    # Associate each constrained segment with its parent triangle
    print("constrained_seg=", constrained_seg)
    print("triangles=", triangles)
    assoc = gpd.sjoin(
        constrained_seg,
        triangles,
        how="inner",
        predicate="within",
    )
    print("assoc=", assoc)
    # After the join we have: index_left (segment), index_right (triangle)
    # assoc = assoc.rename(columns={ "index_right": id_col })
    assoc = assoc[[id_col, "segment_geom"]].copy()
    # Count constrained segments per triangle
    seg_counts = assoc.groupby(id_col).size().rename("constrained_cnt")
    # Attach the count back to the triangle GeoDataFrame
    triangles_cnt = gdf_triangle.merge(
        seg_counts, how="left", left_on=id_col, right_index=True
    )
    triangles_cnt["constrained_cnt"] = (
        triangles_cnt["constrained_cnt"].fillna(0).astype(int)
    )
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
    # def _proj(row) -> Optional[LineString]:
    #     """Wrapper that safely calls find_projection."""
    #     try:
    #         return find_projection(row["triangle_geom"], row["segment_geom"])
    #     except Exception:
    #         return None

    eligible_with_seg["geometry"] = eligible_with_seg.apply(
        lambda row: find_projection(row["triangle_geom"], row["segment_geom"]), axis=1
    )
    # Build the final GeoDataFrame
    result = eligible_with_seg[[id_col, "geometry"]].copy()
    return result.dropna(subset=["geometry"])


def get_profiles(
    gdf_segments: gpd.GeoDataFrame, line: LineString, direct: bool = True
) -> Union[
    tuple[
        list[tuple[tuple[float, float], bool, int | None]],
        list[tuple[tuple[float, float], bool, int | None]],
    ],
    None,
]:
    """
    Computes the left and right profiles for the input line using the given triangle segments.
    """
    intersecting_segments = gdf_segments.loc[gdf_segments.intersects(line)]
    intersecting_segments_geometry = intersecting_segments["geometry"]
    intersecting_segments_type = intersecting_segments["type"]
    intersecting_segments_shared = intersecting_segments["n_segments"] > 1
    intersections = [
        line.intersection(segment) for segment in intersecting_segments_geometry
    ]
    intersection_zipped = zip(
        intersections,
        intersecting_segments_geometry,
        intersecting_segments_type,
        intersecting_segments_shared,
    )

    def is_not_on_one_end(
        point: Point, segment: LineString, type: str, shared: bool
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
        intersecting_segments_shared,
    ) = zip(*intersection_zipped)
    # zip again with distance
    zipped: tuple[list[LineString], list[str], list[bool], list[Point], list[float]] = (
        zip(
            intersecting_segments_geometry,
            intersecting_segments_type,
            intersecting_segments_shared,
            intersections,
            distances,
        )
    )  # type: ignore
    # sort by distance
    sorted_intersections = sorted(
        zipped, key=lambda intersection: intersection[4], reverse=not direct
    )
    point: tuple[float, ...] = line.coords[0] if direct else line.coords[-1]

    def update(x, y: tuple[LineString, str, bool, Point, float]):
        # the accumulator x: (the previous point, the index of the current bottleneck, left accumulator, right accumulator)
        previous, index, left, right = x
        segment, type, shared, intersection, _ = y
        is_bottleneck = type == "bottleneck"
        edge_index = index if is_bottleneck else None
        if LinearRing([previous, intersection, segment.coords[0]]).is_ccw:
            left.append((segment.coords[0], shared, edge_index))
            right.append((segment.coords[1], shared, edge_index))
        else:
            left.append((segment.coords[1], shared, edge_index))
            right.append((segment.coords[0], shared, edge_index))
        return (intersection, index + 1 if is_bottleneck else index, left, right)

    _, _, left, right = reduce(update, sorted_intersections, (point, 0, [], []))  # type: ignore
    # removing consecutive identical points
    # def get_b(list: list[tuple[tuple[float, float], bool, int | None]]) -> list[int]:
    #     """
    #     Merge bottleneck information. e[2] is a bottleck index (int) or None.
    #     """
    #     return [e[2] for e in list if e[2] is not None]
    # def get_shared(list: list[tuple[tuple[float, float], bool, int | None]]) -> bool:
    #     """
    #     Merge shared information.
    #     """
    #     return any([e[1] for e in list])
    # final_left: list[tuple[tuple[float, float],bool,list[int]]] = [(key, get_shared(list(g)), get_b(list(g))) for key, g in groupby(left, key=lambda l: l[0])]
    # final_right: list[tuple[tuple[float, float],bool,list[int]]] = [(key, get_shared(list(g)), get_b(list(g))) for key, g in groupby(right, key=lambda l: l[0])]
    # return final_left, final_right
    return left, right


def merge_profile_points(
    points: list[tuple[tuple[float, float], bool, int | None]],
) -> list[tuple[tuple[float, float], bool, list[int]]]:
    # removing consecutive identical points
    def get_b(list: list[tuple[tuple[float, float], bool, int | None]]) -> list[int]:
        """
        Merge bottleneck information. e[2] is a bottleck index (int) or None.
        """
        return [e[2] for e in list if e[2] is not None]

    def get_shared(list: list[tuple[tuple[float, float], bool, int | None]]) -> bool:
        """
        Merge shared information.
        """
        return any([e[1] for e in list])

    def build(point:tuple[float, float], list: list[tuple[tuple[float, float], bool, int | None]]):
        return (point, get_shared(list), get_b(list))

    return [
        build(key, list(g)) for key, g in groupby(points, key=lambda l: l[0])
    ]


def split_by_first_false(
    seq: list[tuple[Any, bool, Any]], reverse: bool = False
) -> tuple[list[tuple[Any, bool, Any]], list[tuple[Any, bool, Any]]]:
    if reverse:
        seq.reverse()
    head = list(takewhile(lambda t: t[1] is True, seq))
    tail = seq[len(head) :]
    if reverse:
        head.reverse()
        tail.reverse()
    return head, tail
