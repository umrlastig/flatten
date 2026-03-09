import typer
from typing_extensions import Annotated
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pandas as pd
import shapely
from shapely import Point, LineString, LinearRing
from shapely.geometry import shape
from itertools import chain, groupby
from functools import reduce

app = typer.Typer(
    name="pymatch", help="Pymatch matches geographical features", add_completion=True
)


def get_segments(triangles: gpd.GeoSeries):
    rings = [shapely.get_exterior_ring(triangle) for triangle in triangles]
    rings = [ring for ring in rings if ring is not None]
    list_of_lists = [
        [
            LineString(sorted([ring.coords[i], ring.coords[i + 1]]))
            for i in range(0, len(ring.coords) - 1)
        ]
        for ring in rings
    ]
    return list(chain(*list_of_lists))


@app.command()
def main(
    surface_file: Annotated[Path, typer.Argument(help="Surfaces")],
    segment_file: Annotated[Path, typer.Argument(help="Lines")],
    output_file: Annotated[Path, typer.Argument(help="File to save the results to")],
):
    print(datetime.now(), "Running")
    # read using geopandas to reuse the dataframes in the export
    gpd1 = gpd.read_file(surface_file)
    union = shapely.union_all(gpd1.geometry)
    triangles = shapely.constrained_delaunay_triangles(union)
    print("found triangles:", len(triangles.geoms))
    # use crs from input file
    geom_list = [p for p in triangles.geoms]
    df = pd.DataFrame({"triangle_id": list(range(0, len(geom_list)))})
    gdf = gpd.GeoDataFrame(df, geometry=geom_list, crs=gpd1.crs)
    gdf.to_file(output_file, layer="triangles", driver="GPKG")

    segments = list(set(get_segments(gdf.geometry)))
    gdf_segments = gpd.GeoDataFrame(
        pd.DataFrame({"segment_id": list(range(0, len(segments)))}),
        geometry=segments,
        crs=gpd1.crs,
    )
    gdf_segments.to_file(output_file, layer="segments", driver="GPKG")

    gpd2 = gpd.read_file(segment_file)
    gpd2 = gpd2[gpd2["FICTIF"] == "Oui"]
    segment_id = []
    order = []
    position = []
    geoms = []
    for feature in gpd2.iterfeatures():
        line: LineString = shape(feature["geometry"])  # type: ignore
        direct = feature["properties"]["SENS_ECOUL"] == "Sens direct"
        line.geom_type
        intersecting_segments = gdf_segments.loc[gdf_segments.intersects(line)][
            "geometry"
        ]
        intersections = [
            line.intersection(segment) for segment in intersecting_segments
        ]
        intersection_zipped = zip(intersecting_segments, intersections)
        intersection_zipped = [
            intersection
            for intersection in intersection_zipped
            if intersection[1].geom_type == "Point"
        ]
        # TODO handle MultiPoints?
        if intersection_zipped:
            # print(intersection_zipped)
            distances = [
                shapely.line_locate_point(line, p[1]) for p in intersection_zipped
            ]
            # unzip
            intersecting_segments, intersections = zip(*intersection_zipped)
            # zip again
            zipped: tuple[list[LineString], list[Point], list[float]] = zip(
                intersecting_segments, intersections, distances
            )  # type: ignore
            # sort by distance
            sorted_intersections = sorted(
                zipped, key=lambda intersection: intersection[2], reverse=not direct
            )
            point: tuple[float, ...] = line.coords[0] if direct else line.coords[-1]

            def update(
                x: tuple[
                    tuple[float, ...], list[tuple[float, ...]], list[tuple[float, ...]]
                ],
                y: tuple[LineString, Point, float],
            ):
                previous, left, right = x
                segment, intersection, _ = y
                if LinearRing([previous, intersection, segment.coords[0]]).is_ccw:
                    left.append(segment.coords[0])
                    right.append(segment.coords[1])
                else:
                    left.append(segment.coords[1])
                    right.append(segment.coords[0])
                return (intersection, left, right)

            _, left, right = reduce(update, sorted_intersections, (point, [], []))  # type: ignore
            # removing consecutive identical points
            left = [key for key, _ in groupby(left)]
            right = [key for key, _ in groupby(right)]
            for i, l in enumerate(left):
                segment_id.append(feature["id"])
                order.append(i)
                position.append("left")
                geoms.append(Point(l))
            for i, r in enumerate(right):
                segment_id.append(feature["id"])
                order.append(i)
                position.append("right")
                geoms.append(Point(r))

    gdf_points = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "point_id": list(range(0, len(geoms))),
                "segment_id": segment_id,
                "order": order,
                "position": position,
            }
        ),
        geometry=geoms,
        crs=gpd1.crs,
    )
    gdf_points.to_file(output_file, layer="points", driver="GPKG")

    print(datetime.now(), "Done")


if __name__ == "__main__":
    app()
