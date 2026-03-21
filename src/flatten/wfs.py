import logging

from owslib.wfs import WebFeatureService
import geojson
import geopandas as gpd
from shapely import geometry

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.addHandler(logging.StreamHandler())

def get_wfs_data(url: str, type_name: str, box, srs) -> gpd.GeoDataFrame | None:
    # Specify the url for the backend.
    wfs20 = WebFeatureService(url=url, version="2.0.0")
    if wfs20:
        # Specify parameters (read data in json format) and fetch data from WFS using requests
        response = wfs20.getfeature(
            typename=type_name, bbox=box, srsname=srs, outputFormat="application/json"
        )
        # Create GeoDataFrame from geojson and set coordinate reference system
        return gpd.GeoDataFrame.from_features(geojson.loads(response.read()), crs=srs)
    return None


def get_hydro_data(box, srs):
    surface = get_wfs_data(
        "https://data.geopf.fr/wfs", "BDTOPO_V3:surface_hydrographique", box, srs
    )
    segment = get_wfs_data(
        "https://data.geopf.fr/wfs", "BDTOPO_V3:troncon_hydrographique", box, srs
    )
    nodes = get_wfs_data(
        "https://data.geopf.fr/wfs", "BDTOPO_V3:noeud_hydrographique", box, srs
    )
    if (surface is None) or (segment is None) or (nodes is None):
        logger.error("No surface or no segment found.")
        return None
    logger.debug(f"{len(surface)} Surfaces")
    logger.debug(f"{len(segment)} Segments")
    logger.debug(f"{len(nodes)} Nodes")
    surface = surface.query('nature == "Ecoulement naturel"')
    logger.debug(f"{len(surface)} mask")
    surface = surface.query('persistance == "Permanent"')
    logger.debug(f"{len(surface)} surface mask")
    box_as_polygon = geometry.box(box[0], box[1], box[2], box[3])
    surface = surface[surface.covered_by(box_as_polygon)]
    logger.debug(f"{len(surface)} surface mask")
    # segment = segment.query('nature == "Ecoulement naturel" OR nature == "Conduit buse"')
    segment = segment[
        (segment["nature"] == "Ecoulement naturel")
        | (segment["nature"] == "Conduit buse")
    ]
    logger.debug(f"{len(segment)} segment mask")
    segment = segment[
        segment["liens_vers_surface_hydrographique"].isin(surface["cleabs"])
        | (segment["nature"] == "Conduit buse")
    ]
    logger.debug(f"{len(segment)} segment mask")
    return (surface, segment, nodes)
